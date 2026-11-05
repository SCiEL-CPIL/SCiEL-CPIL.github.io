import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry enforcement:
    # The cell must be symmetric under D4.
    # We can work on the top-left 16x16 quadrant and mirror.
    # Indices 0..15 map to 0..15, 16..31 map to 15..0 (reversed).
    # Specifically: cell[i][j] = cell[size-1-i][size-1-j] (180 rot)
    # cell[i][j] = cell[j][i] (transpose)
    # cell[i][j] = cell[i][size-1-j] (reflection x)
    # So we define the quadrant Q (16x16) and expand.
    
    def expand_quadrant(Q):
        # Q is 16x16
        # Top-left: Q
        # Top-right: Q[:, ::-1]
        # Bottom-left: Q[::-1, :]
        # Bottom-right: Q[::-1, ::-1]
        top = np.hstack((Q, Q[:, ::-1]))
        bottom = np.hstack((Q[::-1, :], Q[::-1, ::-1]))
        return np.vstack((top, bottom))

    def contract_cell(cell):
        return cell[:16, :16]

    # PWE Solver
    # n_g = 8 -> 17x17 G-vectors. Total 289.
    # This is fixed by the evaluator description ("289-plane-wave Galerkin method").
    n_g = 8
    G_vecs = []
    for nx in range(-n_g, n_g+1):
        for ny in range(-n_g, n_g+1):
            G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
    N_G = len(G_vecs)
    
    # Pre-compute G indices for epsilon coupling
    # eps_coeffs[m, n] corresponds to G_m - G_n
    # We need to map (nx, ny) differences to indices in the 17x17 grid.
    # idx = (nx + n_g) * (2*n_g+1) + (ny + n_g)
    
    def get_fourier_coefficients(cell):
        """
        Compute Fourier coefficients of epsilon.
        cell is 32x32 binary or continuous.
        """
        coeffs = np.zeros((2*n_g+1, 2*n_g+1), dtype=complex)
        # Use FFT for speed? 32x32 is small, direct sum is fine.
        # Or use the sinc formula for pixels.
        # Let's use the exact pixel integration formula for accuracy.
        # eps(x,y) = sum c_G exp(i G . r)
        # c_G = 1/A sum_I eps(x_I, y_I) exp(-i G . r_I)
        # For square pixels, the integral is:
        # c_G = 1/A sum_I eps_I (2/Gx sin(Gx/2) e^{-i Gx (x_I+0.5)}) (2/Gy sin(Gy/2) e^{-i Gy (y_I+0.5)})
        # for Gx, Gy != 0.
        
        M = np.array(cell, dtype=float)
        
        # Precompute pixel weights
        # weights[nx, ny, i, j]
        # This might be memory heavy if not careful, but 17*17*32*32 is ~300k complex numbers. OK.
        # Actually, we can compute term_x and term_y separately.
        
        term_x = np.zeros((size, 2*n_g+1), dtype=complex)
        term_y = np.zeros((size, 2*n_g+1), dtype=complex)
        
        for n in range(-n_g, n_g+1):
            idx = n + n_g
            G = 2 * np.pi * n
            if G == 0:
                term_x[:, idx] = 1.0 / size
                term_y[:, idx] = 1.0 / size
            else:
                sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                base_exp = np.exp(-1j * G * 0.5 / size)
                i_vals = np.arange(size)
                exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                term_x[:, idx] = sinc_factor * exp_terms
                term_y[:, idx] = sinc_factor * exp_terms
        
        # f_coeffs = sum_{i,j} M[i,j] term_x[i, nx] term_y[j, ny]
        # f_coeffs[nx, ny] = sum_i sum_j M[i,j] term_x[i, nx] term_y[j, ny]
        # This is equivalent to term_x.T @ M @ term_y
        
        f_coeffs = term_x.T @ (M @ term_y)
        
        delta_eps = eps_high - eps_low
        eps_coeffs = f_coeffs * delta_eps
        eps_coeffs[n_g, n_g] += eps_low
        
        return eps_coeffs

    def solve_bands(eps_coeffs, k_points):
        """
        Solve for bands at given k points.
        Returns list of arrays of omega values.
        """
        all_omegas = []
        all_modes = [] # Store eigenvectors for gradient computation
        
        # Pre-build E indices
        E_indices = np.zeros((N_G, N_G), dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
                else:
                    E_indices[idx_g, idx_gp] = -1
        
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            A = np.zeros((N_G, N_G), dtype=complex)
            A[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            # TM Eigenproblem:
            # sum_j eps_{G_i-G_j} (k+G_i).(k+G_j) c_j = omega^2 c_i
            # Let B_{ij} = eps_{G_i-G_j} (k+G_i).(k+G_j)
            # We can write this as A_mat @ c = omega^2 c
            # Where A_mat is constructed from A and D.
            # Note: The matrix is Hermitian.
            
            # Efficient construction:
            # A_mat[i,j] = A[i,j] * (kx_i*kx_j + ky_i*ky_j)
            # This is not a simple diagonal scaling.
            
            # Let's compute (k+G) vectors
            kG_x = np.array([kx + G[0] for G in G_vecs])
            kG_y = np.array([ky + G[1] for G in G_vecs])
            
            # Outer product term
            # T[i,j] = kG_x[i]*kG_x[j] + kG_y[i]*kG_y[j]
            # A_mat = A * T (element-wise)
            
            # To save memory/time, we can form the matrix directly.
            # T = np.outer(kG_x, kG_x) + np.outer(kG_y, kG_y)
            # A_mat = A * T
            
            # However, A is sparse-ish (banded in G space). T is dense.
            # N_G=289. 289x289 matrix is small.
            
            T = np.outer(kG_x, kG_x) + np.outer(kG_y, kG_y)
            A_mat = A * T
            
            # Hermitian symmetrize
            A_mat = (A_mat + A_mat.conj().T) / 2.0
            
            try:
                # Solve eigenvalue problem
                # np.linalg.eigh for Hermitian
                w2, v = np.linalg.eigh(A_mat)
                w2 = np.real(w2)
                v = np.real(v) # Eigenvectors can be chosen real if matrix is real?
                # A is complex? eps_coeffs are complex.
                # But for symmetric real epsilon, the matrix should be Hermitian.
                # If epsilon is real, eps_{-G} = eps_G^*.
                # The matrix A_mat is Hermitian.
                # The eigenvectors can be complex.
                
                # Filter valid bands
                valid_idx = w2 > 1e-10
                w2_valid = w2[valid_idx]
                v_valid = v[:, valid_idx]
                
                omegas = 1.0 / np.sqrt(w2_valid)
                omegas_sorted = np.sort(omegas)
                # Need to sort modes accordingly
                sort_idx = np.argsort(omegas)
                omegas_sorted = omegas[sort_idx]
                v_sorted = v_valid[:, sort_idx]
                
                all_omegas.append(omegas_sorted)
                all_modes.append(v_sorted)
                
            except Exception:
                continue
        
        return all_omegas, all_modes

    def compute_gap_and_gradients(eps_coeffs, k_points):
        """
        Compute gap and gradients for the objective.
        Objective: Maximize omega_5_min - omega_4_max.
        """
        all_omegas, all_modes = solve_bands(eps_coeffs, k_points)
        
        if not all_omegas:
            return 0.0, None, None, None, None
            
        # Find max omega_4 and min omega_5
        max_omega4 = 0.0
        min_omega5 = float('inf')
        k_idx_max4 = -1
        k_idx_min5 = -1
        
        for i, omegas in enumerate(all_omegas):
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                if w4 > max_omega4:
                    max_omega4 = w4
                    k_idx_max4 = i
                if w5 < min_omega5:
                    min_omega5 = w5
                    k_idx_min5 = i
        
        if k_idx_max4 == -1 or k_idx_min5 == -1:
            return 0.0, None, None, None, None
            
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        ratio = gap / midgap if midgap > 0 else 0.0
        
        # Compute gradients
        # d(omega)/d(eps_m) = (1/(2*omega)) * d(omega^2)/d(eps_m)
        # d(omega^2)/d(eps_m) = (v^* (dA/deps_m) v) / (v^* v)
        # dA/deps_m has non-zero elements at (i,j) where G_i - G_j = G_m.
        # Value is (k+G_i).(k+G_j).
        
        # We need gradients for omega_4 at k_idx_max4 and omega_5 at k_idx_min5.
        # grad_obj = grad_omega5 - grad_omega4
        
        grad_omega4 = np.zeros((2*n_g+1, 2*n_g+1), dtype=float)
        grad_omega5 = np.zeros((2*n_g+1, 2*n_g+1), dtype=float)
        
        def compute_grad_for_band(k_idx, band_idx, grad_target):
            if k_idx == -1:
                return
            kx, ky = k_points[k_idx]
            v = all_modes[k_idx][:, band_idx] # Eigenvector for the band
            w2 = all_omegas[k_idx][band_idx] ** 2
            omega = all_omegas[k_idx][band_idx]
            
            # v is complex
            v_conj = np.conj(v)
            
            # kG vectors
            kG_x = np.array([kx + G[0] for G in G_vecs])
            kG_y = np.array([ky + G[1] for G in G_vecs])
            
            # We need to sum over i, j: v_i^* v_j (k+G_i).(k+G_j) delta(G_i-G_j, m)
            # This is equivalent to:
            # For each m, sum_{i,j: G_i-G_j=G_m} v_i^* v_j T_{ij}
            
            # Efficient way:
            # Let W = v^* T v ? No.
            # The term is v^* (T * Delta_m) v where Delta_m is the matrix with 1s where G_i-G_j=G_m.
            
            # We can iterate over m.
            # For each m (nx, ny), the condition is G_i - G_j = G_m.
            # This means j = i - m.
            # So we sum over i: v_i^* v_{i-m} T_{i, i-m}.
            
            # Let's construct the vector v and T row by row?
            # N_G is 289. m has 289 values.
            # Looping over m and summing over i is 289*289 ~ 80k ops. Fast enough.
            
            for idx_m, (Gx_m, Gy_m, nx_m, ny_m) in enumerate(G_vecs):
                # Find pairs (i, j) such that G_i - G_j = G_m
                # j = i - m
                # We can just iterate i and check if i-m is a valid index.
                # Or precompute mapping.
                
                # Let's use the precomputed E_indices logic in reverse?
                # E_indices[i, j] = idx_m if G_i - G_j = G_m.
                # So for a fixed m, we need all (i,j) where E_indices[i,j] == idx_m.
                
                # This is a sparse structure.
                # Let's just loop i and find j.
                # G_i = (nx_i, ny_i). G_j = (nx_i - nx_m, ny_i - ny_m).
                # j exists if nx_i - nx_m in [-n_g, n_g] and ny_i - ny_m in [-n_g, n_g].
                
                sum_val = 0.0
                for i in range(N_G):
                    nx_i, ny_i = G_vecs[i][2], G_vecs[i][3]
                    nx_j = nx_i - nx_m
                    ny_j = ny_i - ny_m
                    if -n_g <= nx_j <= n_g and -n_g <= ny_j <= n_g:
                        # Find j
                        # j = (nx_j + n_g) * (2*n_g+1) + (ny_j + n_g)
                        j = (nx_j + n_g) * (2*n_g+1) + (ny_j + n_g)
                        # T_{ij} = kG_x[i]*kG_x[j] + kG_y[i]*kG_y[j]
                        T_ij = kG_x[i]*kG_x[j] + kG_y[i]*kG_y[j]
                        sum_val += v_conj[i] * v[j] * T_ij
                
                # grad = (1/(2*omega)) * (sum_val / (v^* v))
                # v is normalized by eigh? np.linalg.eigh returns normalized eigenvectors (norm 1).
                # So v^* v = 1.
                grad_target[nx_m + n_g, ny_m + n_g] = (1.0 / (2.0 * omega)) * sum_val
        
        compute_grad_for_band(k_idx_max4, 3, grad_omega4)
        compute_grad_for_band(k_idx_min5, 4, grad_omega5)
        
        # Objective: Maximize min_omega5 - max_omega4
        # Gradient of objective wrt eps: grad_omega5 - grad_omega4
        grad_obj = grad_omega5 - grad_omega4
        
        return ratio, grad_obj, max_omega4, min_omega5, k_points

    # Optimization Loop
    # Initialize epsilon field
    # Start with a simple dielectric square or circle
    # Let's start with a central dielectric square of size 16x16 (half the cell)
    # This is a valid symmetric start.
    
    Q = np.zeros((16, 16), dtype=float)
    # Central block
    Q[4:12, 4:12] = 1.0
    
    cell = expand_quadrant(Q)
    
    # K-points for optimization
    # Use a subset of the evaluator's k-points for speed, but ensure coverage.
    # The evaluator uses 30 points. We can use the same or a subset.
    # Let's use the same 30 points for consistency with the score.
    k_points_opt = []
    for i in range(10):
        k_points_opt.append((0, i / 9.0 * np.pi))
    for i in range(10):
        k_points_opt.append((i / 9.0 * np.pi, np.pi))
    for i in range(1, 5):
        k_points_opt.append((i / 4.0 * np.pi, i / 4.0 * np.pi))
    k_points_opt.append((np.pi/4, np.pi/2))
    k_points_opt.append((np.pi/3, 2*np.pi/3))
    k_points_opt.append((np.pi/6, np.pi/3))
    k_points_opt.append((np.pi/5, 3*np.pi/5))
    k_points_opt.append((np.pi/2.5, 7*np.pi/10))
    
    best_ratio = -1.0
    best_cell = cell
    best_Q = Q.copy()
    
    # Gradient Descent
    learning_rate = 0.05
    momentum = 0.0
    v_grad = np.zeros_like(Q)
    
    n_iter = 0
    while time.time() < deadline - 10:
        n_iter += 1
        
        eps_coeffs = get_fourier_coefficients(cell)
        ratio, grad_obj, max_omega4, min_omega5, _ = compute_gap_and_gradients(eps_coeffs, k_points_opt)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            best_Q = Q.copy()
            
        if grad_obj is None:
            break
            
        # Project gradient to quadrant
        # grad_obj is 17x17 Fourier coeffs.
        # We need gradient in real space.
        # grad_eps(x,y) = sum_G grad_G exp(i G . r)
        
        # Compute real-space gradient
        grad_real = np.zeros((size, size), dtype=float)
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                g_val = grad_obj[nx+n_g, ny+n_g]
                if abs(g_val) < 1e-10:
                    continue
                Gx = 2 * np.pi * nx
                Gy = 2 * np.pi * ny
                # Add to grad_real
                # exp(i (Gx x + Gy y))
                # x = (i+0.5)/size - 0.5
                # y = (j+0.5)/size - 0.5
                # This is O(N_G * size^2) = 289 * 1024 ~ 300k ops. Fast.
                
                # Vectorized:
                # i_vals = np.arange(size)
                # x = (i_vals + 0.5)/size - 0.5
                # y = (i_vals + 0.5)/size - 0.5
                # term = np.exp(1j * (Gx[:, None] * x[None, :] + Gy[:, None] * y[None, :]))
                # grad_real += g_val * term.real
                
                # To save time, precompute exponentials?
                # Or just do it.
                
        # Actually, doing this loop inside the gradient step is expensive if done naively.
        # Let's precompute the basis functions for the real space gradient.
        # Basis[i, j, idx] = exp(i G_idx . r_ij)
        
        # Re-initializing basis every iteration is bad.
        # Let's move basis computation outside.
        
        if n_iter == 1:
            i_vals = np.arange(size)
            x = (i_vals + 0.5)/size - 0.5
            y = (i_vals + 0.5)/size - 0.5
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            basis = np.zeros((size, size, N_G), dtype=complex)
            for idx, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                basis[:, :, idx] = np.exp(1j * (Gx * X + Gy * Y))
                
        # Now compute grad_real
        # grad_real = sum_g grad_g * basis_g
        # grad_obj is 17x17. Flatten it to match G_vecs order.
        # G_vecs order: nx from -8 to 8, ny from -8 to 8.
        # grad_obj index: (nx+8, ny+8).
        # So flatten grad_obj in row-major order (nx major) matches G_vecs if G_vecs is nested nx then ny.
        # Yes: for nx in ... for ny in ...
        
        grad_flat = grad_obj.flatten()
        grad_real = np.einsum('ijg,g->ij', basis, grad_flat).real
        
        # Project to quadrant
        grad_Q = grad_real[:16, :16]
        
        # Update
        v_grad = momentum * v_grad + (1 - momentum) * grad_Q
        Q = Q + learning_rate * v_grad
        
        # Threshold or keep continuous?
        # Topology optimization often uses a smooth filter.
        # Here, we just keep it continuous and threshold at the end.
        # However, the solver expects binary? No, get_fourier_coefficients takes float.
        # The evaluator takes binary.
        # We should probably threshold periodically or use a sigmoid.
        # Let's just update Q and clamp to [0,1] to avoid extreme values.
        Q = np.clip(Q, 0.0, 1.0)
        
        cell = expand_quadrant(Q)
        
        # If time is low, break
        if time.time() > deadline - 20:
            break
            
    # Final Thresholding
    # The best_cell was stored as continuous?
    # We stored best_cell from `cell.copy()` which is continuous.
    # We need to return a binary cell.
    # Let's threshold the best_Q.
    
    # It's possible that a continuous field yields a better gap in the PWE sense,
    # but the evaluator uses binary.
    # We should evaluate the binary version of the best candidate.
    
    # Let's collect a few candidates:
    # 1. Thresholded best_Q
    # 2. Thresholded current Q
    
    candidates = []
    
    # Threshold best_Q
    Q_bin = (best_Q > 0.5).astype(int)
    cell_bin = expand_quadrant(Q_bin)
    candidates.append(cell_bin)
    
    # Threshold current Q
    Q_bin_curr = (Q > 0.5).astype(int)
    cell_bin_curr = expand_quadrant(Q_bin_curr)
    candidates.append(cell_bin_curr)
    
    # Evaluate them exactly
    best_score = -1.0
    best_final_cell = cell_bin
    
    for c in candidates:
        if time.time() > deadline - 5:
            break
        eps_coeffs = get_fourier_coefficients(c)
        ratio, _, _, _, _ = compute_gap_and_gradients(eps_coeffs, k_points_opt)
        if ratio > best_score:
            best_score = ratio
            best_final_cell = c
            
    return best_final_cell.tolist()
