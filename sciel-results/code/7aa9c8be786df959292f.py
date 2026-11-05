"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Generate candidate cells using four-quadrant rod configurations.
    3. Use field-intensity-driven optimization to refine the best candidate.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210.0  # 20s margin before 230s kill

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=15, return_eigvecs=False):
        """
        Compute TM band structure using plane-wave expansion.
        Returns min_omega5 and max_omega4 over the BZ path.
        If return_eigvecs, also returns eigenvectors at the critical k-points.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        # Get Fourier coefficients of inv_eps via FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Plane wave indices: use hexagonal-like cutoff for speed
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            mi, ni = g_indices[i]
            for j in range(n_total):
                mj, nj = g_indices[j]
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # k-points along the irreducible BZ boundary
        k_points = []
        n_seg = n_kpts // 3
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0.0, n_seg + 1):
            k_points.append((t, t))
        
        k_unique = []
        for kp in k_points:
            if not any(abs(kp[0]-k2[0]) < 1e-10 and abs(kp[1]-k2[1]) < 1e-10 for k2 in k_unique):
                k_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        critical_k4 = None  # k-point where omega4 is maximized
        critical_k5 = None  # k-point where omega5 is minimized
        eigvec4 = None
        eigvec5 = None
        
        for (kx, ky) in k_unique:
            Gk = np.array([(m + kx, n + ky) for m, n in g_indices])
            Gk_sq = Gk[:, 0]**2 + Gk[:, 1]**2
            
            A = S * Gk_sq[None, :]
            B = S
            
            try:
                from scipy.linalg import eig as sp_eig
                vals, vecs = sp_eig(A, B)
                vals = np.real(vals)
                mask = vals > 1e-6
                vals = vals[mask]
                vecs = vecs[:, mask]
                vals = np.sort(vals)
                # Re-sort eigenvectors to match sorted eigenvalues
                order = np.argsort(vals)
                vals = vals[order]
                vecs = vecs[:, order]
            except:
                try:
                    B_inv_A = np.linalg.solve(B, A)
                    vals = np.linalg.eigvals(B_inv_A)
                    vals = np.real(vals)
                    mask = vals > 1e-6
                    vals = vals[mask]
                    vals = np.sort(vals)
                    vecs = None
                except:
                    continue
            
            if len(vals) < 5:
                continue
            
            omega4 = vals[3]
            omega5 = vals[4]
            
            if omega4 > max_omega4:
                max_omega4 = omega4
                critical_k4 = (kx, ky)
                if vecs is not None:
                    eigvec4 = vecs[:, 3]
            
            if omega5 < min_omega5:
                min_omega5 = omega5
                critical_k5 = (kx, ky)
                if vecs is not None:
                    eigvec5 = vecs[:, 4]
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            if return_eigvecs:
                return -1.0, None, None, None, None
            return -1.0
        
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        
        if return_eigvecs:
            return gap_ratio, critical_k4, critical_k5, eigvec4, eigvec5
        return gap_ratio
    
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """Create a cell with four rods centered in each quadrant."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = xx - cx
            dy = yy - cy
            if rod_radius > 0:
                val = np.abs(dx / rod_radius)**exponent + np.abs(dy / rod_radius)**exponent
            else:
                val = np.zeros_like(xx)
            cell[val <= 1.0 + 1e-9] = 1
        
        # Enforce D4 symmetry
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        cell = np.maximum(cell, cell[::-1, ::-1])
        
        return cell.tolist()
    
    def enforce_d4_symmetry(cell_arr):
        """Enforce full D4 symmetry on a 32x32 cell array."""
        cell_arr = np.array(cell_arr, dtype=int)
        # Reflect across horizontal midline
        cell_arr = np.maximum(cell_arr, cell_arr[::-1, :])
        # Reflect across vertical midline
        cell_arr = np.maximum(cell_arr, cell_arr[:, ::-1])
        # Transpose (diagonal reflection)
        cell_arr = np.maximum(cell_arr, cell_arr.T)
        # 180 degree rotation
        cell_arr = np.maximum(cell_arr, cell_arr[::-1, ::-1])
        return cell_arr
    
    # Generate and evaluate candidates
    best_score = -np.inf
    best_candidate = make_four_quadrant_cell(0.12, 2.0)
    
    # Coarse grid search over radius and exponent
    radii = [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]
    exponents = [2.0, 3.0, 4.0]
    
    candidates = []
    for r in radii:
        for e in exponents:
            candidates.append((r, e))
    
    candidates.sort(key=lambda x: abs(x[0] - 0.12) + abs(x[1] - 2.0))
    
    for (r, e) in candidates:
        if time.time() > deadline - 60:
            break
        cell = make_four_quadrant_cell(r, e)
        score = compute_tm_bands(cell, return_eigvecs=False)
        if isinstance(score, tuple):
            score = score[0]
        if score > best_score:
            best_score = score
            best_candidate = cell
    
    # Field-intensity-driven refinement
    if time.time() < deadline - 30 and best_score > 0:
        best_cell_arr = np.array(best_candidate, dtype=float)
        
        # Get eigenvectors at critical k-points
        result = compute_tm_bands(
            best_candidate, n_pw=13, n_kpts=15, return_eigvecs=True
        )
        if isinstance(result, tuple) and len(result) == 5:
            _, crit_k4, crit_k5, eigvec4, eigvec5 = result
        else:
            crit_k4 = crit_k5 = eigvec4 = eigvec5 = None
        
        if eigvec4 is not None and eigvec5 is not None:
            # Compute field intensity distributions for bands 4 and 5
            # |E|^2 in real space from plane-wave coefficients
            half = 13 // 2
            g_indices = []
            for m in range(-half, half + 1):
                for n in range(-half, half + 1):
                    if m * m + n * n <= half * half + half:
                        g_indices.append((m, n))
            n_total = len(g_indices)
            
            # Compute intensity on a grid
            coords = (np.arange(size) + 0.5) / size - 0.5
            xx, yy = np.meshgrid(coords, coords, indexing='ij')
            
            def compute_intensity(eigvec, kx, ky):
                """Compute |E(x,y)|^2 from plane-wave eigenvector."""
                intensity = np.zeros((size, size), dtype=float)
                for i, (m, n) in enumerate(g_indices):
                    gx = m + kx
                    gy = n + ky
                    phase = np.exp(2j * np.pi * (gx * xx + gy * yy))
                    intensity += np.abs(eigvec[i]) ** 2
                    # Cross terms
                    for j in range(i+1, n_total):
                        mj, nj = g_indices[j]
                        gxj = mj + kx
                        gyj = nj + ky
                        phase_j = np.exp(2j * np.pi * (gxj * xx + gyj * yy))
                        intensity += 2 * np.real(eigvec[i] * np.conj(eigvec[j]) * phase * np.conj(phase_j))
                return intensity
            
            # Intensity of band 4 at its critical k-point
            if crit_k4 is not None:
                int4 = compute_intensity(eigvec4, crit_k4[0], crit_k4[1])
            else:
                int4 = np.ones((size, size))
            
            # Intensity of band 5 at its critical k-point
            if crit_k5 is not None:
                int5 = compute_intensity(eigvec5, crit_k5[0], crit_k5[1])
            else:
                int5 = np.ones((size, size))
            
            # Normalize
            int4 = int4 / (np.max(int4) + 1e-10)
            int5 = int5 / (np.max(int5) + 1e-10)
            
            # Optimization: increase epsilon where band 5 has high intensity
            # and decrease epsilon where band 4 has high intensity
            # This pushes band 4 down and band 5 up
            num_iter = 0
            max_iter = 20
            learning_rate = 0.05
            
            current_cell = best_cell_arr.copy()
            current_score = best_score
            
            for iteration in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradient: d(score)/d(epsilon) ~ int5 - int4
                # We want to increase epsilon where int5 > int4
                gradient = int5 - int4
                gradient = gradient - np.mean(gradient)  # Zero mean to preserve average fill fraction
                
                # Update cell
                update = learning_rate * gradient
                new_cell = current_cell + update
                
                # Threshold to binary
                new_cell_bin = (new_cell > 0.5).astype(int)
                
                # Enforce D4 symmetry
                new_cell_sym = enforce_d4_symmetry(new_cell_bin)
                
                # Evaluate
                new_score = compute_tm_bands(new_cell_sym.tolist(), return_eigvecs=False)
                if isinstance(new_score, tuple):
                    new_score = new_score[0]
                
                if new_score > current_score:
                    current_cell = new_cell_sym.astype(float)
                    current_score = new_score
                    best_score = max(best_score, current_score)
                    best_candidate = new_cell_sym.tolist()
                    num_iter += 1
                else:
                    # Try smaller step
                    update = 0.5 * learning_rate * gradient
                    new_cell = current_cell + update
                    new_cell_bin = (new_cell > 0.5).astype(int)
                    new_cell_sym = enforce_d4_symmetry(new_cell_bin)
                    new_score = compute_tm_bands(new_cell_sym.tolist(), return_eigvecs=False)
                    if isinstance(new_score, tuple):
                        new_score = new_score[0]
                    
                    if new_score > current_score:
                        current_cell = new_cell_sym.astype(float)
                        current_score = new_score
                        best_score = max(best_score, current_score)
                        best_candidate = new_cell_sym.tolist()
                        num_iter += 1
                    else:
                        break  # No improvement, stop
    
    return best_candidate

# EVOLVE-BLOCK-END
