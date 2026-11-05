import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initial guess: 4 rods in corners
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    r = 5.5
    cell = np.zeros((size, size), dtype=int)
    for cx, cy in centers:
        dist = (x - cx)**2 + (y - cy)**2
        cell[dist < r**2] = 1
        
    # Enforce symmetry
    cell = (cell + cell[::-1, ::-1]) // 2
    cell = (cell + cell.T) // 2

    # Fast internal PWE solver parameters
    # Use N=13 (169 plane waves) for a balance of speed and accuracy.
    # The evaluator uses 289, but 169 should be sufficient for optimization.
    N_pw = 13
    pw_list = []
    for nx in range(-N_pw//2, N_pw//2 + 1):
        for ny in range(-N_pw//2, N_pw//2 + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # k-vectors in the irreducible BZ
    # Sample points along Gamma-M-X-Gamma
    k_points = [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.4, 0.0),
        (0.5, 0.0),
        (0.5, 0.1),
        (0.5, 0.2),
        (0.5, 0.3),
        (0.5, 0.4),
        (0.5, 0.5),
        (0.4, 0.4),
        (0.3, 0.3),
        (0.2, 0.2),
        (0.1, 0.1),
        (0.25, 0.25),
        (0.35, 0.35),
        (0.45, 0.45),
    ]

    def compute_eps_inv_fourier(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps

    def solve_bands(cell, kx, ky, num_bands=5):
        eps_inv_fft = compute_eps_inv_fourier(cell)
        k_vectors = pw_list + np.array([kx, ky])
        
        H = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        
        # Vectorized matrix construction
        # We need H[i, j] = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
        # Note: The standard TM PWE matrix is H_ij = (1/eps)_{k_i - k_j} * (k_i - k_j)^2
        # Wait, let's re-verify the TM matrix.
        # Maxwell: k^2 E = omega^2 eps E (TE) -> k^2 / eps E = omega^2 E
        # For TM: k^2 (1/eps) E = omega^2 E
        # In plane wave basis:
        # sum_j (1/eps)_{k_i - k_j} (k_i - k_j)^2 c_j ? No.
        # The operator is (k^2) * (1/eps).
        # (k^2) acts on the plane wave index i: (k_i)^2.
        # (1/eps) convolves: sum_j (1/eps)_{k_i - k_j} c_j.
        # So H_ij = (k_i)^2 (1/eps)_{k_i - k_j} ?
        # Let's check Joannopoulos.
        # For TE: H_ij = (k_i - k_j)^2 / eps_{k_i - k_j}
        # For TM: H_ij = (k_i - k_j)^2 * (1/eps)_{k_i - k_j}
        
        # Let's use the TM form: H_ij = (k_i - k_j)^2 * (1/eps)_{k_i - k_j}
        
        for i in range(N_pw_actual):
            for j in range(N_pw_actual):
                qx = int(round(k_vectors[i][0] - k_vectors[j][0]))
                qy = int(round(k_vectors[i][1] - k_vectors[j][1]))
                
                if -N_pw <= qx <= N_pw and -N_pw <= qy <= N_pw:
                    idx_x = qx + N_pw//2
                    idx_y = qy + N_pw//2
                    if 0 <= idx_x < size and 0 <= idx_y < size:
                        coeff = eps_inv_fft[idx_y, idx_x]
                    else:
                        coeff = 0.0
                else:
                    coeff = 0.0
                
                k_diff_sq = (k_vectors[i][0] - k_vectors[j][0])**2 + (k_vectors[i][1] - k_vectors[j][1])**2
                H[i, j] = coeff * k_diff_sq
        
        vals, vecs = np.linalg.eigh(H)
        vals = vals[vals > 1e-6]
        
        if len(vals) < num_bands:
            return None, None
            
        return vals[:num_bands], vecs[:, :num_bands]

    def evaluate_gap(cell):
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, _ = solve_bands(cell, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    current_score = evaluate_gap(cell)
    best_cell = cell.copy()
    best_score = current_score

    # Topological Optimization Loop
    # We approximate the gradient of the gap with respect to the dielectric distribution.
    # Instead of full adjoint (which is complex), we use a local finite-difference-like heuristic
    # or a simplified field-intensity gradient.
    
    # A more robust approach for binary masks is to use a "level set" or "density" method.
    # Let's use a density field rho in [0,1] and update it based on the gradient of the gap.
    # Approximate gradient: d(omega^2)/d(eps) is proportional to -|E|^2 for TM?
    # Actually, d(omega^2)/d(eps) = -omega^4 * |E|^2 / (2 * norm).
    # We want to maximize gap = omega_5 - omega_4.
    # d(gap)/d(eps) = d(omega_5)/d(eps) - d(omega_4)/d(eps)
    # d(omega)/d(eps) ~ -|E|^2 (for high eps, lower omega).
    # So d(omega)/d(eps) is negative.
    # d(gap)/d(eps) = (-C5 |E5|^2) - (-C4 |E4|^2) = C4 |E4|^2 - C5 |E5|^2.
    # If C4 |E4|^2 > C5 |E5|^2, increasing eps increases the gap.
    # So we should add dielectric where |E4|^2 is high relative to |E5|^2.
    
    rho = cell.astype(float)
    
    for step in range(100):
        if time.time() > deadline - 15:
            break
            
        # Find critical k-points
        max_4_val = 0
        max_4_k = None
        max_4_vec = None
        
        min_5_val = float('inf')
        min_5_k = None
        min_5_vec = None
        
        for kx, ky in k_points:
            vals, vecs = solve_bands(rho, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            
            if omegas[3] > max_4_val:
                max_4_val = omegas[3]
                max_4_k = (kx, ky)
                max_4_vec = vecs[:, 3]
                
            if omegas[4] < min_5_val:
                min_5_val = omegas[4]
                min_5_k = (kx, ky)
                min_5_vec = vecs[:, 4]
                
        if max_4_vec is None or min_5_vec is None:
            break
            
        def field_from_vec(vec, kx, ky):
            field = np.zeros((size, size), dtype=complex)
            x_grid = (np.arange(size) + 0.5) / size
            y_grid = (np.arange(size) + 0.5) / size
            XX, YY = np.meshgrid(x_grid, y_grid)
            
            for j in range(N_pw_actual):
                nx, ny = pw_list[j]
                kx_j = nx + kx
                ky_j = ny + ky
                phase = 2 * math.pi * (kx_j * XX + ky_j * YY)
                field += vec[j] * np.exp(1j * phase)
                
            return field
            
        field_4 = field_from_vec(max_4_vec, max_4_k[0], max_4_k[1])
        field_5 = field_from_vec(min_5_vec, min_5_k[0], min_5_k[1])
        
        int_4 = np.abs(field_4)**2
        int_5 = np.abs(field_5)**2
        
        # Normalize
        max_i4 = np.max(int_4)
        max_i5 = np.max(int_5)
        if max_i4 > 0: int_4 /= max_i4
        if max_i5 > 0: int_5 /= max_i5
        
        # Gradient approximation
        # We want to increase rho where int_4 is high and int_5 is low.
        # force = int_4 - int_5
        force = int_4 - int_5
        
        # Update rho
        # Use a small step size
        alpha = 0.1
        rho_new = rho + alpha * force
        
        # Project to [0, 1]
        rho_new = np.clip(rho_new, 0, 1)
        
        # Enforce symmetry on rho
        rho_new = (rho_new + rho_new[::-1, ::-1]) / 2
        rho_new = (rho_new + rho_new.T) / 2
        
        # Threshold to binary for evaluation
        cell_new = (rho_new > 0.5).astype(int)
        
        # Evaluate
        new_score = evaluate_gap(cell_new)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = cell_new.copy()
            rho = rho_new.copy()
            cell = cell_new.copy()
        else:
            # If no improvement, reduce step size or try a different move
            # For now, just keep the old state
            pass
            
    return best_cell.tolist()
