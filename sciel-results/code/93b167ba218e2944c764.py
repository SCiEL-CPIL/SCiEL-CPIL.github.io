import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 180  # safety margin before 200s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Work on the top-left 16x16 quadrant to enforce C4v symmetry
    half = size // 2
    
    # Fast band solver using a reduced set of plane waves for speed
    # The evaluator uses 289 (17x17), but we use 121 (11x11) for local eval
    N_pw = 11
    pw_list = []
    for nx in range(-N_pw//2, N_pw//2 + 1):
        for ny in range(-N_pw//2, N_pw//2 + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # K-points in the irreducible BZ (Gamma-M-X-Gamma path)
    # Gamma: (0,0), M: (0.5,0), X: (0.5,0.5)
    k_points = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.25)
    ]
    
    # Precompute grid coordinates for field reconstruction
    x_grid = (np.arange(size) + 0.5) / size
    y_grid = (np.arange(size) + 0.5) / size
    XX, YY = np.meshgrid(x_grid, y_grid)
    
    def expand_quadrant(q):
        """Expand 16x16 quadrant to 32x32 with C4v symmetry"""
        cell = np.zeros((size, size), dtype=int)
        cell[0:half, 0:half] = q
        # Mirror vertically
        cell[half:, 0:half] = q[::-1, :]
        # Mirror horizontally
        cell[0:half, half:] = q[:, ::-1]
        cell[half:, half:] = q[::-1, ::-1]
        return cell
    
    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon"""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps
    
    def solve_bands(cell, kx, ky, num_bands=5):
        """Solve for the lowest num_bands eigenvalues"""
        eps_inv_fft = compute_eps_inv_fourier(cell)
        k_vectors = pw_list + np.array([kx, ky])
        
        # Build the TM PWE matrix
        # H_ij = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
        H = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        
        for i in range(N_pw_actual):
            for j in range(N_pw_actual):
                qx = int(round(k_vectors[i][0] - k_vectors[j][0]))
                qy = int(round(k_vectors[i][1] - k_vectors[j][1]))
                
                idx_x = qx + N_pw//2
                idx_y = qy + N_pw//2
                
                if 0 <= idx_x < size and 0 <= idx_y < size:
                    coeff = eps_inv_fft[idx_y, idx_x]
                else:
                    coeff = 0.0
                
                k_diff_sq = (k_vectors[i][0] - k_vectors[j][0])**2 + \
                           (k_vectors[i][1] - k_vectors[j][1])**2
                H[i, j] = coeff * k_diff_sq
        
        # Solve eigenvalue problem
        vals, vecs = np.linalg.eigh(H)
        vals = vals[vals > 1e-6]
        
        if len(vals) < num_bands:
            return None, None
            
        return vals[:num_bands], vecs[:, :num_bands]
    
    def evaluate_gap(cell):
        """Evaluate the gap-midgap ratio between bands 4 and 5"""
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
    
    def field_from_vec(vec, kx, ky):
        """Reconstruct real-space field from plane-wave coefficients"""
        field = np.zeros((size, size), dtype=complex)
        for j in range(N_pw_actual):
            nx, ny = pw_list[j]
            kx_j = nx + kx
            ky_j = ny + ky
            phase = 2 * math.pi * (kx_j * XX + ky_j * YY)
            field += vec[j] * np.exp(1j * phase)
        return field
    
    # Initialize with a simple symmetric pattern
    q = np.zeros((half, half), dtype=int)
    # Try a central rod
    cy, cx = half/2.0, half/2.0
    y, x = np.mgrid[0:half, 0:half]
    y = y + 0.5
    x = x + 0.5
    dist_sq = (x - cx)**2 + (y - cy)**2
    q[dist_sq < 4.0**2] = 1
    
    cell = expand_quadrant(q)
    current_score = evaluate_gap(cell)
    best_cell = cell.copy()
    best_score = current_score
    
    # Optimization loop using field-based gradient heuristic
    for step in range(100):
        if time.time() > deadline - 10:
            break
            
        # Find critical k-points
        max_4_val = 0
        max_4_k = None
        max_4_vec = None
        
        min_5_val = float('inf')
        min_5_k = None
        min_5_vec = None
        
        for kx, ky in k_points:
            vals, vecs = solve_bands(cell, kx, ky, num_bands=5)
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
            
        # Get field intensities
        field_4 = field_from_vec(max_4_vec, max_4_k[0], max_4_k[1])
        field_5 = field_from_vec(min_5_vec, min_5_k[0], min_5_k[1])
        
        int_4 = np.abs(field_4)**2
        int_5 = np.abs(field_5)**2
        
        if np.max(int_4) > 0:
            int_4 /= np.max(int_4)
        if np.max(int_5) > 0:
            int_5 /= np.max(int_5)
            
        # Force map: add dielectric where band 4 is strong, remove where band 5 is strong
        force = int_4 - int_5
        
        # Update the quadrant
        q_float = q.astype(float) + 0.5 * force[0:half, 0:half]
        q_new = (q_float > 0.5).astype(int)
        
        # Evaluate new cell
        cell_new = expand_quadrant(q_new)
        new_score = evaluate_gap(cell_new)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = cell_new.copy()
            cell = cell_new.copy()
            q = q_new.copy()
        else:
            # Try a smaller step if we didn't improve
            q_float = q.astype(float) + 0.25 * force[0:half, 0:half]
            q_new = (q_float > 0.5).astype(int)
            cell_new = expand_quadrant(q_new)
            new_score = evaluate_gap(cell_new)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = cell_new.copy()
                cell = cell_new.copy()
                q = q_new.copy()
    
    return best_cell.tolist()
