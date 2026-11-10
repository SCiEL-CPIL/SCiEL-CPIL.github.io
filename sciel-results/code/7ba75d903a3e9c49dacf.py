import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use multiple deterministic starting configurations (different disk arrangements).
    2. Apply greedy local search (hill climbing) from each starting point.
    3. Enforce C4v symmetry throughout.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 185.0  # Safety margin for 200s limit

    # --- 1. Fast Evaluator ---
    # Use 7x7 plane waves (49 modes) for speed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    # FFT index lookup
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    # Along M-X: (t, 0.5-t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build epsilon grid
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues (we need indices 3 and 4 for bands 4 and 5)
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 2. Symmetry enforcement ---
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                # All 8 symmetric points
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    def get_orbit(i, j):
        """Get all symmetric pixels for position (i,j)"""
        orbit = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, j),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i)
        ]
        return list(set(orbit))

    # --- 3. Multiple starting configurations ---
    def make_disk_cell(centers, radius):
        """Create a cell with disks at given centers"""
        cell = np.zeros((size, size), dtype=int)
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        return cell

    # Different starting configurations to explore
    starting_configs = [
        # Config 1: 4 disks at quarter-points, radius 4
        ([(8, 8), (8, 24), (24, 8), (24, 24)], 4.0),
        # Config 2: 4 disks at quarter-points, radius 3
        ([(8, 8), (8, 24), (24, 8), (24, 24)], 3.0),
        # Config 3: 4 disks at quarter-points, radius 5
        ([(8, 8), (8, 24), (24, 8), (24, 24)], 5.0),
        # Config 4: 1 disk at center, radius 6
        ([(16, 16)], 6.0),
        # Config 5: 1 disk at center, radius 5
        ([(16, 16)], 5.0),
        # Config 6: 4 disks at edge midpoints, radius 3
        ([(0, 16), (31, 16), (16, 0), (16, 31)], 3.0),
        # Config 7: 4 disks at edge midpoints, radius 4
        ([(0, 16), (31, 16), (16, 0), (16, 31)], 4.0),
    ]

    best_cell = None
    best_score = -float('inf')

    # --- 4. Greedy local search from each starting config ---
    for config_idx, (centers, radius) in enumerate(starting_configs):
        if time.time() > deadline:
            break
            
        # Create and symmetrize starting cell
        cell = make_disk_cell(centers, radius)
        cell = enforce_symmetry(cell)
        
        current_score = evaluate_gap(cell)
        
        if best_cell is None or current_score > best_score:
            best_score = current_score
            best_cell = cell.copy()
        
        # Greedy local search: try flipping each orbit, keep if it improves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all possible orbit flips
            best_flip = None
            best_flip_score = current_score
            
            for i in range(size):
                for j in range(size):
                    if time.time() > deadline:
                        break
                    
                    orbit = get_orbit(i, j)
                    new_cell = cell.copy()
                    for pi, pj in orbit:
                        new_cell[pi, pj] = 1 - new_cell[pi, pj]
                    
                    new_score = evaluate_gap(new_cell)
                    
                    if new_score > best_flip_score + 1e-10:
                        best_flip_score = new_score
                        best_flip = new_cell.copy()
                
                if time.time() > deadline:
                    break
            
            if best_flip is not None:
                cell = best_flip
                current_score = best_flip_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_cell = cell.copy()
        
        if time.time() > deadline:
            break

    # --- 5. Final refinement: try single-pixel flips on best cell ---
    if best_cell is not None and time.time() < deadline - 10:
        refined = best_cell.copy()
        refined_score = best_score
        
        # Try flipping each orbit once more
        for i in range(size):
            for j in range(size):
                if time.time() > deadline - 5:
                    break
                
                orbit = get_orbit(i, j)
                new_cell = refined.copy()
                for pi, pj in orbit:
                    new_cell[pi, pj] = 1 - new_cell[pi, pj]
                
                new_score = evaluate_gap(new_cell)
                
                if new_score > refined_score + 1e-10:
                    refined = new_cell
                    refined_score = new_score
            
            if time.time() > deadline - 5:
                break
        
        if refined_score > best_score:
            best_score = refined_score
            best_cell = refined.copy()

    # --- 6. Finalize ---
    if best_cell is None:
        # Fallback: return a simple symmetric pattern
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    best_cell[i, j] = 1
        best_cell = enforce_symmetry(best_cell)
    
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()
