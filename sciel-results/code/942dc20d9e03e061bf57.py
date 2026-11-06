"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _enforce_d4(cell):
    """Enforce D4 symmetry: row-flip, col-flip, transpose."""
    N = cell.shape[0]
    cell = np.array(cell, dtype=float)
    # Average over all 8 symmetric positions
    sym_positions = lambda i, j: [
        (i, j), (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j),
        (j, i), (N-1-j, N-1-i), (j, N-1-i), (N-1-j, i)
    ]
    # For efficiency, just do the three operations
    cell = (cell + cell[::-1, :].copy()) / 2.0
    cell = (cell + cell[:, ::-1].copy()) / 2.0
    cell = (cell + cell.T.copy()) / 2.0
    # Threshold
    return (cell > 0.5).astype(int)

def _compute_eps_fourier(cell, size):
    """Compute Fourier coefficients of epsilon field."""
    eps = 1.0 + 11.0 * cell.astype(float)
    # DFT: c_G = 1/N^2 sum_{x,y} eps(x,y) exp(-i 2pi (Gx x + Gy y)/N)
    # Using FFT: np.fft.fftn gives sum, so divide by N^2
    return np.fft.fftn(eps) / (size * size)

def _tm_eigenvalues(kx, ky, eps_ft, size, n_pw):
    """
    Solve TM eigenvalue problem at (kx, ky) using n_pw x n_pw plane waves.
    For TM: H_{mn} = (k+G_m)·(k+G_n) * eps_{G_m - G_n}
    """
    half = n_pw // 2
    pw = np.arange(-half, half + 1)
    
    Gx = np.repeat(pw, n_pw)
    Gy = np.tile(pw, n_pw)
    
    kx_g = kx + Gx
    ky_g = ky + Gy
    
    # K dot K matrix
    Kx2 = kx_g ** 2
    Ky2 = ky_g ** 2
    KdotK = Kx2[:, None] + Ky2[None, :]
    
    # Fourier coefficients of epsilon at differences
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    
    # Map to FFT indices (mod size)
    idx_m = Gx_diff % size
    idx_n = Gy_diff % size
    
    eps_coeffs = eps_ft[idx_m, idx_n]
    
    H = KdotK * eps_coeffs
    H = (H + H.conj().T) / 2.0
    
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)

def _evaluate_gap(cell, size, n_pw, k_points):
    """Evaluate gap-midgap ratio between band 4 and 5."""
    eps_ft = _compute_eps_fourier(cell, size)
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        eigvals = _tm_eigenvalues(kx, ky, eps_ft, size, n_pw)
        if len(eigvals) >= 5:
            band4_max = max(band4_max, eigvals[3])
            band5_min = min(band5_min, eigvals[4])
    
    if band5_min <= band4_max:
        return -1.0
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _make_k_points():
    """Generate 30 k-points in the irreducible BZ."""
    k_points = []
    # Gamma (0,0) to X (pi, 0)
    for i in range(10):
        t = i / 9.0
        k_points.append((t * np.pi, 0.0))
    # X (pi, 0) to M (pi, pi)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((np.pi, t * np.pi))
    # M (pi, pi) to Gamma (0,0)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((np.pi * (1 - t), np.pi * (1 - t)))
    return k_points

def _mutate_cell(cell, n_flips, size):
    """Randomly flip n pixels while maintaining D4 symmetry."""
    cell = cell.copy()
    for _ in range(n_flips):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        new_val = 1 - cell[i, j]
        positions = [
            (i, j), (j, i),
            (size-1-i, size-1-j), (size-1-j, size-1-i),
            (i, size-1-j), (size-1-j, i),
            (size-1-i, j), (j, size-1-i),
        ]
        # Remove duplicates
        unique_pos = list(set(positions))
        for pi, pj in unique_pos:
            cell[pi, pj] = new_val
    return cell

def _hill_climb(cell, size, k_points, n_pw, deadline, max_iter=5000):
    """Hill climbing with adaptive step size."""
    current_cell = cell.copy()
    current_score = _evaluate_gap(current_cell, size, n_pw, k_points)
    best_cell = current_cell.copy()
    best_score = current_score
    
    no_improve = 0
    step_size = 1
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive step: increase when stuck, decrease when improving
        if no_improve > 20:
            step_size = min(5, step_size + 1)
        elif no_improve == 0:
            step_size = max(1, step_size - 1)
        
        trial_cell = _mutate_cell(current_cell, step_size, size)
        trial_score = _evaluate_gap(trial_cell, size, n_pw, k_points)
        
        if trial_score > current_score:
            current_cell = trial_cell
            current_score = trial_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            no_improve += 1
        
        # Occasionally reset to best to avoid drifting
        if no_improve > 100:
            current_cell = best_cell.copy()
            current_score = best_score
            no_improve = 0
    
    return best_cell, best_score

def _multi_start_optimize(size, k_points, n_pw, deadline, n_starts=5):
    """Try multiple random starts and hill climb from each."""
    best_overall = None
    best_overall_score = -np.inf
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate a random symmetric starting cell
        cell = np.zeros((size, size), dtype=int)
        # Random density around 0.4-0.6
        density = random.uniform(0.3, 0.6)
        for i in range(size):
            for j in range(size):
                if random.random() < density:
                    cell[i, j] = 1
        cell = _enforce_d4(cell)
        
        # Hill climb
        result_cell, result_score = _hill_climb(
            cell, size, k_points, n_pw, deadline - 5, max_iter=2000
        )
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_cell
    
    return best_overall, best_overall_score

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    random.seed(42)
    np.random.seed(42)
    
    k_points = _make_k_points()
    
    # Phase 1: Fast screening with small basis (7x7)
    n_pw_fast = 7
    best_cell, best_score = _multi_start_optimize(
        size, k_points, n_pw_fast, deadline - 60, n_starts=8
    )
    
    # Phase 2: Refine with medium basis (11x11)
    if time.time() < deadline - 30:
        n_pw_med = 11
        refined_cell, refined_score = _hill_climb(
            best_cell, size, k_points, n_pw_med, deadline - 15, max_iter=3000
        )
        if refined_score > best_score:
            best_cell = refined_cell
            best_score = refined_score
    
    # Phase 3: Final refinement with larger basis (13x13) if time permits
    if time.time() < deadline - 10:
        n_pw_final = 13
        final_cell, final_score = _hill_climb(
            best_cell, size, k_points, n_pw_final, deadline - 2, max_iter=1500
        )
        if final_score > best_score:
            best_cell = final_cell
            best_score = final_score
    
    # Ensure valid output
    best_cell = _enforce_d4(best_cell)
    return best_cell.tolist()
# EVOLVE-BLOCK-END
