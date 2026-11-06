"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _enforce_symmetry(cell):
    """
    Enforce the full D4 symmetry group of the square lattice about the center.
    The cell must be invariant under:
    1. Reflection across vertical axis (i, j) -> (i, N-1-j)
    2. Reflection across horizontal axis (i, j) -> (N-1-i, j)
    3. Transpose (i, j) -> (j, i)
    This implies 90-degree rotational symmetry as well.
    """
    N = cell.shape[0]
    cell = np.array(cell, dtype=int)
    # We can enforce this by averaging the symmetric equivalents and thresholding,
    # or simply by constructing the cell from the fundamental domain.
    # For a 32x32 grid, the fundamental domain is the top-left 8x8 quadrant 
    # (indices 0..15 for i, 0..15 for j, but actually due to center symmetry,
    # the unique pixels are in the top-left 16x16 quadrant, mirrored to others).
    # Wait, the symmetry is about the center of the cell.
    # Pixel (i, j) is equivalent to:
    # (i, N-1-j)
    # (N-1-i, j)
    # (N-1-i, N-1-j)
    # (j, i)
    # (j, N-1-i)
    # (N-1-j, i)
    # (N-1-j, N-1-i)
    
    # To enforce strictly:
    # Take the minimum of all symmetric counterparts to ensure if any is 0, it's 0?
    # Or just define the cell from the unique 1/8th sector?
    # The prompt says: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    # This is D4 symmetry.
    
    # Simplest robust way:
    # 1. Flip vertically
    cell = np.minimum(cell, cell[::-1, :].copy())
    # 2. Flip horizontally
    cell = np.minimum(cell, cell[:, ::-1].copy())
    # 3. Transpose
    cell = np.minimum(cell, cell.T.copy())
    # This might be too restrictive if the initial cell wasn't symmetric.
    # It's better to generate symmetric cells directly.
    
    return cell

def _make_superellipse_cell(size, n_exp, a_size, offset):
    """
    Generate a cell with 4 superellipses centered at (+/-offset, +/-offset).
    Equation: |dx/a|^n + |dy/a|^n <= 1
    """
    cell = np.zeros((size, size), dtype=int)
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='xy')
    
    centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
    
    for cx, cy in centers:
        dx = X - cx
        dy = Y - cy
        # Handle zero division or negative bases for non-integer n
        # abs handles the sign.
        val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
        cell[val <= 1.0] = 1
        
    return _enforce_symmetry(cell)

def _compute_eps_fft(cell, size):
    """Compute the 2D FFT of the epsilon field, normalized."""
    eps = np.array(cell, dtype=float)
    eps = 1.0 + (12.0 - 1.0) * eps
    # np.fft.fftn computes sum_{x,y} eps(x,y) exp(-i 2pi (m x + n y)/N)
    # The coefficient c_G = 1/N^2 sum ...
    return np.fft.fftn(eps) / (size * size)

def _tm_eigvals(kx, ky, full_eps_fft, size, pw_idx):
    """
    Solve TM eigenvalue problem at a single k-point.
    H_{ij} = (k_i . k_j) * eps_{G_i - G_j}
    """
    n_pw = len(pw_idx)
    N = n_pw * n_pw
    
    Gx = np.repeat(pw_idx, n_pw)
    Gy = np.tile(pw_idx, n_pw)
    
    kx_g = kx + Gx
    ky_g = ky + Gy
    
    K = np.column_stack((kx_g, ky_g))
    K_dot_K = K @ K.T
    
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    
    # Map differences to FFT indices
    # FFT index for frequency f is f % size
    idx_m = Gx_diff % size
    idx_n = Gy_diff % size
    
    eps_coeffs = full_eps_fft[idx_m, idx_n]
    
    H = K_dot_K * eps_coeffs
    
    # H is Hermitian
    H = (H + H.conj().T) / 2.0
    
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)

def _evaluate_gap(cell, size, n_pw, k_points):
    """
    Evaluate the gap-midgap ratio between band 4 and 5.
    """
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
        
    full_eps_fft = _compute_eps_fft(cell, size)
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        eigvals = _tm_eigvals(kx, ky, full_eps_fft, size, pw_idx)
        if len(eigvals) >= 5:
            # Sort is done by eigvalsh
            band4_max = max(band4_max, eigvals[3])
            band5_min = min(band5_min, eigvals[4])
            
    if band5_min <= band4_max:
        return -1.0
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    return 2.0 * gap / midgap

def _mutate_cell(cell, n_flips):
    """Randomly flip n pixels while maintaining D4 symmetry."""
    cell = cell.copy()
    size = cell.shape[0]
    for _ in range(n_flips):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        new_val = 1 - cell[i, j]
        # The 8 symmetric positions
        positions = [
            (i, j), (j, i),
            (size-1-i, size-1-j), (size-1-j, size-1-i),
            (i, size-1-j), (size-1-j, i),
            (size-1-i, j), (j, size-1-i),
        ]
        for pi, pj in positions:
            cell[pi, pj] = new_val
    return cell

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    # 120s limit, leave 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Define k-points for the evaluator (30 points in irreducible BZ)
    # The irreducible BZ of a square lattice is the triangle (0,0)-(pi,0)-(pi,pi).
    # However, the prompt says "30 wave vectors covering the boundary and interior".
    # Standard set:
    # Gamma (0,0) to M (pi/2, pi/2) is not the boundary of the full BZ, 
    # but the irreducible BZ is 1/8th of the square BZ.
    # Square BZ: [-pi, pi] x [-pi, pi]. Irreducible: 0 <= kx <= ky <= pi.
    # Boundary: (0,0)->(pi,0) [X], (pi,0)->(pi,pi) [M], (pi,pi)->(0,0) [Gamma-M].
    # Let's generate points on these paths.
    
    k_points = []
    # Path 1: Gamma (0,0) to X (pi, 0)
    for i in range(10):
        t = i / 9.0
        k_points.append((t * np.pi, 0.0))
    # Path 2: X (pi, 0) to M (pi, pi)
    for i in range(1, 10): # Skip start to avoid duplicate
        t = i / 9.0
        k_points.append((np.pi, t * np.pi))
    # Path 3: M (pi, pi) to Gamma (0,0)
    for i in range(1, 10): # Skip start to avoid duplicate
        t = i / 9.0
        k_points.append((np.pi * (1 - t), np.pi * (1 - t)))
        
    k_points = np.array(k_points)
    
    # Phase 1: Superellipse Parameter Search
    # Using fast 7x7 plane waves for speed
    n_pw_fast = 7
    
    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Search space
    n_exp_vals = np.linspace(1.5, 8.0, 15)
    a_size_vals = np.linspace(0.08, 0.22, 15)
    offset_vals = np.linspace(0.15, 0.35, 15)
    
    # Coarse Grid Search
    for n_exp in n_exp_vals:
        for a_size in a_size_vals:
            for offset in offset_vals:
                if time.time() > deadline - 60.0:
                    break
                cell = _make_superellipse_cell(size, n_exp, a_size, offset)
                score = _evaluate_gap(cell, size, n_pw_fast, k_points)
                if score > best_score:
                    best_score = score
                    best_cell = cell
                    best_params = (n_exp, a_size, offset)
            if time.time() > deadline - 60.0:
                break
        if time.time() > deadline - 60.0:
            break
            
    if best_cell is None:
        # Fallback
        best_cell = _make_superellipse_cell(size, 2.0, 0.12, 0.25)
        best_score = _evaluate_gap(best_cell, size, n_pw_fast, k_points)
        best_params = (2.0, 0.12, 0.25)

    # Phase 2: Local Optimization of Superellipse Parameters
    current_n, current_a, current_off = best_params
    current_cell = best_cell
    current_score = best_score
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 30.0 and no_improve < max_no_improve:
        # Perturb parameters
        new_n = current_n + np.random.normal(0, 0.5)
        new_a = current_a + np.random.normal(0, 0.02)
        new_off = current_off + np.random.normal(0, 0.02)
        
        new_n = np.clip(new_n, 1.0, 20.0)
        new_a = np.clip(new_a, 0.05, 0.30)
        new_off = np.clip(new_off, 0.10, 0.40)
        
        new_cell = _make_superellipse_cell(size, new_n, new_a, new_off)
        new_score = _evaluate_gap(new_cell, size, n_pw_fast, k_points)
        
        if new_score > current_score:
            current_n, current_a, current_off = new_n, new_a, new_off
            current_cell = new_cell
            current_score = new_score
            no_improve = 0
        else:
            no_improve += 1
            
    # Phase 3: Pixel Refinement
    # Take the best cell and refine pixels
    current_cell = current_cell.copy()
    current_score = _evaluate_gap(current_cell, size, n_pw_fast, k_points)
    
    # Use a slightly larger basis for refinement if time permits?
    # 7x7 is very fast. Let's stick to 7x7 for the loop, but do a final check with 17x17.
    
    n_iter = 0
    while time.time() < deadline - 20.0:
        n_iter += 1
        # Adaptive mutation count
        if n_iter % 10 == 0:
            n_flips = 2
        else:
            n_flips = 1
            
        trial_cell = _mutate_cell(current_cell, n_flips)
        trial_score = _evaluate_gap(trial_cell, size, n_pw_fast, k_points)
        
        if trial_score > current_score:
            current_cell = trial_cell
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            # Occasional random jump to escape local minima
            if n_iter % 50 == 0:
                # Revert to superellipse shape and perturb slightly
                n_exp, a_size, offset = best_params
                new_cell = _make_superellipse_cell(size, 
                                                   n_exp + np.random.normal(0, 0.2), 
                                                   a_size + np.random.normal(0, 0.01), 
                                                   offset + np.random.normal(0, 0.01))
                new_score = _evaluate_gap(new_cell, size, n_pw_fast, k_points)
                if new_score > current_score:
                    current_cell = new_cell
                    current_score = new_score

    # Final Verification with Full 17x17 Basis
    # This is slow, so we only do it for the best candidate
    if time.time() < deadline - 5.0:
        n_pw_full = 17
        final_score = _evaluate_gap(best_cell, size, n_pw_full, k_points)
        # We return the cell, the score is used by the evaluator.
        # But we can check if the fast surrogate was misleading.
        # If the full score is significantly worse, we might want to try the 2nd best?
        # For now, we trust the 7x7 ranking is decent.
        
    return best_cell.tolist()
# EVOLVE-BLOCK-END
