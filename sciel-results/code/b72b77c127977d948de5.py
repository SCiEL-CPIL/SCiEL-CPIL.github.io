import numpy as np
import time
import math

_DEADLINE = 225.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# ---- fast plane-wave TM surrogate (vectorized) ------------------------------
def _build_pw_evaluator(size, eps_low, eps_high, pw_half=3, n_k=12):
    """
    Precompute everything needed for fast TM band-structure evaluation.
    Uses (2*pw_half+1)^2 plane waves and n_k points in the irreducible BZ.
    """
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1)
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=np.float64)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=np.float64)
    
    # G difference indices for FFT lookup
    G_diff_idx_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_idx_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_idx_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_idx_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size
    
    # K-points in irreducible BZ (triangle: 0 <= ky <= kx, kx+ky <= 0.5)
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t) for t in [0, 0.5]
    for t in np.linspace(0.1, 0.5, 5):
        k_points.append((0.0, float(t)))
    # Along M-X: (0.5-t, t) for t in [0, 0.5] -- this is the M-X edge
    # Actually irreducible BZ: kx >= ky >= 0, kx + ky <= 0.5
    # M = (0.5, 0), X = (0.5, 0.5), Gamma = (0,0)
    # Edges: Gamma-M: (t, 0), M-X: (0.5, t), Gamma-X: (t, t)
    for t in np.linspace(0.1, 0.5, 5):
        k_points.append((0.5, float(t)))
    # Interior points
    k_points.append((0.3, 0.1))
    k_points.append((0.4, 0.2))
    
    k_points = np.array(k_points, dtype=np.float64)
    
    return Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw

def _evaluate_gap_fast(cell_grid, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low=1.0, eps_high=12.0):
    """
    Fast evaluation of the gap-midgap ratio between bands 4 and 5.
    cell_grid: 32x32 array of 0/1
    """
    size = cell_grid.shape[0]
    eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
    inv_eps = 1.0 / eps_grid
    
    # FFT of 1/epsilon
    fft_1eps = np.fft.fft2(inv_eps)
    # coeffs[i,j] = (1/eps)_{G_i - G_j}
    coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        Kx = Gx_arr + kx
        Ky = Gy_arr + ky
        
        # H[i,j] = (K_i · K_j) * coeffs_{G_i - G_j}
        dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
        H = dots * coeffs
        H_real = H.real
        # Symmetrize for numerical stability
        H_real = (H_real + H_real.T) * 0.5
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(H_real)
        
        if len(eigenvalues) >= 5:
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            
            if w4_sq > 0:
                w4 = math.sqrt(w4_sq)
                if w4 > max_omega_4:
                    max_omega_4 = w4
            if w5_sq > 0:
                w5 = math.sqrt(w5_sq)
                if w5 < min_omega_5:
                    min_omega_5 = w5
    
    if min_omega_5 == float('inf') or max_omega_4 <= 0:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) * 0.5
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def _enforce_c4v(cell, size):
    """Enforce C4v symmetry by averaging over all 8 symmetry operations."""
    sym = np.zeros_like(cell, dtype=float)
    for i in range(size):
        for j in range(size):
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
            s = sum(cell[pi, pj] for pi, pj in points)
            sym[i, j] = 1.0 if s >= 4 else 0.0
    return sym.astype(int)

def _make_cell_fourier(coeffs, size=32, n_modes=6):
    """
    Generate a C4v-symmetric binary mask from Fourier coefficients.
    coeffs: array of length n_modes with real/imag pairs for each mode.
    Uses modes that respect C4v symmetry.
    """
    # C4v-compatible modes: (l, m) where the pattern is invariant under 90-degree rotation
    # For a 32x32 grid, we use the first few non-trivial modes
    # Mode (0,0) is DC, skip it
    # Modes like (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2), etc.
    # For C4v, we need modes that transform trivially or in allowed representations
    
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size
    y = (yy + 0.5) / size
    
    field = np.zeros((size, size), dtype=np.float64)
    
    # Define C4v-symmetric basis functions using trigonometric functions
    # For C4v, we can use: cos(2pi*l*x)*cos(2pi*m*y) + cos(2pi*m*x)*cos(2pi*l*y) for l != m
    # And cos(2pi*l*x)*cos(2pi*l*y) for l == m (diagonal modes)
    
    # Use a set of modes that are C4v symmetric
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2), (2, 1), (1, 2)]
    
    for idx, (l, m) in enumerate(modes):
        if idx >= len(coeffs) // 2:
            break
        amp = coeffs[idx]
        # For C4v, use the symmetric combination
        field += amp * (np.cos(2 * np.pi * l * x) * np.cos(2 * np.pi * m * y) +
                        np.cos(2 * np.pi * m * x) * np.cos(2 * np.pi * l * y))
    
    # Add a radial component for the "hole" or "rod" structure
    if len(coeffs) > len(modes):
        r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        field += coeffs[len(modes)] * (1.0 - 2.0 * r)
    
    # Threshold to binary
    threshold = 0.0
    cell = (field > threshold).astype(int)
    
    # Enforce C4v symmetry
    cell = _enforce_c4v(cell, size)
    
    return cell

def _search_fourier(size=32, eps_low=1.0, eps_high=12.0):
    """
    Search over Fourier coefficients to find a good design.
    Uses simulated annealing on the coefficient space.
    """
    Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw = _build_pw_evaluator(
        size, eps_low, eps_high, pw_half=3, n_k=12)
    
    # Number of Fourier coefficients to optimize
    n_coeffs = 9  # 8 mode amplitudes + 1 radial
    
    # Initial guess: start with a simple rod-like structure
    np.random.seed(42)
    best_coeffs = np.zeros(n_coeffs)
    best_coeffs[0] = 0.5   # (1,0) mode
    best_coeffs[1] = 0.5   # (0,1) mode
    best_coeffs[4] = 0.3   # (2,0) mode
    best_coeffs[8] = 0.5   # radial
    
    best_cell = _make_cell_fourier(best_coeffs, size)
    best_score = _evaluate_gap_fast(best_cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    temperature = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    while _remaining() > 10.0:
        # Pick a random coefficient to perturb
        d = np.random.randint(n_coeffs)
        delta = np.random.normal(0, 0.1)
        
        new_coeffs = current_coeffs.copy()
        new_coeffs[d] += delta
        # Clamp to reasonable range
        new_coeffs[d] = np.clip(new_coeffs[d], -2.0, 2.0)
        
        new_cell = _make_cell_fourier(new_coeffs, size)
        new_score = _evaluate_gap_fast(new_cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
        
        # Metropolis criterion
        delta_s = new_score - current_score
        if delta_s > 0 or np.random.random() < math.exp(delta_s / max(temperature, 1e-6)):
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_coeffs = new_coeffs.copy()
                best_cell = new_cell
        
        # Cool down
        temperature = max(temperature * cooling_rate, min_temp)
    
    return best_cell, best_score

def _search_pixel_sa(size=32, eps_low=1.0, eps_high=12.0, init_cell=None):
    """
    Simulated annealing on pixels, starting from init_cell.
    Uses the fast evaluator.
    """
    Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw = _build_pw_evaluator(
        size, eps_low, eps_high, pw_half=3, n_k=12)
    
    if init_cell is None:
        # Initialize with 4 dielectric disks at quarter-points
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        radius = 4.0
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        cell = _enforce_c4v(cell, size)
    else:
        cell = init_cell.copy()
    
    current_score = _evaluate_gap_fast(cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
    best_cell = cell.copy()
    best_score = current_score
    
    temperature = 0.3
    cooling_rate = 0.998
    min_temp = 0.005
    
    # Precompute orbits for speed
    orbits = []
    for i in range(size):
        for j in range(size):
            pts = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            orbit = list(set(pts))
            orbits.append(orbit)
    
    while _remaining() > 10.0:
        # Pick a random orbit to flip
        idx = np.random.randint(len(orbits))
        orbit = orbits[idx]
        
        new_cell = cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = _evaluate_gap_fast(new_cell, Gx_arr, Gy_arr, G_diff_idx_h, G_diff_idx_k, k_points, N_pw, eps_low, eps_high)
        
        delta_s = new_score - current_score
        if delta_s > 0 or np.random.random() < math.exp(delta_s / max(temperature, 1e-6)):
            cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = new_cell.copy()
        
        temperature = max(temperature * cooling_rate, min_temp)
    
    return best_cell, best_score

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Seed: centred dielectric rod radius 0.2 (guaranteed valid, C4v symmetric)
    seed = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        seed.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    if _remaining() < 20.0:
        return seed
    
    best_cell = None
    best_score = -1e30
    
    # Strategy 1: Fourier-based search
    try:
        if _remaining() > 60.0:
            cell1, score1 = _search_fourier(size, eps_low, eps_high)
            if score1 > best_score:
                best_score = score1
                best_cell = cell1
    except Exception:
        pass
    
    # Strategy 2: Pixel-level SA starting from Fourier result or default
    try:
        if _remaining() > 30.0:
            init = best_cell if best_cell is not None else None
            cell2, score2 = _search_pixel_sa(size, eps_low, eps_high, init_cell=init)
            if score2 > best_score:
                best_score = score2
                best_cell = cell2
    except Exception:
        pass
    
    # Strategy 3: Pixel-level SA starting from 4-disk configuration
    try:
        if _remaining() > 30.0:
            # Create 4-disk init
            cell_init = np.zeros((size, size), dtype=int)
            centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
            radius = 4.0
            for cy, cx in centers:
                for i in range(size):
                    for j in range(size):
                        dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                        if dist < radius:
                            cell_init[i, j] = 1
            cell_init = _enforce_c4v(cell_init, size)
            
            cell3, score3 = _search_pixel_sa(size, eps_low, eps_high, init_cell=cell_init)
            if score3 > best_score:
                best_score = score3
                best_cell = cell3
    except Exception:
        pass
    
    # Fallback to seed if nothing found
    if best_cell is None:
        return seed
    
    # Final C4v enforcement
    c = np.asarray(best_cell, dtype=int)
    c = _enforce_c4v(c, size)
    
    # Verify symmetry
    if not np.array_equal(c, c.T):
        c = (c + c.T) // 2
    if not np.array_equal(c, c[::-1, :]):
        c = (c + c[::-1, :]) // 2
    if not np.array_equal(c, c[:, ::-1]):
        c = (c + c[:, ::-1]) // 2
    
    return c.tolist()
