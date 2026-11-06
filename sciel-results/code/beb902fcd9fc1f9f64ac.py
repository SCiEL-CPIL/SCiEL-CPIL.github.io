"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(size, offset, radius, shape_param=0.0):
    """Generate a 4-rod symmetric cell."""
    cell = np.zeros((size, size), dtype=int)
    centers = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if shape_param < 0.5:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radius:
                        cell[i, j] = 1
                else:
                    p = 2.0 + 2.0 * shape_param
                    val = (abs(dx)/radius)**p + (abs(dy)/radius)**p
                    if val < 1.0:
                        cell[i, j] = 1
    return _apply_symmetry(cell).tolist()

def _make_checkerboard_cell(size, Px, Py, Dx, Dy, phi_x, phi_y):
    """
    Generate a cell from the product of two 1D square wave gratings.
    """
    cell = np.zeros((size, size), dtype=int)
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    
    ux = (xs + 0.5 * Px) % Px
    Sy_x = (np.abs(ux - 0.5 * Px) < 0.5 * Dx * Px).astype(int)
    
    uy = (ys + 0.5 * Py) % Py
    Sy_y = (np.abs(uy - 0.5 * Py) < 0.5 * Dy * Py).astype(int)
    
    cell_2d = np.outer(Sy_x, Sy_y)
    
    return _apply_symmetry(cell_2d).tolist()

def _compute_eps_ft_fast(cell, size, pw_idx):
    """
    Compute 2D Fourier coefficients of epsilon field for given plane-wave indices.
    Uses FFT for speed.
    """
    eps = np.array(cell, dtype=float)
    eps = 1.0 + (12.0 - 1.0) * eps
    
    # We need coefficients for G = (m, n) where m, n in pw_idx.
    # eps_ft[m, n] = 1/N^2 sum_{x,y} eps(x,y) exp(-i 2 pi (m x + n y) / N)
    # This is exactly the 2D DFT (scaled by 1/N^2).
    # np.fft.fftn computes sum exp(-i 2 pi ...).
    # So eps_ft = np.fft.fftn(eps) / (size * size)
    
    full_fft = np.fft.fftn(eps) / (size * size)
    
    n_pw = len(pw_idx)
    eps_ft = np.zeros((n_pw, n_pw), dtype=complex)
    
    # Map pw_idx values to FFT indices
    # FFT index k corresponds to frequency k if k < N/2, else k - N
    # We need to extract specific frequencies.
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            # Convert frequency m to FFT index
            idx_m = m % size
            idx_n = n % size
            eps_ft[im, in_] = full_fft[idx_m, idx_n]
            
    return eps_ft

def _tm_eigenvalues_fast(kx, ky, eps_ft, pw_idx):
    """
    Solve TM eigenvalue problem at a single k-point using vectorized operations.
    H_{ij} = (k_i . k_j) * eps_{G_i - G_j}
    where k_i = k + G_i.
    """
    n_pw = len(pw_idx)
    N = n_pw * n_pw
    
    # Create arrays of G vectors
    # G_i = (m_i, n_i)
    # We need to flatten the 2D grid of Gs into 1D
    Gx = np.repeat(pw_idx, n_pw)
    Gy = np.tile(pw_idx, n_pw)
    
    # k_i = k + G_i
    kx_g = kx + Gx
    ky_g = ky + Gy
    k_g_sq = kx_g**2 + ky_g**2
    
    # We need the matrix M where M_{ij} = k_i . k_j = (kx_i + kx_j) * 0.5? No.
    # k_i . k_j = (kx_i)(kx_j) + (ky_i)(ky_j)
    # Let's construct the dot product matrix.
    # K is N x 2 matrix of k_i vectors.
    K = np.column_stack((kx_g, ky_g))
    K_dot_K = K @ K.T
    
    # Now we need to multiply each element K_dot_K[i, j] by eps_{G_i - G_j}
    # G_i - G_j = (Gx_i - Gx_j, Gy_i - Gy_j)
    # We need to map (dm, dn) to index in eps_ft.
    
    # Create difference matrices
    # Gx_diff[i, j] = Gx[i] - Gx[j]
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    
    # Map differences to indices in eps_ft
    # eps_ft is indexed by (im, in_) corresponding to pw_idx[im], pw_idx[in_]
    # We need to find indices im, in_ such that pw_idx[im] == Gx_diff[i, j] and pw_idx[in_] == Gy_diff[i, j]
    
    # Create lookup tables for pw_idx values
    # Since pw_idx is a range of integers, we can use direct indexing if we shift.
    # Let min_pw = pw_idx[0], max_pw = pw_idx[-1]
    # The difference dm can range from min_pw - max_pw to max_pw - min_pw.
    # However, eps_ft only contains coefficients for differences that are in pw_idx?
    # NO. The convolution theorem says we need eps_ft for ALL differences G_i - G_j that appear.
    # In the Galerkin method, the matrix element is:
    # H_{ij} = (k_i . k_j) * \epsilon_{G_i - G_j}
    # \epsilon_{G} is the Fourier coefficient of epsilon at wavevector G.
    # If G_i - G_j is not in our set of plane waves, we still need its Fourier coefficient!
    # The previous code checked `if dm in pw_map`. This is INCORRECT for the exact method.
    # The exact method requires \epsilon_{G} for all G in the set of differences.
    # However, if we use a finite basis, we usually assume \epsilon_{G} = 0 if G is large.
    # But for a pixelated dielectric, \epsilon_{G} decays as 1/G.
    # The "exact-coefficient" method in the prompt implies we should use the true coefficients.
    # If we only computed eps_ft for pw_idx, we are missing coefficients for differences outside pw_idx.
    
    # Let's re-read the prompt: "exact-coefficient plane-wave method".
    # This usually means the matrix H_{ij} = (k+G_i) \cdot (k+G_j) \epsilon_{G_i-G_j}.
    # If we truncate the basis to N plane waves, we still need \epsilon_{G} for all G_i-G_j.
    # The differences G_i-G_j can be up to 2*max_pw.
    # So we need to compute eps_ft for a larger range of Gs.
    
    # Let's modify _compute_eps_ft_fast to return a dictionary or a larger array.
    # Or better, compute the full FFT and index into it directly.
    
    # Let's refactor to use the full FFT result directly in the eigenvalue solver.
    # This avoids the intermediate eps_ft array limitation.
    
    return None # Placeholder, will be replaced by logic below

def _tm_eigenvalues_full(kx, ky, full_eps_fft, size, pw_idx):
    """
    Solve TM eigenvalue problem using full FFT for epsilon coefficients.
    """
    n_pw = len(pw_idx)
    N = n_pw * n_pw
    
    Gx = np.repeat(pw_idx, n_pw)
    Gy = np.tile(pw_idx, n_pw)
    
    kx_g = kx + Gx
    ky_g = ky + Gy
    k_g_sq = kx_g**2 + ky_g**2
    
    K = np.column_stack((kx_g, ky_g))
    K_dot_K = K @ K.T
    
    Gx_diff = Gx[:, None] - Gx[None, :]
    Gy_diff = Gy[:, None] - Gy[None, :]
    
    # Map differences to FFT indices
    # FFT index for frequency f is f % size
    idx_m = Gx_diff % size
    idx_n = Gy_diff % size
    
    # Extract epsilon coefficients from full FFT
    # full_eps_fft is size x size
    eps_coeffs = full_eps_fft[idx_m, idx_n]
    
    H = K_dot_K * eps_coeffs
    
    # H should be Hermitian. Due to numerical errors or symmetry, it might be slightly complex.
    # We take the real part for eigvalsh? Or use eigh?
    # The matrix is Hermitian if eps is real.
    # H_{ij} = k_i.k_j eps_{G_i-G_j}
    # H_{ji} = k_j.k_i eps_{G_j-G_i} = k_i.k_j eps_{-(G_i-G_j)} = k_i.k_j conj(eps_{G_i-G_j}) (since eps is real)
    # So H is Hermitian.
    
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)

def _quick_gap_estimate(cell, size=32, n_pw=7, n_kpts=8):
    """Fast surrogate evaluation using reduced plane-wave basis."""
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
        
    eps = np.array(cell, dtype=float)
    eps = 1.0 + (12.0 - 1.0) * eps
    full_eps_fft = np.fft.fftn(eps) / (size * size)
    
    k_points = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.25), (0.4, 0.1), (0.1, 0.4)
    ][:n_kpts]
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        eigvals = _tm_eigenvalues_full(kx, ky, full_eps_fft, size, pw_idx)
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

def _random_symmetric_cell(size, fill_prob):
    """Generate a random symmetric binary cell."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i <= j and i + j < size:
                val = 1 if random.random() < fill_prob else 0
                cell[i, j] = val
                cell[j, i] = val
                cell[size-1-i, size-1-j] = val
                cell[size-1-j, size-1-i] = val
    return cell

def _mutate_cell(cell, n_flips):
    """Randomly flip n pixels while maintaining symmetry."""
    cell = cell.copy()
    size = cell.shape[0]
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
        for pi, pj in positions:
            cell[pi, pj] = new_val
    return cell

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0 # 120s limit, leave 10s margin
    random.seed(42)
    
    best_cell = None
    best_score = -np.inf
    
    # Phase 1: Parametric Seeds (Checkerboards and Rods)
    seeds = []
    
    # Checkerboard seeds: Target (2,0) and (2,2) modes
    for Px in [0.4, 0.5, 0.6]:
        for Py in [0.4, 0.5, 0.6]:
            for Dx in [0.3, 0.4, 0.5]:
                for Dy in [0.3, 0.4, 0.5]:
                    seeds.append(_make_checkerboard_cell(size, Px, Py, Dx, Dy, 0.5, 0.5))
    
    # Rod seeds
    for offset in [0.20, 0.25, 0.30]:
        for radius in [0.10, 0.14, 0.18]:
            seeds.append(_make_four_rod_cell(size, offset, radius, 0.0))
            seeds.append(_make_four_rod_cell(size, offset, radius, 1.0))
    
    for cell in seeds:
        if time.time() > deadline - 50.0:
            break
        score = _quick_gap_estimate(cell, size)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Stochastic search from best seed
    if best_cell is not None:
        best_arr = np.array(best_cell)
        current_arr = best_arr.copy()
        current_score = best_score
            
        # Local search
        while time.time() < deadline - 10.0:
            n_flips = 1
            # Occasionally try larger jumps
            if random.random() < 0.1:
                n_flips = 3
                    
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
                
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
            else:
                pass
    
    # Phase 3: Random restarts
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        # Mix of random and parametric restarts
        if random.random() < 0.5:
            fill_prob = random.uniform(0.2, 0.5)
            rand_cell = _random_symmetric_cell(size, fill_prob)
        else:
            rand_cell = _make_checkerboard_cell(
                size, 
                random.uniform(0.4, 0.6), 
                random.uniform(0.4, 0.6), 
                random.uniform(0.3, 0.5), 
                random.uniform(0.3, 0.5), 
                0.5, 0.5
            )
                
        rand_score = _quick_gap_estimate(rand_cell, size)
            
        if rand_score > best_score:
            best_score = rand_score
            best_cell = rand_cell
                
        current_arr = np.array(rand_cell)
        current_score = rand_score
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            n_flips = 1
            trial = _mutate_cell(current_arr, n_flips)
            trial_score = _quick_gap_estimate(trial.tolist(), size)
            if trial_score > current_score:
                current_arr = trial
                current_score = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_cell = trial.tolist()
    
    if best_cell is None:
        best_cell = _make_four_rod_cell(size, 0.25, 0.12, 0.0)
    
    return best_cell
# EVOLVE-BLOCK-END
