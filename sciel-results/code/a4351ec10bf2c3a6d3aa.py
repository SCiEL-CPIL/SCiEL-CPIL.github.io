"""Photonic crystal unit cell design for TM band gap optimization."""
import numpy as np
import time
import math

_DEADLINE = None

def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry."""
    n = len(cell)
    c = np.array(cell, dtype=np.float64)
    sym = c + c[::-1, :] + c[:, ::-1] + c[::-1, ::-1] + c.T + c[::-1, :].T + c[:, ::-1].T + c[::-1, ::-1].T
    sym /= 8.0
    return (sym >= 0.5).astype(int)

def _get_g_vectors(n_pw):
    """Get G-vector indices and their 2D coordinates."""
    g_idx = list(range(-n_pw, n_pw + 1))
    n_g = len(g_idx)
    g_coords = np.array([(gx, gy) for gx in g_idx for gy in g_idx], dtype=np.float64)
    return g_idx, g_coords, n_g

def _pwe_tm_bands_fast(cell, n_pw=13, n_bands=6, k_points=None):
    """Fast PWE TM band solver using vectorized operations."""
    n = len(cell)
    eps_vals = 1.0 + 11.0 * np.array(cell, dtype=np.float64)
    
    # Compute Fourier coefficients via FFT
    fft_vals = np.fft.fft2(eps_vals) / (n * n)
    
    g_idx, g_coords, n_g = _get_g_vectors(n_pw)
    n_total = n_g * n_g
    
    if k_points is None:
        k_points = [(t, 0.0) for t in np.linspace(0, 0.5, 6)] + \
                   [(0.5, t) for t in np.linspace(0, 0.5, 6)] + \
                   [(t, t) for t in np.linspace(0.5, 0, 6)]
    
    # Precompute G-G' differences
    # For each pair (ig, jg), compute dgx = gx1 - gx2, dgy = gy1 - gy2
    gx_arr = g_coords[:, 0]
    gy_arr = g_coords[:, 1]
    
    # Build difference matrix
    dgx_mat = gx_arr[:, None] - gx_arr[None, :]  # (n_total, n_total)
    dgy_mat = gy_arr[:, None] - gy_arr[None, :]
    
    # Map to FFT indices
    fx_mat = np.mod(dgx_mat, n).astype(int)
    fy_mat = np.mod(dgy_mat, n).astype(int)
    
    # Get epsilon coefficients for all G-G' pairs
    eps_coeff = fft_vals[fx_mat, fy_mat]  # (n_total, n_total)
    
    # Replace near-zero with large value to avoid division issues
    eps_coeff_safe = np.where(np.abs(eps_coeff) > 1e-10, eps_coeff, 1e10)
    
    results = []
    for kx, ky in k_points:
        # k+G vectors
        kx_g = kx + gx_arr
        ky_g = ky + gy_arr
        k_sq = kx_g ** 2 + ky_g ** 2  # (n_total,)
        
        # M[ig, jg] = k_sq[ig] / eps_coeff[ig, jg]
        M = k_sq[:, None] / eps_coeff_safe
        
        # Solve eigenvalue problem
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
            positive = eigvals > 1e-8
            if np.sum(positive) >= n_bands:
                omega = np.sqrt(eigvals[positive][:n_bands])
                ev4 = eigvecs[:, positive][:, 3] if np.sum(positive) >= 4 else np.zeros(n_total, dtype=np.complex128)
                ev5 = eigvecs[:, positive][:, 4] if np.sum(positive) >= 5 else np.zeros(n_total, dtype=np.complex128)
            else:
                omega = np.zeros(n_bands)
                ev4 = np.zeros(n_total, dtype=np.complex128)
                ev5 = np.zeros(n_total, dtype=np.complex128)
        except np.linalg.LinAlgError:
            omega = np.zeros(n_bands)
            ev4 = np.zeros(n_total, dtype=np.complex128)
            ev5 = np.zeros(n_total, dtype=np.complex128)
        
        results.append((kx, ky, omega, ev4, ev5))
    
    return results, g_idx, g_coords

def _compute_gap(results):
    """Compute gap-midgap ratio between bands 4 and 5."""
    max_4 = -np.inf
    min_5 = np.inf
    for kx, ky, omega, ev4, ev5 in results:
        if len(omega) >= 5:
            max_4 = max(max_4, omega[3])
            min_5 = min(min_5, omega[4])
    if max_4 == -np.inf or min_5 == np.inf:
        return -1.0
    midgap = (min_5 + max_4) / 2.0
    if midgap < 1e-10:
        return -1.0
    return 2.0 * (min_5 - max_4) / midgap

def _field_intensity_vec(ev, g_coords, n):
    """Compute field intensity from eigenvector (vectorized)."""
    n_total = len(ev)
    # field[i,j] = sum_G ev[G] * exp(2*pi*i*(Gx*i + Gy*j)/n)
    i_idx = np.arange(n)
    j_idx = np.arange(n)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    # Phase: 2*pi*(Gx*I + Gy*J)/n
    phase = 2j * np.pi * (g_coords[:, 0][:, None, None] * I[None, :, :] + 
                           g_coords[:, 1][:, None, None] * J[None, :, :]) / n
    field = np.sum(ev[:, None, None] * np.exp(phase), axis=0)
    return np.abs(field) ** 2

def _optimize(cell, max_time=100.0, n_pw=11):
    """Topological optimization."""
    n = len(cell)
    start = time.time()
    
    # Fewer k-points for speed
    k_points = [(t, 0.0) for t in np.linspace(0, 0.5, 5)] + \
               [(0.5, t) for t in np.linspace(0, 0.5, 5)] + \
               [(t, t) for t in np.linspace(0.5, 0, 5)]
    
    current = np.array(cell, dtype=int)
    best_cell = current.copy()
    best_score = -np.inf
    
    results, g_idx, g_coords = _pwe_tm_bands_fast(current, n_pw=n_pw, n_bands=6, k_points=k_points)
    score = _compute_gap(results)
    best_score = score
    
    # Compute accumulated field intensities
    int4 = np.zeros((n, n), dtype=np.float64)
    int5 = np.zeros((n, n), dtype=np.float64)
    for kx, ky, omega, ev4, ev5 in results:
        int4 += _field_intensity_vec(ev4, g_coords, n)
        int5 += _field_intensity_vec(ev5, g_coords, n)
    diff = int4 - int5
    
    # Build symmetric groups
    visited = set()
    groups = []
    for i in range(n):
        for j in range(n):
            if (i, j) in visited:
                continue
            pos = set()
            for ii in [i, n-1-i]:
                for jj in [j, n-1-j]:
                    pos.add((ii, jj))
                    pos.add((jj, ii))
            visited.update(pos)
            groups.append(list(pos))
    
    iteration = 0
    while time.time() - start < max_time:
        iteration += 1
        remaining = max_time - (time.time() - start)
        if remaining < 5:
            break
        
        # Sort groups by |diff|
        group_info = []
        for g in groups:
            i0, j0 = g[0]
            group_info.append((abs(diff[i0, j0]), g, diff[i0, j0]))
        group_info.sort(key=lambda x: -x[0])
        
        n_flips = min(10, max(1, int(remaining)))
        flipped = False
        for s, g, d in group_info[:n_flips]:
            if s < 1e-8:
                break
            target = 1 if d > 0 else 0
            needs = any(current[i, j] != target for i, j in g)
            if needs:
                for i, j in g:
                    current[i, j] = target
                flipped = True
        
        if not flipped:
            # Random symmetric flip
            for _ in range(3):
                i = np.random.randint(n)
                j = np.random.randint(n)
                pos = set()
                for ii in [i, n-1-i]:
                    for jj in [j, n-1-j]:
                        pos.add((ii, jj))
                        pos.add((jj, ii))
                for ii, jj in pos:
                    current[ii, jj] = 1 - current[ii, jj]
        
        current = np.array(_enforce_symmetry(current.tolist()), dtype=int)
        
        results, g_idx, g_coords = _pwe_tm_bands_fast(current, n_pw=n_pw, n_bands=6, k_points=k_points)
        score = _compute_gap(results)
        
        if score > best_score:
            best_score = score
            best_cell = current.copy()
        
        # Update intensities periodically
        if iteration % 3 == 0:
            int4 = np.zeros((n, n), dtype=np.float64)
            int5 = np.zeros((n, n), dtype=np.float64)
            for kx, ky, omega, ev4, ev5 in results:
                int4 += _field_intensity_vec(ev4, g_coords, n)
                int5 += _field_intensity_vec(ev5, g_coords, n)
            diff = int4 - int5
    
    return best_cell.tolist(), best_score

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell to maximize TM band gap."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 210.0
    
    size = problem["pixels"]
    
    # Start with a ring design
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            row.append(1 if 0.15 < r < 0.32 else 0)
        cell.append(row)
    
    cell = _enforce_symmetry(cell)
    
    if time.time() > _DEADLINE - 40:
        return cell
    
    try:
        opt_cell, score = _optimize(cell, max_time=100.0, n_pw=11)
        if time.time() < _DEADLINE - 20:
            return opt_cell
    except Exception:
        pass
    
    return cell
