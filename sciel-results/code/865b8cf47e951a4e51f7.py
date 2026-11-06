"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def design(problem):
    """Design a photonic crystal unit cell to maximize TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 215  # 215s deadline, 15s margin
    
    def _make_cell_from_params(params, sz):
        """Create a 32x32 cell from shape parameters with 4-fold symmetry."""
        r0, a4, a8 = params
        cell = []
        for i in range(sz):
            row = []
            for j in range(sz):
                x = (j + 0.5) / sz - 0.5
                y = (i + 0.5) / sz - 0.5
                r = np.sqrt(x * x + y * y)
                theta = np.arctan2(y, x)
                r_bound = r0 + a4 * np.cos(4 * theta) + a8 * np.cos(8 * theta)
                row.append(1 if r < r_bound else 0)
            cell.append(row)
        return cell
    
    def _compute_gap(params, n_pw_max, n_k_pts):
        """Compute the gap-midgap ratio using plane-wave method."""
        cell = _make_cell_from_params(params, size)
        eps = np.array(cell, dtype=float)
        inv_eps = np.where(eps > 1, 1.0 / eps_high, 1.0)
        
        # Compute 2D FFT of 1/eps
        fft_1_over_eps = np.fft.fft2(inv_eps) / (size * size)
        fft_1_over_eps = np.fft.fftshift(fft_1_over_eps)
        
        # Build G vectors
        g_vecs = []
        for mx in range(-n_pw_max, n_pw_max + 1):
            for my in range(-n_pw_max, n_pw_max + 1):
                if mx * mx + my * my <= n_pw_max * n_pw_max:
                    g_vecs.append((mx, my))
        g_vecs = np.array(g_vecs, dtype=float)
        n_pw = len(g_vecs)
        
        # Precompute coefficient matrix [1/eps_tilde](G_i - G_j)
        coeff_matrix = np.zeros((n_pw, n_pw), dtype=complex)
        for i in range(n_pw):
            gi_x, gi_y = g_vecs[i]
            for j in range(n_pw):
                gj_x, gj_y = g_vecs[j]
                dx = int(round(gi_x - gj_x))
                dy = int(round(gi_y - gj_y))
                ix = (dx + size // 2) % size
                iy = (dy + size // 2) % size
                coeff_matrix[i, j] = fft_1_over_eps[ix, iy]
        
        # Define k-points in irreducible BZ
        k_points = [(0.0, 0.0), (np.pi, 0.0), (np.pi, np.pi)]
        remaining = n_k_pts - 3
        if remaining > 0:
            n_div = max(1, remaining // 3)
            for t in np.linspace(0.1, 0.9, n_div):
                k_points.append((t * np.pi, 0.0))
                k_points.append((np.pi, t * np.pi))
                k_points.append((t * np.pi, t * np.pi))
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kg2 = (kx + g_vecs[:, 0]) ** 2 + (ky + g_vecs[:, 1]) ** 2
            M = coeff_matrix * kg2[np.newaxis, :]
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        
        if max_omega4 > 0 and min_omega5 < np.inf:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2
            ratio = gap / midgap if midgap > 0 else 0
        else:
            ratio = -1
        
        return ratio
    
    def _fast_gap(params):
        return _compute_gap(params, n_pw_max=6, n_k_pts=12)
    
    # Initial guess: disk with small harmonics
    x0 = [0.2, 0.01, 0.005]
    best_params = x0.copy()
    best_score = -np.inf
    
    try:
        s = _fast_gap(x0)
        best_score = s
    except:
        pass
    
    def objective(params):
        try:
            return -_fast_gap(params)
        except:
            return 1e10
    
    # Nelder-Mead optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4,
                                 'disp': False})
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
    except:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return _make_cell_from_params(best_params, size)
    
    # Try additional random starts
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > deadline - 15:
            break
        x_try = [0.15 + np.random.rand() * 0.15,
                 np.random.rand() * 0.04 - 0.02,
                 np.random.rand() * 0.02 - 0.01]
        try:
            s = _fast_gap(x_try)
            if s > best_score:
                best_score = s
                best_params = x_try
                result = minimize(objective, x_try, method='Nelder-Mead',
                                 options={'maxiter': 40, 'xatol': 1e-4, 'fatol': 1e-4,
                                         'disp': False})
                if result.fun < -best_score:
                    best_params = result.x
                    best_score = -result.fun
        except:
            pass
    
    return _make_cell_from_params(best_params, size)

# EVOLVE-BLOCK-END
