"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_inv_eps_fourier(cell, n_pw=17):
    """Compute Fourier coefficients of 1/eps for the 32x32 cell."""
    size = cell.shape[0]
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    half = n_pw // 2
    m_vals = np.arange(-half, half + 1)
    
    # 2D FFT
    fft_2d = np.fft.fft2(inv_eps)
    
    G = np.zeros((n_pw, n_pw), dtype=complex)
    for im, m in enumerate(m_vals):
        for in_, n in enumerate(m_vals):
            kx = m % size
            ky = n % size
            G[im, in_] = fft_2d[kx, ky] / (size * size)
    
    return G

def _band_structure(cell, k_points, n_pw=17, max_band=5):
    """Compute TM band structure using plane-wave expansion."""
    G = _compute_inv_eps_fourier(cell, n_pw)
    half = n_pw // 2
    m_vals = np.arange(-half, half + 1)
    n_pw_total = n_pw * n_pw
    
    n_k = len(k_points)
    bands = np.zeros((n_k, max_band))
    
    # Precompute G-G' indices
    # For each pair (im, in_, im2, in2), compute dm = m - m2, dn = n - n2
    # and check if |dm| <= half, |dn| <= half
    
    for ik, (kx, ky) in enumerate(k_points):
        M = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        
        for im, m in enumerate(m_vals):
            for in_, n in enumerate(m_vals):
                idx1 = im * n_pw + in_
                for im2, m2 in enumerate(m_vals):
                    for in2, n2 in enumerate(m_vals):
                        idx2 = im2 * n_pw + in2
                        dot = (kx + m) * (kx + m2) + (ky + n) * (ky + n2)
                        dm = m - m2
                        dn = n - n2
                        if abs(dm) <= half and abs(dn) <= half:
                            im_g = dm + half
                            in_g = dn + half
                            coeff = G[im_g, in_g]
                        else:
                            coeff = 0.0
                        M[idx1, idx2] = dot * coeff
        
        try:
            eigvals = np.linalg.eigvalsh(M)
            pos_eig = eigvals[eigvals > 1e-10]
            if len(pos_eig) >= max_band:
                bands[ik, :] = np.sqrt(pos_eig[:max_band])
            else:
                bands[ik, :] = np.sqrt(pos_eig)
                bands[ik, len(pos_eig):] = np.inf
        except:
            bands[ik, :] = np.nan
    
    return bands

def _irreducible_k_points(n_k=30):
    """Generate k-points in the irreducible Brillouin zone."""
    k_points = []
    # Irreducible BZ: triangle with vertices (0,0), (0.5, 0.5), (0, 0.5)
    # Use a grid
    n = 8
    for i in range(n + 1):
        for j in range(n + 1 - i):
            kx = i / n * 0.5
            ky = (i + j) / n * 0.5
            if kx <= ky and ky <= 0.5:
                k_points.append((kx, ky))
    
    if len(k_points) > n_k:
        indices = np.linspace(0, len(k_points) - 1, n_k).astype(int)
        k_points = [k_points[i] for i in indices]
    elif len(k_points) < n_k:
        while len(k_points) < n_k:
            kx = np.random.uniform(0, 0.5)
            ky = np.random.uniform(kx, 0.5)
            k_points.append((kx, ky))
    
    return k_points[:n_k]

def _make_symmetric_cell(quadrant):
    """Create a 32x32 symmetric cell from a 16x16 top-left quadrant."""
    size = 32
    half = 16
    q = np.array(quadrant, dtype=int)
    
    cell = np.zeros((size, size), dtype=int)
    cell[:half, :half] = q
    cell[:half, half:] = q.T
    cell[half:, :half] = q[::-1, :]
    cell[half:, half:] = q.T[::-1, :]
    
    return cell.tolist()

def _evaluate_design(cell, k_points=None, n_pw=17):
    """Evaluate the gap-midgap ratio."""
    if k_points is None:
        k_points = _irreducible_k_points(30)
    
    bands = _band_structure(cell, k_points, n_pw=n_pw, max_band=5)
    
    if np.any(np.isnan(bands)) or np.any(np.isinf(bands)):
        return -1.0
    
    omega_4 = bands[:, 3]
    omega_5 = bands[:, 4]
    
    max_omega_4 = np.max(omega_4)
    min_omega_5 = np.min(omega_5)
    
    if min_omega_5 + max_omega_4 <= 0:
        return -1.0
    
    return 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)

def design(problem):
    """Design a photonic crystal unit cell to maximize the TM band gap."""
    start_time = time.time()
    deadline = start_time + 215.0
    
    size = problem["pixels"]
    k_points = _irreducible_k_points(30)
    
    # Reference design
    cell0 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.04:
                cell0[i, j] = 1
    
    best_score = _evaluate_design(cell0, k_points)
    best_cell = cell0.copy()
    best_q = cell0[:16, :16].copy().astype(float)
    
    # Try different initial designs
    designs = [
        # Rod r=0.2
        np.array([[1 if ((i+0.5)/32-0.5)**2 + ((j+0.5)/32-0.5)**2 < 0.04 else 0 for j in range(16)] for i in range(16)], dtype=float),
        # Rod r=0.25
        np.array([[1 if ((i+0.5)/32-0.5)**2 + ((j+0.5)/32-0.5)**2 < 0.0625 else 0 for j in range(16)] for i in range(16)], dtype=float),
        # Rod r=0.3
        np.array([[1 if ((i+0.5)/32-0.5)**2 + ((j+0.5)/32-0.5)**2 < 0.09 else 0 for j in range(16)] for i in range(16)], dtype=float),
        # Square rod
        np.array([[1 if 3 <= i < 13 and 3 <= j < 13 else 0 for j in range(16)] for i in range(16)], dtype=float),
        # Ring
        np.array([[1 if 0.01 < ((i+0.5)/32-0.5)**2 + ((j+0.5)/32-0.5)**2 < 0.0625 else 0 for j in range(16)] for i in range(16)], dtype=float),
    ]
    
    for q_init in designs:
        cell = np.array(_make_symmetric_cell(q_init))
        score = _evaluate_design(cell, k_points)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            best_q = q_init.copy()
    
    # Optimize
    def _obj(params):
        if time.time() - start_time > deadline:
            return 1e10
        cell = np.array(_make_symmetric_cell((params.reshape(16, 16) > 0.5).astype(int)))
        return -_evaluate_design(cell, k_points)
    
    try:
        result = minimize(_obj, best_q.flatten(), method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 0.05, 'fatol': 1e-3})
        if result.fun < -best_score:
            best_q = result.x.reshape(16, 16)
            best_cell = np.array(_make_symmetric_cell((best_q > 0.5).astype(int)))
            best_score = -result.fun
    except:
        pass
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
