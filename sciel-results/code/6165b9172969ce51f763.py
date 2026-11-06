"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 200s limit

    def expand_d4(cell_quad):
        """
        Expand an 8x8 fundamental domain to a 32x32 cell with D4 symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(8):
            for j in range(8):
                val = cell_quad[i, j]
                cell[i, j] = val
                cell[i, 31-j] = val
                cell[31-i, j] = val
                cell[31-i, 31-j] = val
                cell[j, i] = val
                cell[j, 31-i] = val
                cell[31-j, i] = val
                cell[31-j, 31-i] = val
        return cell

    def compute_fourier_coefficients(cell, max_order):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps = np.zeros((size, size))
        eps[cell == 1] = eps_high
        eps[cell == 0] = eps_low
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                m_idx = m % size
                n_idx = n % size
                coeffs[(m, n)] = fft_2d[m_idx, n_idx] / (size * size)
        return coeffs

    def approximate_gap_ratio(cell, max_order=8, g_range=8):
        """
        Approximate the gap ratio between bands 4 and 5.
        Uses G-range 8 (289 waves) to match evaluator basis size.
        """
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        g_vectors = [(m, n) for m in range(-g_range, g_range + 1) for n in range(-g_range, g_range + 1)]
        N = len(g_vectors)
        
        k_points = [
            (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
            (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1),
            (0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2),
            (0.1, 0.3), (0.2, 0.3), (0.3, 0.3), (0.4, 0.3), (0.5, 0.3),
            (0.1, 0.4), (0.2, 0.4), (0.3, 0.4), (0.4, 0.4), (0.5, 0.4),
            (0.5, 0.5)
        ]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            H = np.zeros((N, N), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    H[i, j] = k_sq * eps_inv_G
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    def generate_cell_from_params(params):
        """
        Generate a 32x32 cell from continuous parameters.
        params: [w0..w9, threshold]
        """
        w = params[:10]
        threshold = params[10]
        
        u, v = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        
        phi0 = np.ones_like(u)
        phi1 = np.exp(-((u - 0.5)**2 + (v - 0.5)**2) / 0.1)
        phi2 = np.exp(-(u**2 + v**2) / 0.1)
        phi3 = np.exp(-((u - 0.5)**2 + v**2) / 0.1)
        phi4 = np.exp(-(u**2 + (v - 0.5)**2) / 0.1)
        phi5 = np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v)
        phi6 = np.cos(4 * np.pi * u) * np.cos(4 * np.pi * v)
        phi7 = np.sin(2 * np.pi * u) * np.sin(2 * np.pi * v)
        phi8 = np.exp(-((u-0.25)**2 + (v-0.25)**2) / 0.05)
        phi9 = np.exp(-((u-0.75)**2 + (v-0.75)**2) / 0.05)
        
        F = w[0]*phi0 + w[1]*phi1 + w[2]*phi2 + w[3]*phi3 + w[4]*phi4 + w[5]*phi5 + w[6]*phi6 + w[7]*phi7 + w[8]*phi8 + w[9]*phi9
        
        cell_quad = (F > threshold).astype(int)
        return expand_d4(cell_quad)

    def objective(params):
        if time.time() > deadline:
            return 1.0
        cell = generate_cell_from_params(params)
        score = approximate_gap_ratio(cell)
        return -score

    x0 = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    
    best_score = -1.0
    best_cell = None
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < -best_score:
            best_score = -res.fun
            best_cell = generate_cell_from_params(res.x)
    except Exception:
        pass

    if best_cell is not None and time.time() < deadline - 20:
        base_cell = best_cell
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            best_local = best_score
            best_local_cell = base_cell
            
            for i in range(8):
                for j in range(8):
                    if time.time() > deadline - 5:
                        break
                    
                    quad = base_cell[0:8, 0:8].copy()
                    quad[i, j] = 1 - quad[i, j]
                    cell_trial = expand_d4(quad)
                    
                    if not np.array_equal(cell_trial, base_cell):
                        score = approximate_gap_ratio(cell_trial)
                        if score > best_local + 1e-6:
                            best_local = score
                            best_local_cell = cell_trial
                            improved = True
                
                if time.time() > deadline - 5:
                    break
            
            if improved:
                best_score = best_local
                best_cell = best_local_cell
                base_cell = best_cell

    if best_cell is None:
        best_cell = expand_d4(np.ones((8, 8), dtype=int))
        
    final_cell = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return final_cell

# EVOLVE-BLOCK-END
