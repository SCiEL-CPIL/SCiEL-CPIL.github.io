"""Design a symmetric 32x32-pixel unit cell for a 2D square-lattice photonic crystal
that maximizes the TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell for a 2D square-lattice photonic crystal.
    Uses level-set / spline-based parameterization with Nelder-Mead optimization
    to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    time_limit = 200.0
    start_time = time.time()
    
    # Irreducible Brillouin zone k-points
    def get_k_points():
        """Return k-points in the irreducible BZ for square lattice."""
        k_points = []
        # Γ to X: (k, 0)
        for k in np.linspace(0, 0.5, 10):
            k_points.append((k, 0))
        # X to M: (0.5, k)
        for k in np.linspace(0, 0.5, 10):
            k_points.append((0.5, k))
        # M to Γ: (0.5-k, 0.5-k)
        for k in np.linspace(0, 0.5, 10):
            k_points.append((0.5 - k, 0.5 - k))
        return k_points
    
    k_points = get_k_points()
    
    def generate_cell(params):
        """
        Generate a symmetric 32x32 cell from parameters.
        params[0]: base radius
        params[1]: superellipse exponent
        params[2]: modulation amplitude
        params[3]: modulation frequency
        """
        cell = [[0 for _ in range(size)] for _ in range(size)]
        cx = cy = size / 2.0
        
        base_r = params[0]
        n_exp = max(1.0, params[1])  # Ensure exponent >= 1
        mod_amp = params[2]
        mod_freq = max(1, int(round(params[3])))
        
        for i in range(size):
            for j in range(size):
                dx = (j + 0.5) - cx
                dy = (i + 0.5) - cy
                r = math.sqrt(dx*dx + dy*dy)
                if r == 0:
                    theta = 0
                else:
                    theta = math.atan2(dy, dx)
                
                # Compute boundary radius at this angle
                cos_t = abs(math.cos(theta))
                sin_t = abs(math.sin(theta))
                
                # Superellipse boundary
                if n_exp > 1.0:
                    boundary_r = base_r / ((cos_t ** (2.0/n_exp)) + (sin_t ** (2.0/n_exp))) ** (1.0/n_exp)
                else:
                    boundary_r = base_r
                
                # Add modulation
                boundary_r *= (1.0 + mod_amp * math.cos(mod_freq * theta))
                
                if r < boundary_r:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
        
        # Enforce C4v symmetry
        for i in range(size):
            for j in range(size):
                # Transpose symmetry: cell[i][j] == cell[j][i]
                if i != j:
                    avg = (cell[i][j] + cell[j][i]) // 2
                    cell[i][j] = avg
                    cell[j][i] = avg
                # Inversion symmetry: cell[i][j] == cell[size-1-i][size-1-j]
                i_inv = size - 1 - i
                j_inv = size - 1 - j
                if i != i_inv or j != j_inv:
                    avg = (cell[i][j] + cell[i_inv][j_inv]) // 2
                    cell[i][j] = avg
                    cell[i_inv][j_inv] = avg
        
        return cell
    
    def compute_band_gap(cell):
        """
        Compute the TM band structure using plane-wave method.
        Returns the gap-midgap ratio between bands 4 and 5.
        """
        N = 17  # 17x17 = 289 plane waves
        N_half = N // 2  # 8
        
        # Get reciprocal lattice vectors
        G_vectors = []
        for m in range(-N_half, N_half + 1):
            for n in range(-N_half, N_half + 1):
                G_vectors.append((2.0 * math.pi * m, 2.0 * math.pi * n))
        
        NG = len(G_vectors)  # 289
        
        # Compute Fourier coefficients of f(r) = cell (0 or 1)
        f_fourier = np.zeros(NG, dtype=complex)
        for gi, (gx, gy) in enumerate(G_vectors):
            f_g = 0.0
            for i in range(size):
                for j in range(size):
                    if cell[i][j] == 1:
                        x = (j + 0.5) / size
                        y = (i + 0.5) / size
                        f_g += math.exp(-1j * (gx * x + gy * y))
            f_fourier[gi] = f_g / (size * size)
        
        # Inverse permittivity parameters
        delta_eps_inv = 1.0/eps_low - 1.0/eps_high
        
        def compute_eigenvalues(kx, ky):
            """Compute TM eigenvalues for given k-point."""
            H = np.zeros((NG, NG), dtype=complex)
            
            for gi in range(NG):
                gx, gy = G_vectors[gi]
                for gj in range(NG):
                    gx2, gy2 = G_vectors[gj]
                    k2_gj = (kx + gx2)**2 + (ky + gy2)**2
                    
                    # <G_i|1/eps|G_j> = (1/eps_low)*delta_{ij} - delta_eps_inv * f_{G_i - G_j}
                    dx = gx - gx2
                    dy = gy - gy2
                    dm = int(round(dx / (2.0 * math.pi)))
                    dn = int(round(dy / (2.0 * math.pi)))
                    
                    if -N_half <= dm <= N_half and -N_half <= dn <= N_half:
                        idx = (dm + N_half) * N + (dn + N_half)
                        f_coeff = f_fourier[idx]
                    else:
                        f_coeff = 0.0
                    
                    if gi == gj:
                        matrix_elem = (1.0/eps_low) - delta_eps_inv * f_coeff
                    else:
                        matrix_elem = -delta_eps_inv * f_coeff
                    
                    H[gi, gj] = matrix_elem * k2_gj
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigvals = eigenvalues[lower_band-1:lower_band+1]
                w4 = math.sqrt(max(eigvals[0], 0))
                w5 = math.sqrt(max(eigvals[1], 0))
                return w4, w5
            except Exception:
                return None
        
        # Compute gap-midgap ratio over all k-points
        max_band4 = 0.0
        min_band5 = float('inf')
        
        for kx, ky in k_points:
            result = compute_eigenvalues(kx, ky)
            if result is None:
                continue
            w4, w5 = result
            max_band4 = max(max_band4, w4)
            min_band5 = min(min_band5, w5)
        
        if min_band5 == float('inf'):
            return -1.0
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap == 0:
            return -1.0
        
        return gap / midgap
    
    # Initial parameters
    initial_params = [0.3, 4.0, 0.0, 4.0]
    
    # Precompute cell for initial params
    best_cell = generate_cell(initial_params)
    best_score = compute_band_gap(best_cell)
    
    # Nelder-Mead optimization
    def objective(params):
        """Objective function to minimize (negative of the gap ratio)."""
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            return 1.0
        
        cell = generate_cell(params)
        score = compute_band_gap(cell)
        return -score
    
    # Try to optimize
    try:
        res = minimize(objective, initial_params, method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False})
        optimized_params = res.x
        cell_test = generate_cell(optimized_params)
        score_test = compute_band_gap(cell_test)
        if score_test > best_score:
            best_score = score_test
            best_cell = cell_test
    except Exception:
        pass
    
    # If we still have time, try a few random perturbations
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        random.seed(42)
        for _ in range(20):
            if time.time() - start_time > time_limit - 15:
                break
            perturb = [random.uniform(-0.1, 0.1), random.uniform(-0.5, 0.5),
                      random.uniform(-0.1, 0.1), random.uniform(-1, 1)]
            params = [initial_params[i] + perturb[i] for i in range(4)]
            params[0] = max(0.1, min(0.5, params[0]))
            params[1] = max(1.0, min(10.0, params[1]))
            params[2] = max(-0.5, min(0.5, params[2]))
            params[3] = max(1, min(8, params[3]))
            
            cell = generate_cell(params)
            score = compute_band_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    return best_cell
# EVOLVE-BLOCK-END
