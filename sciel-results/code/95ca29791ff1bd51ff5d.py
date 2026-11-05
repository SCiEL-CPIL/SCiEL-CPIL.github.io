"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.linalg import eigh

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using the standard Galerkin method.
    2. Use a parametric representation with symmetry-constrained geometric primitives.
    3. Perform a coarse-grained search over the parameter space.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    # Time budget: 230s limit, keep 20s margin for overhead
    start_time = time.time()
    deadline = start_time + 210.0

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=15, n_kpts=20):
        """
        Compute TM band structure using plane-wave expansion.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell  # epsilon map
        inv_eps = 1.0 / eps
        N = size
        
        # Compute Fourier coefficients of 1/epsilon
        fft_coeffs = np.fft.fft2(inv_eps)
        
        def get_coeff(m, n):
            """Get Fourier coefficient for G = 2π*(m, n)"""
            idx_m = m % N
            idx_n = n % N
            return fft_coeffs[idx_m, idx_n] / (N * N)
        
        # Build the list of G vectors
        m_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        if len(m_vals) < n_pw:
            m_vals = m_vals[:n_pw]
        else:
            m_vals = m_vals[-n_pw:]
        
        g_indices = []
        for mi, m in enumerate(m_vals):
            for ni, n in enumerate(m_vals):
                g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = 1/eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=float)
        for i, (mi, ni) in enumerate(g_indices):
            for j, (mj, nj) in enumerate(g_indices):
                gm = mi - mj
                gn = ni - nj
                S[i, j] = get_coeff(gm, gn).real
        
        # Define k-points in the irreducible Brillouin zone
        k_points = []
        n_seg = max(1, n_kpts // 3)
        # Gamma to X: (0, t) for t in [0, 1]
        for t in np.linspace(0, 1, n_seg + 1):
            k_points.append((0.0, t))
        # X to M: (t, 1) for t in [0, 1]
        for t in np.linspace(0, 1, n_seg + 1):
            k_points.append((t, 1.0))
        # M to Gamma: (t, t) for t in [1, 0]
        for t in np.linspace(1, 0, n_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        n_kpts_actual = len(k_points_unique)
        omega = np.zeros((n_kpts_actual, n_total))
        
        n_bands_needed = lower_band + 1
        
        for ki, (kx, ky) in enumerate(k_points_unique):
            # Build A and B matrices
            A = np.zeros((n_total, n_total), dtype=float)
            for i in range(n_total):
                for j in range(n_total):
                    mj, nj = g_indices[j]
                    gm = 2 * mj + kx
                    gn = 2 * nj + ky
                    g_sq = gm * gm + gn * gn
                    A[i, j] = S[i, j] * g_sq
            
            B = S.copy()
            
            # Solve generalized eigenvalue problem: A c = λ B c
            try:
                eigenvalues = eigh(A, B, eigvals_only=True)
                eigenvalues = np.sort(eigenvalues)
                omega[ki, :n_bands_needed] = np.sqrt(np.maximum(eigenvalues[:n_bands_needed], 0))
            except:
                try:
                    B_inv = np.linalg.inv(B)
                    M = B_inv @ A
                    eigenvalues = np.linalg.eigvals(M)
                    eigenvalues = np.sort(np.real(eigenvalues))
                    omega[ki, :n_bands_needed] = np.sqrt(np.maximum(eigenvalues[:n_bands_needed], 0))
                except:
                    omega[ki, :] = np.nan
        
        return omega
    
    # Parametric cell generation using symmetry-constrained primitives
    def make_parametric_cell(params):
        """
        Create a cell from parametric geometric primitives.
        params: dict with keys for different primitive parameters
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords)
        
        # Central dielectric square
        if params.get('central_size', 0) > 0:
            cs = params['central_size']
            cell[np.abs(X) < cs/2, np.abs(Y) < cs/2] = 1
        
        # Four corner dielectric squares
        if params.get('corner_size', 0) > 0 and params.get('corner_dist', 0) > 0:
            cs = params['corner_size']
            cd = params['corner_dist']
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    mask = (np.abs(X - sx*cd) < cs/2) & (np.abs(Y - sy*cd) < cs/2)
                    cell[mask] = 1
        
        # Four edge-center dielectric squares
        if params.get('edge_size', 0) > 0 and params.get('edge_dist', 0) > 0:
            es = params['edge_size']
            ed = params['edge_dist']
            for sx in [-1, 1]:
                mask = (np.abs(X - sx*ed) < es/2) & (np.abs(Y) < es/2)
                cell[mask] = 1
                mask = (np.abs(X) < es/2) & (np.abs(Y - sx*ed) < es/2)
                cell[mask] = 1
        
        # Cross-shaped dielectric bars
        if params.get('cross_width', 0) > 0 and params.get('cross_length', 0) > 0:
            cw = params['cross_width']
            cl = params['cross_length']
            # Horizontal bar
            cell[np.abs(Y) < cw/2, np.abs(X) < cl/2] = 1
            # Vertical bar
            cell[np.abs(X) < cw/2, np.abs(Y) < cl/2] = 1
        
        # Air holes in the dielectric regions
        if params.get('hole_size', 0) > 0:
            hs = params['hole_size']
            # Central air hole
            if params.get('central_hole', False):
                cell[np.abs(X) < hs/2, np.abs(Y) < hs/2] = 0
            # Corner air holes
            if params.get('corner_hole', False) and params.get('corner_dist', 0) > 0:
                cd = params['corner_dist']
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        mask = (np.abs(X - sx*cd) < hs/2) & (np.abs(Y - sy*cd) < hs/2)
                        cell[mask] = 0
        
        return cell.tolist()
    
    # Score a candidate cell
    def score_cell(cell, n_pw=13, n_kpts=15):
        try:
            omega = compute_tm_bands(cell, n_pw=n_pw, n_kpts=n_kpts)
            if np.any(np.isnan(omega)):
                return -np.inf
            max_omega_4 = np.max(omega[:, 3])
            min_omega_5 = np.min(omega[:, 4])
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2.0
            if midgap > 0:
                return 2.0 * gap / midgap
            else:
                return -np.inf
        except:
            return -np.inf
    
    # Coarse search over parameter space
    best_score = -np.inf
    best_params = None
    
    # Parameter ranges for coarse search
    central_sizes = [0.0, 0.1, 0.2, 0.3]
    corner_sizes = [0.0, 0.08, 0.12, 0.16]
    corner_dists = [0.25, 0.3, 0.35]
    edge_sizes = [0.0, 0.08, 0.12]
    edge_dists = [0.25, 0.3, 0.35]
    cross_widths = [0.0, 0.05, 0.1]
    cross_lengths = [0.0, 0.2, 0.3, 0.4]
    hole_sizes = [0.0, 0.05, 0.1]
    central_holes = [False, True]
    corner_holes = [False, True]
    
    n_pw = 13
    n_kpts = 15
    
    # Coarse grid search
    for cs in central_sizes:
        for corr_s in corner_sizes:
            for corr_d in corner_dists:
                for es in edge_sizes:
                    for ed in edge_dists:
                        for cw in cross_widths:
                            for cl in cross_lengths:
                                for hs in hole_sizes:
                                    for ch in central_holes:
                                        for co_h in corner_holes:
                                            if time.time() > deadline - 15:
                                                break
                                            
                                            params = {
                                                'central_size': cs,
                                                'corner_size': corr_s,
                                                'corner_dist': corr_d,
                                                'edge_size': es,
                                                'edge_dist': ed,
                                                'cross_width': cw,
                                                'cross_length': cl,
                                                'hole_size': hs,
                                                'central_hole': ch,
                                                'corner_hole': co_h
                                            }
                                            
                                            # Skip invalid combinations
                                            if corr_s > 0 and corr_d < 0.2:
                                                continue
                                            if es > 0 and ed < 0.2:
                                                continue
                                            if cw > 0 and cl < 0.1:
                                                continue
                                            
                                            try:
                                                cell = make_parametric_cell(params)
                                                score = score_cell(cell, n_pw=n_pw, n_kpts=n_kpts)
                                                if score > best_score:
                                                    best_score = score
                                                    best_params = params
                                            except:
                                                continue
                        
                        if time.time() > deadline - 15:
                            break
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break
    if time.time() > deadline - 15:
        break
    
    # Fine-tune around the best parameters if we have time
    if best_params is not None and time.time() < deadline - 30:
        # Local refinement around the best parameters
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            
            # Try small perturbations
            for delta in [-0.02, 0.02]:
                for key in ['central_size', 'corner_size', 'edge_size', 'cross_width', 'cross_length', 'hole_size']:
                    if key in best_params and best_params[key] > 0:
                        new_params = best_params.copy()
                        new_params[key] = max(0, new_params[key] + delta)
                        try:
                            cell = make_parametric_cell(new_params)
                            score = score_cell(cell, n_pw=n_pw, n_kpts=n_kpts)
                            if score > best_score:
                                best_score = score
                                best_params = new_params
                        except:
                            pass
    
    # Generate the final cell from the best parameters
    if best_params is not None:
        best_candidate = make_parametric_cell(best_params)
    else:
        # Fallback: simple four-quadrant rods
        best_candidate = make_parametric_cell({
            'central_size': 0,
            'corner_size': 0.12,
            'corner_dist': 0.25,
            'edge_size': 0,
            'edge_dist': 0,
            'cross_width': 0,
            'cross_length': 0,
            'hole_size': 0,
            'central_hole': False,
            'corner_hole': False
        })
    
    return best_candidate

# EVOLVE-BLOCK-END
