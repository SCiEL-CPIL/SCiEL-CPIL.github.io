import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def make_cell(params):
        """
        Generate a symmetric 32x32 cell from parametric primitives.
        params = (ring_width, cross_width, corner_width)
        - ring_width: width of concentric square ring (0-0.5)
        - cross_width: width of dielectric cross through center (0-0.5)  
        - corner_width: width of corner dielectric blocks (0-0.5)
        """
        ring_w, cross_w, corner_w = params
        cell = [[0] * size for _ in range(size)]
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                ax, ay = abs(x), abs(y)
                
                is_dielectric = False
                
                # Concentric square ring: dielectric where |x| or |y| is in [ring_inner, ring_outer]
                # This creates a square annulus
                if ring_w > 0.01:
                    # Ring occupies where max(|x|,|y|) is between ring_w/2 and ring_w
                    r_max = max(ax, ay)
                    if ring_w / 2 < r_max < ring_w:
                        is_dielectric = True
                
                # Cross: dielectric where |x| < cross_w/2 OR |y| < cross_w/2
                if cross_w > 0.01:
                    if ax < cross_w / 2 or ay < cross_w / 2:
                        is_dielectric = True
                
                # Corner blocks: dielectric in four corners
                if corner_w > 0.01:
                    if ax > 0.5 - corner_w and ay > 0.5 - corner_w:
                        is_dielectric = True
                
                cell[i][j] = 1 if is_dielectric else 0
        
        return cell
    
    def make_cell_quadrant(params):
        """
        Alternative: four quadrant rods for 2x2 periodicity.
        params = (rod_radius, rod_spacing)
        Creates 4 circular rods in quadrants.
        """
        rod_r, spacing = params
        cell = [[0] * size for _ in range(size)]
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                ax, ay = abs(x), abs(y)
                
                is_dielectric = False
                if rod_r > 0.01:
                    # Four rods centered at (±spacing, ±spacing)
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            dx = x - sx * spacing
                            dy = y - sy * spacing
                            if dx * dx + dy * dy < rod_r * rod_r:
                                is_dielectric = True
                
                cell[i][j] = 1 if is_dielectric else 0
        
        return cell

    def compute_gap_coarse(cell, n_g=5, n_k=8):
        """
        Coarse-grained gap computation using fewer G-vectors and k-points.
        Much faster for local search.
        """
        def get_fourier_coefficients_coarse(cell, n_g):
            coeffs = np.zeros((2*n_g+1, 2*n_g+1), dtype=complex)
            term_x = np.zeros((size, 2*n_g+1), dtype=complex)
            term_y = np.zeros((size, 2*n_g+1), dtype=complex)
            
            for n in range(-n_g, n_g+1):
                idx = n + n_g
                G = 2 * np.pi * n
                if G == 0:
                    term_x[:, idx] = 1.0 / size
                    term_y[:, idx] = 1.0 / size
                else:
                    sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                    base_exp = np.exp(-1j * G * 0.5 / size)
                    i_vals = np.arange(size)
                    exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                    term_x[:, idx] = sinc_factor * exp_terms
                    term_y[:, idx] = sinc_factor * exp_terms
            
            M = np.array(cell, dtype=float)
            f_coeffs = term_x.T @ (M @ term_y)
            
            delta_eps = eps_high - eps_low
            eps_coeffs = f_coeffs * delta_eps
            eps_coeffs[n_g, n_g] += eps_low
            
            return eps_coeffs

        eps_coeffs = get_fourier_coefficients_coarse(cell, n_g)
        
        # G vectors
        G_vecs = []
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        N_G = len(G_vecs)
        
        # K points: coarse grid in IBZ
        k_points = []
        for i in range(n_k):
            for j in range(n_k):
                kx = i / (n_k-1) * np.pi
                ky = j / (n_k-1) * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
        
        all_omegas = []
        
        # Pre-build E indices
        E_indices = np.zeros((N_G, N_G), dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
                else:
                    E_indices[idx_g, idx_gp] = -1
        
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    def compute_gap(cell):
        """
        Exact gap computation with 289 G-vectors and 30 k-points.
        """
        def get_fourier_coefficients_exact(cell):
            n_g = 8
            coeffs = np.zeros((17, 17), dtype=complex)
            term_x = np.zeros((size, 17), dtype=complex)
            term_y = np.zeros((size, 17), dtype=complex)
            
            for n in range(-8, 9):
                idx = n + 8
                G = 2 * np.pi * n
                if G == 0:
                    term_x[:, idx] = 1.0 / size
                    term_y[:, idx] = 1.0 / size
                else:
                    sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                    base_exp = np.exp(-1j * G * 0.5 / size)
                    i_vals = np.arange(size)
                    exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                    term_x[:, idx] = sinc_factor * exp_terms
                    term_y[:, idx] = sinc_factor * exp_terms
            
            M = np.array(cell, dtype=float)
            f_coeffs = term_x.T @ (M @ term_y)
            
            delta_eps = eps_high - eps_low
            eps_coeffs = f_coeffs * delta_eps
            eps_coeffs[8, 8] += eps_low
            
            return eps_coeffs

        eps_coeffs = get_fourier_coefficients_exact(cell)
        
        G_vecs = []
        for nx in range(-8, 9):
            for ny in range(-8, 9):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        N_G = 289
        
        # 30 k-points covering IBZ boundary and interior
        k_points = []
        # Boundary: kx=0, ky from 0 to pi (10 points)
        for i in range(10):
            k_points.append((0, i / 9.0 * np.pi))
        # Boundary: ky=pi, kx from 0 to pi (10 points)
        for i in range(10):
            k_points.append((i / 9.0 * np.pi, np.pi))
        # Boundary: kx=ky, from (0,0) to (pi,pi) (5 points, excluding endpoints already counted)
        for i in range(1, 5):
            k_points.append((i / 4.0 * np.pi, i / 4.0 * np.pi))
        # Interior points (5 more)
        k_points.append((np.pi/4, np.pi/2))
        k_points.append((np.pi/3, 2*np.pi/3))
        k_points.append((np.pi/6, np.pi/3))
        k_points.append((np.pi/5, 3*np.pi/5))
        k_points.append((np.pi/2.5, 7*np.pi/10))
        
        all_omegas = []
        
        E_indices = np.zeros((N_G, N_G), dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -8 <= dx <= 8 and -8 <= dy <= 8:
                    E_indices[idx_g, idx_gp] = (dx + 8) * 17 + (dy + 8)
                else:
                    E_indices[idx_g, idx_gp] = -1
        
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            if time.time() > deadline:
                break
                
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio

    # Phase 1: Coarse search with parametric primitives
    # Use coarse evaluator for speed
    
    best_coarse_score = -1.0
    best_params = None
    best_cell = None
    
    # Initial grid search over parametric space
    # ring_width, cross_width, corner_width
    ring_vals = [0.1, 0.15, 0.2, 0.25, 0.3]
    cross_vals = [0.0, 0.05, 0.1, 0.15]
    corner_vals = [0.0, 0.05, 0.1, 0.15]
    
    for rw in ring_vals:
        if time.time() > deadline - 30:
            break
        for cw in cross_vals:
            if time.time() > deadline - 30:
                break
            for cw2 in corner_vals:
                if time.time() > deadline - 30:
                    break
                params = (rw, cw, cw2)
                cell = make_cell(params)
                score = compute_gap_coarse(cell, n_g=5, n_k=8)
                
                if score > best_coarse_score:
                    best_coarse_score = score
                    best_params = params
                    best_cell = cell
    
    # Also try quadrant rods
    rod_r_vals = [0.08, 0.1, 0.12, 0.15]
    spacing_vals = [0.15, 0.2, 0.25]
    
    for rr in rod_r_vals:
        if time.time() > deadline - 30:
            break
        for sp in spacing_vals:
            if time.time() > deadline - 30:
                break
            params = (rr, sp)
            cell = make_cell_quadrant(params)
            score = compute_gap_coarse(cell, n_g=5, n_k=8)
            
            if score > best_coarse_score:
                best_coarse_score = score
                best_params = params
                best_cell = cell
                # Mark as quadrant type
                best_cell_is_quadrant = True
            else:
                best_cell_is_quadrant = False
    
    # Phase 2: Local refinement around best coarse solution
    if best_params is not None and time.time() < deadline - 20:
        # Try small perturbations
        if len(best_params) == 3:
            base_rw, base_cw, base_cw2 = best_params
            perturbations = [
                (base_rw + 0.02, base_cw, base_cw2),
                (base_rw - 0.02, base_cw, base_cw2),
                (base_rw, base_cw + 0.02, base_cw2),
                (base_rw, base_cw - 0.02, base_cw2),
                (base_rw, base_cw, base_cw2 + 0.02),
                (base_rw, base_cw, base_cw2 - 0.02),
                (base_rw + 0.01, base_cw + 0.01, base_cw2),
                (base_rw - 0.01, base_cw + 0.01, base_cw2),
                (base_rw + 0.01, base_cw, base_cw2 + 0.01),
                (base_rw, base_cw + 0.01, base_cw2 + 0.01),
            ]
            for p in perturbations:
                if time.time() > deadline - 20:
                    break
                if p[0] < 0.01 or p[1] < 0 or p[2] < 0:
                    continue
                cell = make_cell(p)
                score = compute_gap_coarse(cell, n_g=5, n_k=8)
                if score > best_coarse_score:
                    best_coarse_score = score
                    best_params = p
                    best_cell = cell
        else:
            base_rr, base_sp = best_params
            perturbations = [
                (base_rr + 0.01, base_sp),
                (base_rr - 0.01, base_sp),
                (base_rr, base_sp + 0.02),
                (base_rr, base_sp - 0.02),
                (base_rr + 0.01, base_sp + 0.01),
                (base_rr - 0.01, base_sp + 0.01),
                (base_rr + 0.01, base_sp - 0.01),
                (base_rr - 0.01, base_sp - 0.01),
            ]
            for p in perturbations:
                if time.time() > deadline - 20:
                    break
                if p[0] < 0.01 or p[1] < 0.05:
                    continue
                cell = make_cell_quadrant(p)
                score = compute_gap_coarse(cell, n_g=5, n_k=8)
                if score > best_coarse_score:
                    best_coarse_score = score
                    best_params = p
                    best_cell = cell
    
    # Phase 3: Evaluate top candidates with exact method
    if time.time() < deadline - 10:
        # Collect unique cells from search
        candidate_cells = []
        
        # Add best from coarse search
        if best_cell is not None:
            candidate_cells.append((best_cell, best_coarse_score))
        
        # Add some diverse candidates
        test_params = [
            (0.15, 0.1, 0.05),
            (0.2, 0.05, 0.1),
            (0.25, 0.1, 0.0),
            (0.1, 0.15, 0.1),
            (0.18, 0.08, 0.08),
        ]
        for p in test_params:
            if time.time() > deadline - 10:
                break
            cell = make_cell(p)
            score = compute_gap_coarse(cell, n_g=5, n_k=8)
            candidate_cells.append((cell, score))
        
        # Sort by coarse score and evaluate top ones exactly
        candidate_cells.sort(key=lambda x: x[1], reverse=True)
        
        best_exact_score = -1.0
        best_exact_cell = best_cell
        
        for cell, coarse_score in candidate_cells[:5]:  # Top 5
            if time.time() > deadline - 5:
                break
            exact_score = compute_gap(cell)
            if exact_score > best_exact_score:
                best_exact_score = exact_score
                best_exact_cell = cell
        
        if best_exact_cell is not None:
            return best_exact_cell
    
    # Fallback
    if best_cell is None:
        # Simple circle as fallback
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.04:  # r=0.2
                    cell[i][j] = 1
        return cell
    
    return best_cell
