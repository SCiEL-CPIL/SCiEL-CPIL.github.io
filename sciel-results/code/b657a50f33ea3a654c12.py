import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def make_cell_v2(params):
        """
        Generate a symmetric 32x32 cell.
        params: (center_is_dielectric, shape_type, size1, size2, rot_deg, corner_is_dielectric)
        - center_is_dielectric: 1 if center is dielectric, 0 if air
        - shape_type: 0 for square, 1 for circle
        - size1: Half-width (or radius) of central feature (0 to 0.5)
        - size2: Half-width (or radius) of outer feature (0 to 0.5)
        - rot_deg: Rotation angle in degrees (0 to 45)
        - corner_is_dielectric: 1 if corners are dielectric, 0 if air
        """
        c_diel, shape_t, s1, s2, rot, c_diel_corner = params
        cell = [[0] * size for _ in range(size)]
        
        rot_rad = math.radians(rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                ax, ay = abs(x), abs(y)
                
                is_dielectric = 0
                
                # Determine if pixel is in central feature
                # Rotate coordinates back to align with feature
                xr = x * cos_r + y * sin_r
                yr = -x * sin_r + y * cos_r
                axr, ayr = abs(xr), abs(yr)
                
                if shape_t == 0: # Square
                    if max(axr, ayr) < s1:
                        is_dielectric = 1 if c_diel == 1 else 0
                    else:
                        is_dielectric = 0 if c_diel == 1 else 1 # Background is opposite of center
                else: # Circle
                    if (xr*xr + yr*yr) < s1*s1:
                        is_dielectric = 1 if c_diel == 1 else 0
                    else:
                        is_dielectric = 0 if c_diel == 1 else 1

                # Override with outer feature (ring or corner block)
                # If s2 > s1, we have an outer ring/band
                if s2 > s1:
                    if shape_t == 0:
                        if max(axr, ayr) >= s1 and max(axr, ayr) < s2:
                            # This is the ring region
                            # Let's assume the ring is dielectric if center is air, or air if center is dielectric?
                            # Actually, let's just make the ring dielectric if c_diel_corner is 1?
                            # Let's keep it simple: The "background" between center and corner is determined by c_diel.
                            # If center is dielectric (1), background is air (0).
                            # If center is air (0), background is dielectric (1).
                            pass # The logic above already sets background. 
                            # But we want to explicitly control the ring?
                            # Let's refine: 
                            # Region 1: Center (size s1) -> Material A
                            # Region 2: Middle (s1 to s2) -> Material B
                            # Region 3: Corners (outside s2) -> Material C
                            
                            # Let's redefine:
                            # Material A: c_diel
                            # Material B: 1 - c_diel (contrast)
                            # Material C: c_diel_corner
                            
                            pass # We need to re-evaluate based on regions
                            
                    # Re-implementing region logic for clarity
                    if max(axr, ayr) < s1:
                        is_dielectric = 1 if c_diel == 1 else 0
                    elif max(axr, ayr) < s2:
                        # Middle ring
                        is_dielectric = 1 if c_diel == 0 else 0 # Contrast
                    else:
                        # Outer corners
                        is_dielectric = 1 if c_diel_corner == 1 else 0
                        
                else:
                    # No middle ring, just center and corners
                    if max(axr, ayr) < s1:
                        is_dielectric = 1 if c_diel == 1 else 0
                    else:
                        is_dielectric = 1 if c_diel_corner == 1 else 0

                cell[i][j] = is_dielectric
        
        return cell

    def compute_gap_coarse(cell, n_g=5, n_k=8):
        """
        Coarse-grained gap computation using fewer G-vectors and k-points.
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
        
        G_vecs = []
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        N_G = len(G_vecs)
        
        k_points = []
        for i in range(n_k):
            for j in range(n_k):
                kx = i / (n_k-1) * np.pi
                ky = j / (n_k-1) * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
        
        all_omegas = []
        
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
        
        k_points = []
        for i in range(10):
            k_points.append((0, i / 9.0 * np.pi))
        for i in range(10):
            k_points.append((i / 9.0 * np.pi, np.pi))
        for i in range(1, 5):
            k_points.append((i / 4.0 * np.pi, i / 4.0 * np.pi))
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

    def nelder_mead(f, x0, bounds, max_iter=50, tol=1e-4):
        """Simple Nelder-Mead implementation."""
        n = len(x0)
        simplex = [list(x0) for _ in range(n+1)]
        for i in range(n):
            simplex[i+1][i] += 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        
        fx = [f(s) for s in simplex]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            idx = np.argsort(fx)
            simplex = [simplex[i] for i in idx]
            fx = [fx[i] for i in idx]
            
            if abs(fx[0] - fx[-1]) < tol:
                break
            
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            
            # Reflection
            xr = [centroid[j] + 1.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fxr = f(xr)
            
            if fx[0] <= fxr < fx[-2]:
                simplex[-1] = xr
                fx[-1] = fxr
            elif fxr < fx[0]:
                # Expansion
                xe = [centroid[j] + 2.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fxe = f(xe)
                if fxe < fxr:
                    simplex[-1] = xe
                    fx[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx[-1] = fxr
            else:
                # Contraction
                xc = [centroid[j] + 0.5 * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fxc = f(xc)
                if fxc < fx[-1]:
                    simplex[-1] = xc
                    fx[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fx[i] = f(simplex[i])
        
        idx_min = np.argmin(fx)
        return simplex[idx_min], fx[idx_min]

    # Search Strategy
    # 1. Initialize a few diverse starting points
    # 2. Run Nelder-Mead on each
    # 3. Keep the best coarse score
    # 4. Evaluate top candidates exactly
    
    bounds = [
        [0, 1], # c_diel (0 or 1, but NM will handle floats, we round in make_cell? No, make_cell uses 0/1 logic. We should restrict search to discrete or handle carefully. Let's fix c_diel and c_diel_corner to 0 or 1 in the search space by trying combinations.)
        [0, 1], # shape_type
        [0.05, 0.45], # s1
        [0.05, 0.45], # s2
        [0, 45], # rot
        [0, 1] # c_diel_corner
    ]
    
    # We will iterate over discrete choices for c_diel, shape_type, c_diel_corner
    # and optimize continuous vars s1, s2, rot
    
    best_coarse_score = -10.0
    best_params_coarse = None
    best_cell_coarse = None
    
    for c_diel in [0, 1]:
        for shape_t in [0, 1]:
            for c_diel_corner in [0, 1]:
                # Define objective function for continuous vars
                def obj_continuous(s1, s2, rot):
                    params = (c_diel, shape_t, s1, s2, rot, c_diel_corner)
                    cell = make_cell_v2(params)
                    return compute_gap_coarse(cell, n_g=4, n_k=6) # Even coarser for speed
                
                # Initial guess
                x0 = [0.2, 0.3, 45.0]
                
                # Adjust bounds for continuous vars
                b_cont = [
                    [0.05, 0.45],
                    [0.05, 0.45],
                    [0, 45]
                ]
                
                try:
                    x_opt, f_opt = nelder_mead(obj_continuous, x0, b_cont, max_iter=30, tol=1e-3)
                    
                    if f_opt > best_coarse_score:
                        best_coarse_score = f_opt
                        best_params_coarse = (c_diel, shape_t, x_opt[0], x_opt[1], x_opt[2], c_diel_corner)
                        best_cell_coarse = make_cell_v2(best_params_coarse)
                except Exception:
                    continue
                
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # If no good coarse result, fallback to a known good structure
    if best_cell_coarse is None:
        # Fallback: Central dielectric square, air corners
        best_params_coarse = (1, 0, 0.2, 0.4, 0.0, 0)
        best_cell_coarse = make_cell_v2(best_params_coarse)
    
    # Phase 2: Evaluate top candidates with exact method
    # We only have one best from NM. Let's also evaluate a few diverse hand-picked ones.
    candidates = []
    if best_cell_coarse is not None:
        candidates.append((best_cell_coarse, best_coarse_score))
    
    # Add some diverse structures
    test_params = [
        (1, 0, 0.2, 0.4, 45.0, 0), # Diamond center, square corners
        (0, 0, 0.2, 0.4, 0.0, 1), # Air square center, dielectric corners
        (1, 1, 0.2, 0.4, 0.0, 0), # Circle center
        (0, 1, 0.2, 0.4, 0.0, 1), # Air circle center
    ]
    for p in test_params:
        cell = make_cell_v2(p)
        # Quick coarse score to rank
        cs = compute_gap_coarse(cell, n_g=4, n_k=6)
        candidates.append((cell, cs))
    
    # Sort by coarse score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_exact_score = -10.0
    best_exact_cell = best_cell_coarse
    
    for cell, cs in candidates[:3]:
        if time.time() > deadline - 5:
            break
        es = compute_gap(cell)
        if es > best_exact_score:
            best_exact_score = es
            best_exact_cell = cell
    
    if best_exact_cell is None:
        # Final fallback
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.04:
                    cell[i][j] = 1
        return cell
        
    return best_exact_cell
