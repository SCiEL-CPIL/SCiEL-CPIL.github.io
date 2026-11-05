import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

def construct():
    start_time = time.time()
    timeout = 135.0  # 15s margin before 150s kill

    def _elapsed():
        return time.time() - start_time

    n = 8

    def log_product_exact(points):
        """
        Compute log product of pairwise distances after scaling to diameter 2.
        Uses exact max for diameter.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)

        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]

        if len(off_diag_dists) == 0:
            return -np.inf

        diameter = np.max(off_diag_dists)
        if diameter < 1e-10:
            return -np.inf

        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)

        return np.sum(np.log(scaled_dists))

    def log_product_smooth(points, beta=100.0):
        """
        Compute log product of pairwise distances after scaling to diameter 2.
        Uses soft-max for diameter when beta is finite.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)

        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]

        if len(off_diag_dists) == 0:
            return -np.inf

        if beta < 1e6:
            scaled = off_diag_dists / beta
            m = np.max(scaled)
            log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
            diameter = beta * log_sum_exp
        else:
            diameter = np.max(off_diag_dists)

        if diameter < 1e-10:
            return -np.inf

        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)

        return np.sum(np.log(scaled_dists))

    def objective_smooth(x, beta=100.0):
        points = x.reshape(n, 2)
        lp = log_product_smooth(points, beta)
        return -lp

    def objective_exact(x):
        points = x.reshape(n, 2)
        lp = log_product_exact(points)
        return -lp

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = log_product_exact(reg_oct)
    best_points = reg_oct.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations (finer grid)
    for rot in np.linspace(0.0, 0.5, 30):
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs (more seeds)
    for seed_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Antipodal pairs with structured angles (equally spaced base)
    for offset in np.linspace(0.0, 0.5, 10):
        base_angles = np.sort(np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4]) + offset)
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Random points
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Points on circle with perturbations
    for _ in range(10):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.15, 0.15, n)
        radii = 1.0 + np.random.uniform(-0.15, 0.15, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # Phase 1: Local optimization with soft-max surrogate (L-BFGS-B)
    # Evaluate all starts first, then optimize the best ones
    start_scores = []
    for s in starts:
        try:
            sc = log_product_exact(s.reshape(n, 2))
        except Exception:
            sc = -np.inf
        start_scores.append((sc, s))
    
    # Sort by score descending, take top starts for optimization
    start_scores.sort(key=lambda x: -x[0])
    top_starts = [s for sc, s in start_scores[:25]]
    
    for s in top_starts:
        if time.time() - start_time > timeout - 35:
            break
        try:
            res = minimize(
                lambda x: objective_smooth(x, beta=50.0),
                s,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < np.inf:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 2: Refine with higher beta
    if time.time() - start_time < timeout - 30:
        for beta_val in [100.0, 200.0, 500.0, 1000.0, 2000.0]:
            if time.time() - start_time > timeout - 25:
                break
            try:
                res = minimize(
                    lambda x: objective_smooth(x, beta=beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 3: Global optimization with differential evolution
    if time.time() - start_time < timeout - 40:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: objective_smooth(x, beta=100.0),
                bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-10,
                polish=True
            )
            if de_res.fun < np.inf:
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_points = de_res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 4: High-precision polish with exact diameter
    if time.time() - start_time < timeout - 10:
        for method in ['L-BFGS-B', 'BFGS']:
            if time.time() - start_time > timeout - 8:
                break
            try:
                final_res = minimize(
                    objective_exact,
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if final_res.fun < np.inf:
                    score = -final_res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 5: Multi-start polish - perturb best solution and re-optimize
    if _elapsed() < timeout - 20:
        np.random.seed(123)
        for perturb_scale in [0.05, 0.02, 0.01, 0.005]:
            if _elapsed() > timeout - 15:
                break
            for trial in range(3):
                if _elapsed() > timeout - 15:
                    break
                perturbed = best_points + np.random.normal(0, perturb_scale, (n, 2))
                try:
                    res_p = minimize(
                        lambda x: objective_smooth(x, beta=500.0),
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res_p.fun < np.inf:
                        score_p = -res_p.fun
                        if score_p > best_score:
                            best_score = score_p
                            best_points = res_p.x.reshape(n, 2)
                except Exception:
                    pass
        def poly_discriminant_log(coeffs):
            """
            coeffs: array of 8 real coefficients [c0, c1, ..., c7]
            P(z) = z^8 + c7*z^7 + c6*z^6 + ... + c1*z + c0
            Returns -log(discriminant) + penalty for roots outside unit disk
            """
            # Build polynomial coefficients for numpy.roots
            # P(z) = z^8 + c7*z^7 + ... + c0
            poly_coeffs = np.concatenate([[1.0], coeffs[::-1]])
            try:
                roots = np.roots(poly_coeffs)
            except Exception:
                return 1e10
            
            # Check all roots are distinct (not too close)
            root_dists = []
            for i in range(8):
                for j in range(i+1, 8):
                    d = np.abs(roots[i] - roots[j])
                    if d < 1e-8:
                        return 1e10
                    root_dists.append(d)
            
            if len(root_dists) == 0:
                return 1e10
            
            # Log product of all pairwise distances
            log_prod = sum(np.log(d) for d in root_dists)
            
            # Penalty: max modulus should be close to 1 (diameter = 2*max_mod)
            # We want diameter = 2, so max|root| = 1
            max_mod = np.max(np.abs(roots))
            # Soft penalty for deviation from max_mod = 1
            penalty = 50.0 * (max_mod - 1.0)**2
            
            # Also penalize if min distance is too small
            min_d = min(root_dists)
            if min_d < 0.01:
                penalty += 100.0 * (0.01 - min_d)**2
            
            return -log_prod + penalty

        # Try multiple starting points for polynomial optimization
        poly_starts = []
        
        # Regular octagon as polynomial
        angles_reg = 2 * np.pi * np.arange(8) / 8
        roots_reg = np.exp(1j * angles_reg)
        # Compute coefficients from roots
        poly_reg = np.poly(roots_reg)
        # Normalize: leading coefficient should be 1
        poly_reg = poly_reg / poly_reg[0]
        # Coefficients c7, c6, ..., c0 are poly_reg[1:9]
        coeffs_reg = np.real(poly_reg[1:9])[::-1]  # reverse to get [c0, c1, ..., c7]
        poly_starts.append(coeffs_reg)
        
        # Two-squares configurations
        for rot in [0.1, 0.2, 0.3, 0.4, np.pi/8]:
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
            roots_sq = np.concatenate([np.exp(1j * a1), np.exp(1j * a2)])
            poly_sq = np.poly(roots_sq)
            poly_sq = poly_sq / poly_sq[0]
            coeffs_sq = np.real(poly_sq[1:9])[::-1]
            poly_starts.append(coeffs_sq)
        
        # Antipodal pairs
        for seed_val in [7, 8, 9]:
            np.random.seed(seed_val)
            base_angles = np.sort(np.random.uniform(0, np.pi, 4))
            all_angles = np.concatenate([base_angles, base_angles + np.pi])
            roots_ant = np.exp(1j * all_angles)
            poly_ant = np.poly(roots_ant)
            poly_ant = poly_ant / poly_ant[0]
            coeffs_ant = np.real(poly_ant[1:9])[::-1]
            poly_starts.append(coeffs_ant)
        
        # Optimize each polynomial start
        for ps in poly_starts:
            if _elapsed() > timeout - 15:
                break
            try:
                res_poly = minimize(
                    poly_discriminant_log,
                    ps,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_poly.fun < 1e9:
                    # Convert back to points
                    poly_coeffs_final = np.concatenate([[1.0], res_poly.x[::-1]])
                    roots_final = np.roots(poly_coeffs_final)
                    # Check validity
                    max_mod = np.max(np.abs(roots_final))
                    min_d = min(np.abs(roots_final[i] - roots_final[j]) 
                               for i in range(8) for j in range(i+1, 8))
                    if min_d > 1e-6 and max_mod > 1e-6:
                        pts_poly = np.column_stack([np.real(roots_final), np.imag(roots_final)])
                        score = log_product_exact(pts_poly)
                        if score > best_score:
                            best_score = score
                            best_points = pts_poly.copy()
            except Exception:
                pass

    # Phase 6: Dedicated optimization for two-squares configuration
    if _elapsed() < timeout - 5:
        def two_squares_score(theta):
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
            sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
            sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
            pts = np.vstack([sq1, sq2])
            return -log_product_exact(pts)

        # Minimize over theta in [0, pi/4] by symmetry
        try:
            theta_res = minimize_scalar(
                two_squares_score,
                bounds=(0.0, np.pi/4),
                method='bounded',
                options={'xatol': 1e-12}
            )
            if theta_res.success:
                theta = theta_res.x
                a1 = 2 * np.pi * np.arange(4) / 4
                a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
                sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
                sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
                pts = np.vstack([sq1, sq2])
                score = log_product_exact(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    # Additional high-precision polish: try multiple methods with tighter tolerances
    if _elapsed() < timeout - 3:
        polish_methods = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}),
            ('BFGS', {'maxiter': 500, 'gtol': 1e-14}),
            ('CG', {'maxiter': 500, 'tol': 1e-14}),
        ]
        for method_name, opts in polish_methods:
            if _elapsed() > timeout - 2:
                break
            try:
                final_res2 = minimize(
                    objective_exact,
                    best_points.flatten(),
                    method=method_name,
                    options=opts
                )
                if final_res2.fun < np.inf:
                    score2 = -final_res2.fun
                    if score2 > best_score:
                        best_score = score2
                        best_points = final_res2.x.reshape(n, 2)
            except Exception:
                pass

    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
