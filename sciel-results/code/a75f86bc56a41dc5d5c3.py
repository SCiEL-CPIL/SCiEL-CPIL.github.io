import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 8

    def compute_score(points):
        """
        Compute the exact score: product of distances over ordered pairs,
        after scaling so diameter = 2, then divide by n^n.
        Returns the actual score value.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return 0.0
        
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return 0.0
        
        # Scale to diameter 2
        scaled_dists = off_diag_dists * (2.0 / diameter)
        
        # Avoid log(0) issues if points coincide, though distinctness is required
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        # Score is Delta / n^n
        score = np.exp(log_prod - n * np.log(n))
        return score

    def log_discriminant_from_roots(roots):
        """
        Compute log |discriminant| of a monic polynomial with given complex roots.
        The discriminant is prod_{i<j} (z_i - z_j)^2, so log|disc| = 2 * sum_{i<j} log|z_i - z_j|.
        This is proportional to the log-product of pairwise distances.
        """
        # For n=8, compute all pairwise distances
        diff = roots[:, np.newaxis] - roots[np.newaxis, :]
        dists = np.abs(diff)
        
        # Upper triangle indices (i < j)
        iu = np.triu_indices(n, k=1)
        off_dists = dists[iu]
        
        # Clamp to avoid log(0)
        off_dists = np.maximum(off_dists, 1e-15)
        
        # log|disc| = 2 * sum_{i<j} log|z_i - z_j|
        log_disc = 2.0 * np.sum(np.log(off_dists))
        return log_disc

    def poly_objective(coeffs):
        """
        Objective for polynomial optimization.
        coeffs: 16 real values = [Re(c0), Im(c0), Re(c1), Im(c1), ..., Re(c7), Im(c7)]
        We form P(z) = z^8 + c7*z^7 + c6*z^6 + ... + c0
        Find roots, compute log-discriminant, penalize if max |root| > 1.
        """
        if time.time() > deadline - 5:
            return 1e10
        
        # Build complex coefficients
        c = coeffs[0::2] + 1j * coeffs[1::2]
        
        # Polynomial: z^8 + c7*z^7 + c6*z^6 + ... + c0
        # numpy roots expects [leading, ..., constant]
        poly_coeffs = np.concatenate([np.array([1.0+0j]), c[::-1]])
        
        try:
            roots = np.roots(poly_coeffs)
        except Exception:
            return 1e10
        
        # Check for numerical issues
        if np.any(~np.isfinite(roots)):
            return 1e10
        
        # Compute log-discriminant
        ld = log_discriminant_from_roots(roots)
        
        if not np.isfinite(ld):
            return 1e10
        
        # Soft constraint: penalize max |root| > 1
        max_mod = np.max(np.abs(roots))
        penalty = 0.0
        if max_mod > 1.0:
            penalty = 100.0 * (max_mod - 1.0)**2
        
        # We want to maximize log-discriminant, so minimize negative
        return -ld + penalty

    def get_points_from_coeffs(coeffs):
        """Convert polynomial coefficients to point coordinates."""
        c = coeffs[0::2] + 1j * coeffs[1::2]
        poly_coeffs = np.concatenate([np.array([1.0+0j]), c[::-1]])
        try:
            roots = np.roots(poly_coeffs)
        except Exception:
            return None
        if np.any(~np.isfinite(roots)):
            return None
        # Convert to real 2D points
        pts = np.column_stack([roots.real, roots.imag])
        return pts

    def regular_octagon_coeffs():
        """Get polynomial coefficients for regular octagon (roots at e^{i*2πk/8})."""
        angles = 2 * np.pi * np.arange(n) / n
        roots = np.exp(1j * angles)
        # Compute polynomial coefficients from roots
        c = np.poly(roots)  # This gives [1, c7, c6, ..., c0]
        c = c[1:]  # Remove leading 1
        # c is now [c7, c6, ..., c0] as complex
        # We need [c0, c1, ..., c7] in our format
        c_reversed = c[::-1]
        flat = np.zeros(16)
        flat[0::2] = c_reversed.real
        flat[1::2] = c_reversed.imag
        return flat

    best_score = -np.inf
    best_points = None

    # Start with regular octagon
    oct_angles = 2 * np.pi * np.arange(n) / n
    oct_points = np.column_stack([np.cos(oct_angles), np.sin(oct_angles)])
    best_score = compute_score(oct_points)
    best_points = oct_points.copy()

    # Phase 1: Global search in polynomial coefficient space
    # Initialize from regular octagon with some perturbation
    oct_coeffs = regular_octagon_coeffs()
    
    # Try multiple random restarts in coefficient space
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = oct_coeffs.copy()
        else:
            # Perturb the regular octagon coefficients
            noise_scale = 0.1 * restart
            x0 = oct_coeffs + noise_scale * np.random.randn(16)
        
        # Use Nelder-Mead for local optimization
        try:
            res = minimize(
                poly_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if res.fun < 1e9:
                pts_candidate = get_points_from_coeffs(res.x)
                if pts_candidate is not None:
                    # Check distinctness
                    dists_mat = np.sqrt(np.sum((pts_candidate[:, np.newaxis, :] - pts_candidate[np.newaxis, :, :])**2, axis=2))
                    np.fill_diagonal(dists_mat, np.inf)
                    if np.min(dists_mat) > 1e-8:
                        s_candidate = compute_score(pts_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_points = pts_candidate.copy()
        except Exception:
            pass

    # Phase 2: Also try the two-concentric-squares family for comparison
    def objective_symmetric(r2, rot_offset):
        """
        Evaluate score for two concentric squares.
        Square 1: radius 1, angles 0, pi/2, pi, 3pi/2
        Square 2: radius r2, angles pi/4 + rot_offset, etc.
        """
        a1 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        
        a2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + rot_offset
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        
        pts = np.vstack([p1, p2])
        return -compute_score(pts)

    # Grid search in symmetric family
    r2_vals = np.linspace(0.2, 1.0, 40)
    rot_vals = np.linspace(0.0, np.pi/4, 8)
    
    best_sym_r2 = 1.0
    best_sym_rot = 0.0
    best_sym_score = -np.inf
    
    for r2 in r2_vals:
        if time.time() > deadline - 15:
            break
        for rot in rot_vals:
            pts1 = np.array([
                [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
            ])
            angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + rot
            pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
            pts = np.vstack([pts1, pts2])
            
            s = compute_score(pts)
            if s > best_sym_score:
                best_sym_score = s
                best_sym_r2 = r2
                best_sym_rot = rot

    if best_sym_score > best_score:
        best_score = best_sym_score
        pts1 = np.array([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
        ])
        angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + best_sym_rot
        pts2 = best_sym_r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        best_points = np.vstack([pts1, pts2]).copy()

    # Phase 3: Local optimization in full 16D space on best_points
    if best_points is not None and time.time() < deadline - 10:
        def full_objective(flat_pts):
            pts = flat_pts.reshape(8, 2)
            return -compute_score(pts)
        
        try:
            res_full = minimize(
                full_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if res_full.fun < np.inf:
                pts_candidate = res_full.x.reshape(8, 2)
                dists_mat = np.sqrt(np.sum((pts_candidate[:, np.newaxis, :] - pts_candidate[np.newaxis, :, :])**2, axis=2))
                np.fill_diagonal(dists_mat, np.inf)
                if np.min(dists_mat) > 1e-8:
                    s_candidate = compute_score(pts_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_points = pts_candidate.copy()
        except Exception:
            pass

    # Phase 4: High-precision polish
    if best_points is not None and time.time() < deadline - 5:
        def full_objective(flat_pts):
            pts = flat_pts.reshape(8, 2)
            return -compute_score(pts)
        
        try:
            res_polish = minimize(
                full_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 5000, 'ftol': 1e-16, 'xtol': 1e-14}
            )
            
            if res_polish.fun < np.inf:
                pts_candidate = res_polish.x.reshape(8, 2)
                dists_mat = np.sqrt(np.sum((pts_candidate[:, np.newaxis, :] - pts_candidate[np.newaxis, :, :])**2, axis=2))
                np.fill_diagonal(dists_mat, np.inf)
                if np.min(dists_mat) > 1e-8:
                    s_candidate = compute_score(pts_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_points = pts_candidate.copy()
        except Exception:
            pass

    # Fallback to regular octagon if something went wrong
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    # Final check: distinct points
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return [[float(p[0]), float(p[1])] for p in best_points]
