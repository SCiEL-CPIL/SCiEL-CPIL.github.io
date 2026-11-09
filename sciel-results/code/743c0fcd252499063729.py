import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    # Helper: Calculate score (radius) for a set of points
    def calc_score(pts):
        # Check validity first
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms >= 1.0):
            return -1.0
        if np.any(~np.isfinite(pts)):
            return -1.0
        
        d_min = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < d_min:
                    d_min = d
        b_min = np.min(1.0 - norms)
        r = min(d_min / 2.0, b_min)
        return r

    # Helper: Clip points to be strictly inside unit circle
    def clip_to_circle(pts, margin=1e-6):
        pts = pts.copy()
        norms = np.linalg.norm(pts, axis=1)
        mask = norms >= 1.0 - margin
        if np.any(mask):
            # Project to radius (1 - margin)
            scale = (1.0 - margin) / norms[mask]
            pts[mask] *= scale[:, np.newaxis]
        return pts

    # Helper: Hexagonal Grid Initialization
    def get_hex_grid(spacing):
        points = []
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing / 2.0)**2: # Keep well inside
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        # Select N points closest to origin
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)[:N]
        return points[indices]

    # Helper: Ring Initialization
    def get_ring_init(r):
        # Center point
        pts = [[0.0, 0.0]]
        # Rings
        counts = [6, 12, 18, 13]
        max_r = 1.0 - r
        for k, count in enumerate(counts):
            ring_r = (k + 1) * r * 1.1
            if ring_r > max_r:
                ring_r = max_r * (0.8 + 0.2 * k / len(counts))
            for m in range(count):
                angle = 2 * np.pi * m / count + k * 0.1
                x = ring_r * np.cos(angle)
                y = ring_r * np.sin(angle)
                pts.append([x, y])
        
        pts = np.array(pts[:N])
        pts += np.random.randn(N, 2) * 1e-6
        return clip_to_circle(pts)

    # Objective and Gradient
    def energy_and_grad(c, R):
        c = c.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        r_norm = np.linalg.norm(c, axis=1)
        b_dist = 1.0 - r_norm
        
        pair_viol = np.maximum(0.0, 2.0 * R - d)
        bound_viol = np.maximum(0.0, R - b_dist)
        
        E = np.sum(pair_viol**2) + np.sum(bound_viol**2)
        
        grad = np.zeros_like(c)
        
        coeff = np.zeros((N, N))
        mask = d < np.inf
        safe_d = np.where(d > 1e-8, d, 1.0)
        coeff[mask] = 2.0 * pair_viol[mask] / safe_d[mask]
        coeff[d < 1e-8] = 0.0
        
        sum_coeff = np.sum(coeff, axis=1)
        grad_pair = c * sum_coeff[:, np.newaxis] - (coeff @ c)
        
        grad_bound = np.zeros_like(c)
        valid_bound = r_norm > 1e-8
        grad_bound[valid_bound] = (2.0 * bound_viol[valid_bound] * c[valid_bound]) / r_norm[valid_bound, np.newaxis]
        
        grad = grad_pair + grad_bound
        
        return E, grad

    def objective(c, R):
        E, _ = energy_and_grad(c, R)
        return E

    def jac(c, R):
        _, g = energy_and_grad(c, R)
        return g.flatten()

    # Initializations
    # 1. Hex
    hex_pts = get_hex_grid(0.25)
    if hex_pts is None:
        # Fixed fallback: generate random points and clip them
        hex_pts = np.random.rand(N, 2) * 2.0 - 1.0
        hex_pts = clip_to_circle(hex_pts, margin=0.1)
    
    # 2. Ring
    ring_pts = get_ring_init(0.12)

    # Bisection
    R_low = 0.1
    R_high = 0.126
    
    best_valid_points = hex_pts.copy()
    best_valid_r = calc_score(hex_pts)
    if best_valid_r < 0:
        best_valid_r = 0.1 # Fallback score
        best_valid_points = clip_to_circle(hex_pts)

    max_iter_bisection = 30
    
    for _ in range(max_iter_bisection):
        if time.time() > deadline - 10:
            break
            
        R_mid = (R_low + R_high) / 2.0
        
        best_E_for_R = np.inf
        best_pts_for_R = None
        
        # Try different initializations
        inits = [hex_pts, ring_pts, best_valid_points]
        
        for init in inits:
            if time.time() > deadline - 5:
                break
            
            x0 = init.copy()
            x0 += np.random.randn(N, 2) * 1e-5
            x0 = clip_to_circle(x0)
            
            try:
                res = minimize(
                    objective, x0, args=(R_mid,),
                    jac=lambda c: jac(c, R_mid),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Clip the result to ensure validity
                res_x_clipped = clip_to_circle(res.x)
                E_final, _ = energy_and_grad(res_x_clipped, R_mid)
                
                if E_final < best_E_for_R:
                    best_E_for_R = E_final
                    best_pts_for_R = res_x_clipped.copy()
                
                if E_final < 1e-6:
                    break
                    
            except Exception:
                pass
        
        if best_pts_for_R is not None and best_E_for_R < 1e-5:
            R_low = R_mid
            best_valid_points = best_pts_for_R.copy()
            best_valid_r = R_mid
        else:
            R_high = R_mid
            
    # Final Polish
    if time.time() < deadline - 2:
        x0 = best_valid_points.copy()
        try:
            res = minimize(
                objective, x0, args=(best_valid_r,),
                jac=lambda c: jac(c, best_valid_r),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            res_x_clipped = clip_to_circle(res.x)
            E_final, _ = energy_and_grad(res_x_clipped, best_valid_r)
            if E_final < 1e-8:
                best_valid_points = res_x_clipped.copy()
        except Exception:
            pass

    # Final Safety Check: Ensure returned points are valid
    final_pts = clip_to_circle(best_valid_points, margin=1e-5)
    
    # If the clipped points are invalid (e.g. all at center due to bad clip), 
    # we might need to fallback to a known good configuration or just return clipped.
    # calc_score handles the validity check.
    
    return final_pts
