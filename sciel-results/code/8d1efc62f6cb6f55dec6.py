import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
RING_SIZES = [1, 6, 12, 18, 13]  # Total = 50

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Minimum distance between centers
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def ring_params_to_points(params):
    """
    Convert ring parameters to point positions.
    params: [r1, offset1, r2, offset2, r3, offset3, r4, offset4]
    where r_i is the radius of ring i (i=1..4, ring 0 is center at origin)
    and offset_i is the angular offset of ring i.
    """
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]  # Center
    idx = 1
    for i, size in enumerate(RING_SIZES[1:], start=1):
        r = params[2*(i-1)]
        offset = params[2*(i-1)+1]
        for k in range(size):
            angle = offset + 2 * np.pi * k / size
            points[idx] = [r * np.cos(angle), r * np.sin(angle)]
            idx += 1
    return points

def ring_energy(params, beta=50.0):
    """
    Smooth energy for ring parameters. Minimizing this maximizes the packing radius.
    """
    points = ring_params_to_points(params)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    b = 1.0 - norms
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    E_pair = np.exp(-beta * d / 2.0)
    np.fill_diagonal(E_pair, 0.0)
    E_bound = np.exp(-beta * b)
    
    S = np.sum(E_pair) + np.sum(E_bound)
    return np.log(S) / beta

def ring_energy_grad(params, beta=50.0):
    """
    Numerical gradient of ring energy (8-dimensional, so numerical is fine).
    """
    val = ring_energy(params, beta)
    grad = np.zeros_like(params)
    eps = 1e-7
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (ring_energy(params_plus, beta) - val) / eps
    return val, grad

def _local_refine(points, deadline, max_iter=300):
    """
    Local refinement using L-BFGS-B on the full 100D space with smooth objective.
    """
    def obj_and_grad(x, beta=100.0):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        E_pair = np.exp(-beta * d / 2.0)
        np.fill_diagonal(E_pair, 0.0)
        E_bound = np.exp(-beta * b)
        S = np.sum(E_pair) + np.sum(E_bound)
        val = np.log(S) / beta
        
        # Gradient
        grad = np.zeros_like(x)
        inv_norms = np.zeros_like(norms)
        mask = norms > 1e-8
        inv_norms[mask] = 1.0 / norms[mask]
        grad_bound = (E_bound * beta)[:, np.newaxis] * pts * inv_norms[:, np.newaxis]
        
        d_safe = d.copy()
        np.fill_diagonal(d_safe, 1.0)
        factor = (E_pair * (-beta / 2.0)) / d_safe
        np.fill_diagonal(factor, 0.0)
        row_sums = np.sum(factor, axis=1)
        weighted_points = factor @ pts
        grad_pair = pts * row_sums[:, np.newaxis] - weighted_points
        
        grad_flat = (grad_bound + grad_pair).flatten()
        return val, grad_flat
    
    x0 = points.flatten()
    # Project inside circle
    pts_check = x0.reshape(N, 2)
    norms = np.linalg.norm(pts_check, axis=1)
    out_mask = norms > 1.0
    if np.any(out_mask):
        pts_check[out_mask] = pts_check[out_mask] / norms[out_mask, np.newaxis]
        x0 = pts_check.flatten()
    
    try:
        res = minimize(
            obj_and_grad, x0, args=(100.0,),
            jac=True, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_pts = res.x.reshape(N, 2)
        if np.all(np.isfinite(new_pts)):
            return new_pts
    except Exception:
        pass
    return points

def construct():
    deadline = time.time() + 145.0
    best_r = 0.0
    best_points = None
    
    # Phase 1: Global search over ring parameters using differential evolution
    # 8 parameters: [r1, off1, r2, off2, r3, off3, r4, off4]
    bounds = [
        (0.1, 0.4),    # r1 (ring of 6)
        (0.0, 2*np.pi),  # offset1
        (0.3, 0.65),   # r2 (ring of 12)
        (0.0, 2*np.pi),  # offset2
        (0.5, 0.85),   # r3 (ring of 18)
        (0.0, 2*np.pi),  # offset3
        (0.7, 0.98),   # r4 (ring of 13)
        (0.0, 2*np.pi),  # offset4
    ]
    
    def de_objective(params):
        if time.time() > deadline - 30:
            return 1e10
        return ring_energy(params, beta=50.0)
    
    try:
        de_result = differential_evolution(
            de_objective, bounds,
            maxiter=200, popsize=30, tol=1e-10,
            mutation=(0.3, 1.0), recombination=0.8,
            seed=42, polish=False,
            updating='deferred'
        )
        best_params = de_result.x
    except Exception:
        # Fallback to reasonable defaults
        best_params = np.array([0.25, 0.0, 0.5, 0.0, 0.75, 0.0, 0.95, 0.0])
    
    # Convert to points
    pts = ring_params_to_points(best_params)
    current_r = compute_radius(pts)
    if current_r > best_r:
        best_r = current_r
        best_points = pts.copy()
    
    # Phase 2: Local refinement in full 100D space
    if time.time() < deadline - 15 and best_points is not None:
        refined_pts = _local_refine(best_points, deadline, max_iter=300)
        refined_r = compute_radius(refined_pts)
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_pts.copy()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5 and best_points is not None:
        def final_obj_and_grad(x, beta=500.0):
            pts = x.reshape(N, 2)
            norms = np.linalg.norm(pts, axis=1)
            b = 1.0 - norms
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(d, np.inf)
            
            E_pair = np.exp(-beta * d / 2.0)
            np.fill_diagonal(E_pair, 0.0)
            E_bound = np.exp(-beta * b)
            S = np.sum(E_pair) + np.sum(E_bound)
            val = np.log(S) / beta
            
            grad = np.zeros_like(x)
            inv_norms = np.zeros_like(norms)
            mask = norms > 1e-8
            inv_norms[mask] = 1.0 / norms[mask]
            grad_bound = (E_bound * beta)[:, np.newaxis] * pts * inv_norms[:, np.newaxis]
            
            d_safe = d.copy()
            np.fill_diagonal(d_safe, 1.0)
            factor = (E_pair * (-beta / 2.0)) / d_safe
            np.fill_diagonal(factor, 0.0)
            row_sums = np.sum(factor, axis=1)
            weighted_points = factor @ pts
            grad_pair = pts * row_sums[:, np.newaxis] - weighted_points
            
            grad_flat = (grad_bound + grad_pair).flatten()
            return val, grad_flat
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                final_obj_and_grad, x0, args=(500.0,),
                jac=True, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                new_r = compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback
        best_params = np.array([0.25, 0.0, 0.5, 0.0, 0.75, 0.0, 0.95, 0.0])
        best_points = ring_params_to_points(best_params)
    
    # Final validity check
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        best_points = best_points / norms[:, np.newaxis]
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
