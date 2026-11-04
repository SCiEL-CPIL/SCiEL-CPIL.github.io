"""
Optimized sphere covering with 130 points via Smooth Surrogate & L-BFGS-B Optimization.

Strategy:
1. Force-directed initialization to get a good starting configuration.
2. Use a smooth log-sum-exp surrogate for the covering radius (max of facet circumradii).
3. Compute analytical gradients of this surrogate w.r.t. point positions.
4. Use L-BFGS-B with sphere constraint (project back to unit sphere after each step).
5. Final high-precision polish on the worst facets.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    """Alias for _covering_radius."""
    return _covering_radius(pts)

def _facet_data(pts):
    """Compute facet circumradii and circumcenters for all hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    n_facets = len(hull.simplices)
    circumradii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    valid = np.ones(n_facets, dtype=bool)
    
    for idx, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            valid[idx] = False
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        circumradii[idx] = r
        circumcenters[idx] = x
    
    return circumradii, circumcenters, valid

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spectral_force_step(pts, lr=0.05, beta=2.0):
    """
    Vectorized force-directed step using Coulomb repulsion on the sphere.
    Maximizes minimum pairwise distance via inverse-cube repulsion.
    """
    pts = _normalize(pts.copy())
    
    # Compute 3D distance vectors
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0)
    dist3 = np.sqrt(dist3_sq)
    
    # Coulomb force: F_i = sum_j (pts[i] - pts[j]) / dist3^3
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Small centering correction to prevent drift to one hemisphere
    centroid = np.mean(pts, axis=0)
    F_spread = -0.05 * centroid
    
    F = F_coulomb + F_spread
    
    # Normalize force magnitude for stable steps
    F_norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F = F / F_norms
    
    new_pts = pts + lr * F
    new_pts = _normalize(new_pts)
    
    return new_pts

def _smooth_objective_and_grad(x, temp=0.01):
    """
    Smooth log-sum-exp surrogate for covering radius with analytical gradient.
    
    x: shape (N, 3) flattened to (3N,)
    Returns: (objective_value, gradient)
    """
    N = len(x) // 3
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    circumradii, circumcenters, valid = _facet_data(pts)
    
    if circumradii is None or not np.any(valid):
        return 1.0, np.zeros_like(x)
    
    r_valid = circumradii[valid]
    cc_valid = circumcenters[valid]
    hull = ConvexHull(pts)
    simplices_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 1.0, np.zeros_like(x)
    
    # Log-sum-exp for numerical stability
    r_max = np.max(r_valid)
    exp_vals = np.exp((r_valid - r_max) / temp)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Objective: T * log(sum(exp(r_i/T))) = r_max + T * log(sum(exp((r_i-r_max)/T)))
    obj = r_max + temp * np.log(sum_exp)
    
    # Gradient: For each point i, grad_i = sum over facets f containing i of 
    #   w_f * d(r_f)/d(x_i)
    # 
    # r_f = arccos(dot(x_f, a)) where x_f is circumcenter, a is one vertex
    # d(arccos(u))/du = -1/sqrt(1-u^2)
    # d(dot(x_f, a))/d(a) = x_f
    # But x_f also depends on the three vertices...
    # 
    # Simplification: For a triangle with vertices a,b,c on unit sphere,
    # the circumcenter direction is proportional to (b-a) x (c-a).
    # The circumradius r = arccos(n . a) where n is the unit normal.
    # 
    # For small perturbations, we approximate the gradient as:
    # Each vertex contributes w_f * (n_f - (n_f . v_i) * v_i) / sqrt(1 - (n_f . v_i)^2)
    # where v_i is the vertex position and n_f is the facet normal (circumcenter direction)
    
    grad = np.zeros((N, 3))
    
    for idx_f, (w_f, cc_f, simplex) in enumerate(zip(weights, cc_valid, simplices_valid)):
        if w_f < 1e-10:
            continue
        for i in simplex:
            v_i = pts[i]
            dot_val = np.dot(cc_f, v_i)
            dot_val_clipped = np.clip(dot_val, -0.999999, 0.999999)
            denom = np.sqrt(1.0 - dot_val_clipped ** 2)
            if denom < 1e-12:
                continue
            # Gradient of arccos(dot) w.r.t. v_i: -cc_f / sqrt(1-dot^2)
            # But we want to MINIMIZE r, so we move in direction that decreases r
            # dr/dv_i = -cc_f / sqrt(1-(cc_f.v_i)^2)
            # So gradient of objective w.r.t. v_i includes w_f * dr/dv_i
            # To minimize, we go in direction -grad = w_f * cc_f / sqrt(1-dot^2)
            # But we also need to account for the constraint that v_i stays on sphere
            # Project gradient onto tangent plane: g_tan = g - (g.v_i)*v_i
            g = -w_f * cc_f / denom
            g_tan = g - np.dot(g, v_i) * v_i
            grad[i] += g_tan
    
    return obj, grad.flatten()

def _local_refine(pts, max_iters=50, temp_init=0.02, temp_decay=0.95):
    """
    Fine-tune using L-BFGS-B with smooth log-sum-exp surrogate.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    
    temp = temp_init
    x0 = pts.flatten()
    
    for it in range(max_iters):
        if not _time_ok(5.0):
            break
        
        # Define objective and gradient for this temperature
        def obj_and_grad(x):
            val, grad = _smooth_objective_and_grad(x, temp)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 30, 'gtol': 1e-6, 'ftol': 1e-10}
            )
            x_new = result.x.reshape(-1, 3)
            x_new = _normalize(x_new)
        except Exception:
            break
        
        cr_new = _covering_radius_fast(x_new)
        
        if cr_new < best_r:
            best_r = cr_new
            best_pts = x_new.copy()
            pts = x_new.copy()
            x0 = x_new.flatten()
        else:
            # If L-BFGS-B didn't improve, try a smaller step
            x_trial = 0.5 * x0 + 0.5 * result.x
            x_trial = _normalize(x_trial.reshape(-1, 3))
            cr_trial = _covering_radius_fast(x_trial)
            if cr_trial < best_r:
                best_r = cr_trial
                best_pts = x_trial.copy()
                pts = x_trial.copy()
                x0 = x_trial.flatten()
            else:
                # Decay temperature and continue
                temp *= temp_decay
                if temp < 0.001:
                    break
        
        # Decay temperature each iteration for convergence
        temp *= temp_decay
        if temp < 0.001:
            break
    
    return best_pts, best_r

def construct():
    """
    1. Generate diverse initial configs.
    2. Run fast spectral force-directed optimization (no hull).
    3. Refine the best candidate with slow hull-based local search.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius (fast check)
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Phase 1: Fast Spectral Optimization with adaptive stopping
    for r0, p0 in init_scores:
        if not _time_ok(25.0):
            break
            
        pts = p0.copy()
        prev_cr = _covering_radius_fast(pts)
        # Adaptive step count: stop when improvement plateaus
        for it in range(800):
            if not _time_ok(20.0):
                break
            pts = _spectral_force_step(pts, lr=0.006, beta=2.0)
            
            # Check every 40 iterations
            if (it + 1) % 40 == 0:
                cr_check = _covering_radius_fast(pts)
                if abs(cr_check - prev_cr) < 1e-6:
                    break
                prev_cr = cr_check
        
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    # Phase 2: L-BFGS-B Refinement with smooth surrogate
    if best_pts is not None and _time_ok(15.0):
        best_pts, best_r = _local_refine(best_pts, max_iters=50, temp_init=0.02, temp_decay=0.95)
        
    # Phase 3: High-precision polish on worst facets
    if best_pts is not None and _time_ok(5.0):
        polish_lr = 0.001
        best_polish_pts = best_pts.copy()
        best_polish_r = best_r
        stall = 0
        for polish_iter in range(60):
            if not _time_ok(2.0):
                break
            
            try:
                hull = ConvexHull(best_polish_pts)
            except Exception:
                break
                
            # Find top 5 largest facets
            facet_radii = []
            for idx, simplex in enumerate(hull.simplices):
                a, b, c = best_polish_pts[simplex[0]], best_polish_pts[simplex[1]], best_polish_pts[simplex[2]]
                v1, v2 = b - a, c - a
                cross = np.cross(v1, v2)
                cnorm = np.linalg.norm(cross)
                if cnorm < 1e-12:
                    continue
                x = cross / cnorm
                centroid = (a + b + c) / 3.0
                if np.dot(x, centroid) < 0:
                    x = -x
                dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
                r = np.arccos(dot_val)
                facet_radii.append((r, idx, x))
            
            if not facet_radii:
                break
                
            facet_radii.sort(key=lambda x: -x[0])
            top_facets = facet_radii[:5]
            
            # Try move: move vertices of worst facets toward circumcenter
            trial_pts = best_polish_pts.copy()
            for r, idx, cc in top_facets:
                simplex = hull.simplices[idx]
                for i in simplex:
                    trial_pts[i] = trial_pts[i] + polish_lr * (cc - trial_pts[i])
            
            trial_pts = _normalize(trial_pts)
            cr = _covering_radius_fast(trial_pts)
            
            if cr < best_polish_r - 1e-10:
                best_polish_r = cr
                best_polish_pts = trial_pts
                best_pts = trial_pts.copy()
                best_r = cr
                stall = 0
            else:
                stall += 1
                polish_lr *= 0.8
                if stall > 8 or polish_lr < 0.0001:
                    break
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    # Final validation: ensure all points are on the unit sphere
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
