# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools

def _objective_neg_min_ratio(points):
    """Negative of (d_min/d_max)^2 for minimization."""
    pts = points.reshape(14, 3)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = pdist(pts)
    d_min = d.min()
    d_max = d.max()
    if d_max < 1e-15:
        return 1e10
    ratio = d_min / d_max
    return -(ratio ** 2)

def _soft_min_ratio(points, beta=50.0):
    """Smooth surrogate using log-sum-exp for min and max."""
    pts = points.reshape(14, 3)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = pdist(pts)
    # log-sum-exp for -min (which is max of -d)
    neg_d = -d
    m = neg_d.max()
    log_sum_exp_neg = m + np.log(np.exp(neg_d - m).sum())
    # For max: log-sum-exp of d
    m2 = d.max()
    log_sum_exp_pos = m2 + np.log(np.exp(d - m2).sum())
    # Approximate min ≈ -log_sum_exp_neg, max ≈ log_sum_exp_pos
    # But this isn't quite right. Let me use a different approach.
    # Use: softmin = -logsumexp(-d/beta)*beta, softmax = logsumexp(d/beta)*beta
    softmin = -beta * np.log(np.exp(-d / beta).sum())
    softmax = beta * np.log(np.exp(d / beta).sum())
    if softmax < 1e-15:
        return 1e10
    ratio = softmin / softmax
    return -(ratio ** 2)

def _optimize_from_init(pts0, max_iter=2000, tol=1e-10):
    """Local optimization from initial point configuration."""
    x0 = pts0.flatten()
    
    best_x = x0.copy()
    best_score = _objective_neg_min_ratio(best_x)
    
    # Try multiple beta values for the soft surrogate
    for beta in [100.0, 50.0, 20.0, 10.0, 5.0]:
        def obj(x):
            return _soft_min_ratio(x, beta=beta)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
            if res.x is not None and np.all(np.isfinite(res.x)):
                score = _objective_neg_min_ratio(res.x)
                if score < best_score:  # more negative is better
                    best_score = score
                    best_x = res.x.copy()
        except Exception:
            pass
        
        # Also try with the exact objective for polish
        try:
            res2 = minimize(_objective_neg_min_ratio, best_x, method='L-BFGS-B',
                           options={'maxiter': max_iter // 2, 'ftol': tol, 'gtol': 1e-9})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                score2 = _objective_neg_min_ratio(res2.x)
                if score2 < best_score:
                    best_score = score2
                    best_x = res2.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(14, 3)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    n = 14
    d = 3
    
    best_points = None
    best_score = -1.0
    
    # Generate candidate initial configurations
    
    # 1. Dodecahedron vertices (20 vertices, pick best 14)
    phi = (1 + np.sqrt(5)) / 2.0
    dodeca_verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                dodeca_verts.append([x, y, z])
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            dodeca_verts.append([0, a, b])
            dodeca_verts.append([a, b, 0])
            dodeca_verts.append([b, 0, a])
    dodeca_verts = np.array(dodeca_verts)
    # Normalize to unit sphere
    norms = np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    dodeca_verts = dodeca_verts / norms
    
    # Try all combinations of 14 from 20 dodecahedron vertices
    # C(20,14) = C(20,6) = 38760 - too many, so sample
    np.random.seed(42)
    indices = list(range(20))
    sampled_combos = set()
    for _ in range(5000):
        combo = tuple(sorted(np.random.choice(20, 14, replace=False)))
        if combo not in sampled_combos:
            sampled_combos.add(combo)
            pts = dodeca_verts[list(combo)]
            score = -_objective_neg_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
            if time.time() > deadline - 60:
                break
    
    # 2. Icosahedron vertices (12 vertices) + 2 more points
    phi_i = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [0, 1, phi_i], [0, -1, phi_i], [0, 1, -phi_i], [0, -1, -phi_i],
        [1, phi_i, 0], [-1, phi_i, 0], [1, -phi_i, 0], [-1, -phi_i, 0],
        [phi_i, 0, 1], [-phi_i, 0, 1], [phi_i, 0, -1], [-phi_i, 0, -1]
    ])
    norms_i = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms_i
    
    # Add 2 points on sphere to make 14
    # Try several random additions
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        extra = np.random.randn(2, 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([ico_verts, extra])
        score = -_objective_neg_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # 3. Random points on sphere
    for trial in range(500):
        if time.time() > deadline - 60:
            break
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        score = -_objective_neg_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # 4. Run local optimization from best candidates
    if best_points is not None:
        if time.time() < deadline - 30:
            opt_pts = _optimize_from_init(best_points)
            score = -_objective_neg_min_ratio(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts
    
    # Also optimize from a few other good initial configs
    if best_points is not None and time.time() < deadline - 30:
        # Try optimizing from the first dodecahedron combo
        combo0 = tuple(range(14))
        pts0 = dodeca_verts[list(combo0)]
        opt_pts0 = _optimize_from_init(pts0)
        score0 = -_objective_neg_min_ratio(opt_pts0)
        if score0 > best_score:
            best_score = score0
            best_points = opt_pts0
    
    # Final polish: use exact objective with tighter tolerance
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(_objective_neg_min_ratio, best_points.flatten(), 
                               method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                final_pts = res_final.x.reshape(14, 3)
                score_final = -_objective_neg_min_ratio(final_pts)
                if score_final > best_score:
                    best_points = final_pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = np.random.randn(14, 3)
    
    return best_points

# EVOLVE-BLOCK-END
