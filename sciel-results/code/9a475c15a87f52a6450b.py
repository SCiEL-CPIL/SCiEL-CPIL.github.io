import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 14
D = 3

# Precompute pairwise indices for efficiency
_pair_i, _pair_j = np.triu_indices(N, k=1)
_n_pairs = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_pair_i, _pair_j]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def _make_icosahedron():
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices
    verts = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            verts.append([0, sx, sy * phi])
    for sx in [1, -1]:
        for sy in [1, -1]:
            verts.append([sx, sy * phi, 0])
    for sx in [1, -1]:
        for sy in [1, -1]:
            verts.append([sx * phi, 0, sy])
    ico = np.array(verts, dtype=float)
    # Normalize to unit radius
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def objective_full_space(points_flat, tau=50.0):
    """
    Objective function for full 42-dimensional coordinate space.
    points_flat: array of shape (N*D,)
    """
    points = points_flat.reshape(N, D)
    
    # Center
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    # Normalize to unit variance (or similar) to keep numbers stable
    points = points / np.sqrt(mean_sq_dist)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def objective_epigraph(points_flat, tau=100.0):
    """
    Epigraph formulation: minimize t subject to constraints that all pairwise
    squared distances are <= 1 and >= 1/t.
    Here we use a soft penalty approach.
    params: points_flat of shape (N*D,)
    """
    points = points_flat.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    # Normalize so that max sq dist = 1
    max_sq = np.max(sq_dists)
    if max_sq < 1e-15:
        return 1e10
    
    points = points / np.sqrt(max_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Now max_sq should be ~1
    # We want to minimize (1 / min_sq) which is the ratio
    # Use soft-min to make it differentiable
    log_sq_dists = np.log(sq_dists)
    
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # The objective is -soft_min (we want to maximize min dist, so minimize -min dist)
    # But the raw score is (max/min)^2. Since max is normalized to 1, score = 1/min^2.
    # So minimizing 1/min is equivalent to maximizing min.
    # Let's just return 1/soft_min as a proxy for the ratio? No, that's not smooth.
    # Let's return -soft_min to maximize the minimum distance.
    # Wait, the evaluator wants (max/min)^2.
    # If we normalize max=1, score = 1/min^2.
    # Minimizing 1/min^2 is equivalent to maximizing min.
    # So we can just use -soft_min as the objective for L-BFGS-B (minimization).
    return -soft_min

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        # Check for duplicates
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # 1. Start with an icosahedron + 2 points
    # The icosahedron has 12 vertices. We need 2 more.
    # Place them at the origin? No, distinct points.
    # Place them near the origin? Or maybe on the z-axis?
    # Let's try placing them at (0,0,0) is not allowed (distinct from vertices? Yes, but 0 is distinct).
    # But 0 is equidistant to all 12.
    # Let's try (0,0,1) and (0,0,-1) scaled?
    
    ico = _make_icosahedron()
    # Add two points on the z-axis
    # The vertices of icosahedron have z-coordinates: 0, +/-1, +/-phi (normalized)
    # Let's just add (0,0,r) and (0,0,-r)
    
    # Multi-start Nelder-Mead / L-BFGS-B
    # We will optimize all 14 points in full space.
    
    # Initial configuration: Icosahedron + 2 axial points
    # Scale the icosahedron to unit radius
    # The axial points can be at some radius.
    
    # Let's create a few initial configurations
    inits = []
    
    # Init 1: Icosahedron + points at origin (slightly perturbed to avoid singularity)
    p_origin1 = np.array([0.0, 0.0, 0.01])
    p_origin2 = np.array([0.0, 0.0, -0.01])
    pts1 = np.vstack([ico, [p_origin1, p_origin2]])
    inits.append(pts1)
    
    # Init 2: Icosahedron + points at z=1, z=-1 (on the sphere)
    p_z1 = np.array([0.0, 0.0, 1.0])
    p_z2 = np.array([0.0, 0.0, -1.0])
    pts2 = np.vstack([ico, [p_z1, p_z2]])
    inits.append(pts2)
    
    # Init 3: Random points on sphere
    for i in range(3):
        rand_pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(rand_pts, axis=1, keepdims=True)
        rand_pts = rand_pts / norms
        inits.append(rand_pts)
    
    # Optimize each initial configuration
    for init_pts in inits:
        if time.time() - start_time > timeout - 20:
            break
        
        update_best(init_pts)
        
        start_flat = init_pts.flatten()
        
        # Try L-BFGS-B with full space objective
        try:
            res = minimize(
                objective_full_space,
                start_flat,
                method='L-BFGS-B',
                args=(50.0,),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # Basin hopping from this result
                def bh_obj_full(p_flat):
                    return objective_full_space(p_flat, tau=20.0)
                
                bh_minimizer_kwargs = {
                    'method': 'L-BFGS-B',
                    'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                }
                
                if time.time() - start_time < timeout - 10:
                    bh_res = basinhopping(
                        bh_obj_full,
                        res.x,
                        minimizer_kwargs=bh_minimizer_kwargs,
                        stepsize=0.1,
                        niter=10,
                        seed=rng.integers(1000)
                    )
                    if bh_res.x is not None:
                        pts = bh_res.x.reshape(N, D)
                        update_best(pts)
        except Exception:
            pass
        
        # Try Epigraph formulation
        try:
            res_ep = minimize(
                objective_epigraph,
                start_flat,
                method='L-BFGS-B',
                args=(100.0,),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res_ep.x is not None:
                pts = res_ep.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    # 4. Final polish on the best found points
    if best_points is not None and time.time() - start_time < timeout - 2:
        start_flat = best_points.flatten()
        try:
            res = minimize(
                objective_full_space,
                start_flat,
                method='L-BFGS-B',
                args=(500.0,),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _make_icosahedron()
        # This is only 12 points, need 14.
        # Just return random if all fails
        best_points = rng.normal(size=(N, D))
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
