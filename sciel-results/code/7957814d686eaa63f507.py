import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial.transform import Rotation

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

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale: make the mean squared distance = 1
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of log(squared distances)
    log_sq_dists = np.log(sq_dists)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Objective is the difference (log ratio)
    obj = soft_max_log - soft_min_log
    
    return obj

def sqp_objective(x, penalty=1000.0):
    """
    SQP-style reformulation: maximize min squared distance subject to max <= 1.
    We normalize so that max squared distance is approximately 1, then
    maximize the minimum squared distance. The penalty term softly enforces
    the upper bound constraint d_i^2 <= 1.
    
    This directly targets the ratio: if max_d2 ≈ 1, then ratio = 1/min_d2,
    so maximizing min_d2 minimizes the ratio.
    """
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    max_sq = np.max(sq_dists)
    if max_sq < 1e-12:
        return 1e10
    
    # Normalize so max squared distance = 1
    points = points / np.sqrt(max_sq)
    sq_dists = get_pairwise_sq_dists(points)
    
    min_sq = np.min(sq_dists)
    
    # Penalize any squared distance exceeding 1 (shouldn't happen after normalization,
    # but numerical errors can cause slight violations)
    violations = np.maximum(sq_dists - 1.0, 0.0)
    penalty_term = penalty * np.sum(violations**2)
    
    # We want to MAXIMIZE min_sq, so minimize -min_sq
    # Use soft-min for smoothness
    tau_inner = 50.0
    log_sq = np.log(np.maximum(sq_dists, 1e-15))
    m_min = np.min(log_sq)
    exp_min = np.exp(-tau_inner * (log_sq - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min)) / tau_inner
    
    obj = -soft_min_log + penalty_term
    return obj

def objective_epigraph(x, tau=100.0):
    """
    Epigraph-style smooth objective: penalize deviations from a target log-distance.
    We minimize the sum of squared deviations of log(d_i^2) from their mean,
    which encourages all distances to be similar. Combined with a large-penalty
    term for the range, this can escape local minima of the log-ratio objective.
    """
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    points = points / np.sqrt(mean_sq_dist)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    # Spread of log distances: use soft range (max - min) with log-sum-exp
    m_max = np.max(log_sq_dists)
    m_min = np.min(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Also penalize variance of log distances for extra smoothing
    log_mean = np.mean(log_sq_dists)
    variance = np.mean((log_sq_dists - log_mean) ** 2)
    
    obj = (soft_max - soft_min) + 0.1 * variance
    return obj

def sqp_objective_grad(x, penalty=1000.0):
    """
    Compute objective and gradient for SQP-style formulation.
    """
    points = x.reshape(N, D)
    points_centered = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points_centered)
    max_sq = np.max(sq_dists)
    if max_sq < 1e-12:
        return 1e10, np.zeros_like(x)
    
    scale = np.sqrt(max_sq)
    points_norm = points_centered / scale
    sq_dists_norm = get_pairwise_sq_dists(points_norm)
    
    # Soft-min of log squared distances
    tau_inner = 50.0
    sq_clipped = np.maximum(sq_dists_norm, 1e-15)
    log_sq = np.log(sq_clipped)
    m_min = np.min(log_sq)
    exp_min = np.exp(-tau_inner * (log_sq - m_min))
    sum_exp = np.sum(exp_min)
    soft_min_log = m_min - np.log(sum_exp) / tau_inner
    
    # Penalty for violations
    violations = np.maximum(sq_dists_norm - 1.0, 0.0)
    penalty_term = penalty * np.sum(violations**2)
    
    obj = -soft_min_log + penalty_term
    
    # Gradient computation
    # d(soft_min_log)/d(points) via chain rule
    # Let w_i = exp(-tau * (log_sq_i - m_min)) / sum_exp
    # d(soft_min_log)/d(log_sq_i) = -w_i
    # d(log_sq_i)/d(points) = 2*(p_i - p_j) / sq_dist_i (for pair i,j)
    
    weights = exp_min / sum_exp  # shape (_n_pairs,)
    # d(soft_min_log)/d(sq_dists) = -weights / sq_dists
    dsoftmin_dsq = -weights / sq_clipped  # shape (_n_pairs,)
    
    # Gradient of sq_dists w.r.t. points
    # For pair (i,j), d(sq_dists_ij)/d(p_i) = 2*(p_i - p_j), d(sq_dists_ij)/d(p_j) = 2*(p_j - p_i)
    grad = np.zeros((N, D))
    for k in range(_n_pairs):
        i, j = _pair_i[k], _pair_j[k]
        coeff = 2.0 * dsoftmin_dsq[k] / (scale * scale)
        diff = points_centered[i] - points_centered[j]
        grad[i] += coeff * diff
        grad[j] -= coeff * diff
    
    # Penalty gradient
    for k in range(_n_pairs):
        if sq_dists_norm[k] > 1.0:
            i, j = _pair_i[k], _pair_j[k]
            coeff = 4.0 * penalty * (sq_dists_norm[k] - 1.0) / (scale * scale)
            diff = points_centered[i] - points_centered[j]
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    
    # Negate because we're minimizing -soft_min_log
    grad = -grad
    
    return obj, grad

def _make_icosahedron():
    """Generate 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return ico / np.linalg.norm(ico, axis=1, keepdims=True)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidate: Icosahedron + 2 points
    ico = _make_icosahedron()
    
    # Add 2 points at various symmetric positions (randomly rotated)
    p1 = rng.normal(size=D)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=D)
    p2 = p2 / np.linalg.norm(p2)
    init_pts = np.vstack([ico, p1, p2])
    rot = Rotation.from_quat(rng.normal(size=4))
    init_pts = rot.apply(init_pts)
    x0 = init_pts.flatten()
    
    # Phase 1: Global Exploration using Basin Hopping
    # Use a moderate tau for the local optimizer inside Basin Hopping
    tau_bh = 20.0
    
    def bh_objective(x):
        return objective_log_ratio(x, tau=tau_bh)
    
    bh_minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
    }
    
    bh_best_x = x0.copy()
    bh_best_f = bh_objective(x0)
    
    def bh_callback(x, f, accept):
        nonlocal bh_best_x, bh_best_f
        if f < bh_best_f:
            bh_best_f = f
            bh_best_x = x.copy()
        # Time check
        if time.time() - start_time > timeout - 60:
            raise Exception("Time to stop Basin Hopping")
            
    try:
        bh_res = basinhopping(
            bh_objective,
            x0,
            minimizer_kwargs=bh_minimizer_kwargs,
            stepsize=0.5,
            niter=100, # Will be cut by time
            callback=bh_callback,
            seed=42
        )
        if bh_res.x is not None:
            update_best(bh_res.x.reshape(N, D))
            bh_best_x = bh_res.x
    except Exception as e:
        if "Time to stop" not in str(e):
            pass # Ignore other exceptions, use best found so far
    
    # Ensure we have a best point
    if best_points is None:
        best_points = bh_best_x.reshape(N, D)
        best_score = raw_score(best_points)
    
    # Phase 2: Local Refinement with Multi-Stage Tau using both objectives
    # Take the best point found and polish it
    x_current = best_points.flatten()
    
    tau_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    
    for tau in tau_schedule:
        if time.time() - start_time > timeout - 20:
            break
        try:
            res = minimize(
                objective_log_ratio, 
                x_current, 
                method='L-BFGS-B', 
                args=(tau,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                x_current = res.x
        except Exception:
            pass
    
    # Phase 2b: Try epigraph objective from current best
    if time.time() - start_time < timeout - 15:
        for tau_e in [50.0, 200.0, 500.0]:
            if time.time() - start_time > timeout - 10:
                break
            try:
                res_e = minimize(
                    objective_epigraph,
                    x_current,
                    method='L-BFGS-B',
                    args=(tau_e,),
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_e.x is not None:
                    pts_e = res_e.x.reshape(N, D)
                    update_best(pts_e)
                    # If epigraph improved things, continue refining with log_ratio
                    if best_points is not None and raw_score(pts_e) < best_score + 1e-9:
                        x_current = res_e.x
            except Exception:
                pass
    
    # Phase 2c: SQP-style reformulation - maximize min dist subject to max <= 1
    if time.time() - start_time < timeout - 10:
        for penalty_val in [100.0, 500.0, 2000.0, 10000.0]:
            if time.time() - start_time > timeout - 5:
                break
            try:
                res_sq = minimize(
                    sqp_objective,
                    x_current,
                    method='L-BFGS-B',
                    args=(penalty_val,),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res_sq.x is not None:
                    pts_sq = res_sq.x.reshape(N, D)
                    update_best(pts_sq)
                    if best_points is not None and raw_score(pts_sq) < best_score + 1e-9:
                        x_current = res_sq.x
            except Exception:
                pass
    
    # Phase 3: Final Exact Local Search using SLSQP on true objective
    if time.time() - start_time < timeout - 5:
        def exact_objective(x):
            pts = x.reshape(N, D)
            sq_dists = get_pairwise_sq_dists(pts)
            min_sq = np.min(sq_dists)
            if min_sq < 1e-12:
                return 1e10
            return np.max(sq_dists) / min_sq
        
        try:
            res = minimize(
                exact_objective, 
                x_current, 
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
        
        # Also try SLSQP with SQP formulation
        if time.time() - start_time < timeout - 3:
            try:
                res_sq2 = minimize(
                    sqp_objective,
                    x_current,
                    method='SLSQP',
                    args=(10000.0,),
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res_sq2.x is not None:
                    pts_sq2 = res_sq2.x.reshape(N, D)
                    update_best(pts_sq2)
            except Exception:
                pass
    
    # Phase 4: Structured random restarts if time remains
    # Use icosahedron-based initializations with random perturbations
    restart_count = 0
    while time.time() - start_time < timeout - 5:
        # Build a structured initial guess: icosahedron + 2 perturbed points
        ico_r = _make_icosahedron()
        # Random rotation
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        rot_r = Rotation.from_quat(quat)
        ico_r = rot_r.apply(ico_r)
        
        # Add 2 points: one near the "north pole" of rotated frame, one random
        p1_r = rot_r.apply(np.array([0.0, 0.0, 1.0]))
        p1_r += rng.normal(scale=0.3, size=D)
        p1_r /= np.linalg.norm(p1_r)
        
        p2_r = rng.normal(size=D)
        p2_r /= np.linalg.norm(p2_r)
        
        x0_struct = np.vstack([ico_r, p1_r, p2_r]).flatten()
        
        # Quick local opt with log-ratio
        try:
            res = minimize(
                objective_log_ratio, 
                x0_struct, 
                method='L-BFGS-B', 
                args=(50.0,),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                # If this is good, do a high-precision polish
                if raw_score(pts) < best_score * 1.05 and time.time() - start_time < timeout - 3:
                    try:
                        res2 = minimize(
                            objective_log_ratio, 
                            res.x, 
                            method='L-BFGS-B', 
                            args=(1000.0,),
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass
        restart_count += 1
    
    # Phase 5: Final high-precision polish on the absolute best
    if time.time() - start_time < timeout - 2 and best_points is not None:
        x_final = best_points.flatten()
        for tau_final in [500.0, 2000.0, 5000.0]:
            if time.time() - start_time > timeout - 1:
                break
            try:
                res_f = minimize(
                    objective_log_ratio,
                    x_final,
                    method='L-BFGS-B',
                    args=(tau_final,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_f.x is not None:
                    pts_f = res_f.x.reshape(N, D)
                    update_best(pts_f)
                    x_final = res_f.x
            except Exception:
                pass
        
        # SQP polish
        if time.time() - start_time < timeout - 1:
            try:
                res_sqf = minimize(
                    sqp_objective,
                    x_final,
                    method='L-BFGS-B',
                    args=(50000.0,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_sqf.x is not None:
                    pts_sqf = res_sqf.x.reshape(N, D)
                    update_best(pts_sqf)
                    x_final = res_sqf.x
            except Exception:
                pass
        
        # One last SLSQP on exact objective
        if time.time() - start_time < timeout - 0.5:
            def exact_objective_final(x):
                pts = x.reshape(N, D)
                sq_dists = get_pairwise_sq_dists(pts)
                min_sq = np.min(sq_dists)
                if min_sq < 1e-12:
                    return 1e10
                return np.max(sq_dists) / min_sq
            
            try:
                res_s = minimize(
                    exact_objective_final,
                    x_final,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-15}
                )
                if res_s.x is not None:
                    pts_s = res_s.x.reshape(N, D)
                    update_best(pts_s)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = init_pts
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
