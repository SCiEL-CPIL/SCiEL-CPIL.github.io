import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

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
    Used for final polishing and standard local optimization.
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

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t subject to constraints.
    x = [points_flat (N*D), t]
    Returns t + penalty * max(0, soft_max_log - soft_min_log - t)^2
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    
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
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log(squared distances)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    # Constraint: soft_max_log - soft_min_log <= t
    # Violation is positive if constraint is broken
    violation = soft_max_log - soft_min_log - t
    
    # Penalty term
    penalty_weight = 100.0
    penalty = penalty_weight * np.maximum(0, violation) ** 2
    
    return t + penalty

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
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Icosahedron + 2 points (various placements)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for i in range(8):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())
        
    # 2. Fibonacci sphere with 14 points
    def fibonacci_sphere(n, rng):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts

    for i in range(6):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(4):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # 5. Two rings of 7 points each
    for i in range(3):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.zeros(7)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, 0.5)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    n_restarts = len(candidates)
    
    # --- Phase 1: Local Optimization with Log-Ratio (Fast) ---
    # Use a moderate tau for stability
    tau_local = 20.0
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 30:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_local,),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Sort candidates by raw score to prioritize best ones for expensive search
    scored_candidates = []
    for i in range(n_restarts):
        pts = candidates[i].reshape(N, D)
        s = raw_score(pts)
        scored_candidates.append((s, i))
    scored_candidates.sort()
    
    # --- Phase 2: Basin Hopping with Epigraph Objective ---
    # Basin hopping allows jumping between local minima.
    # We use the epigraph objective here as it handles the 't' variable well
    # and can be more robust for the random perturbations in basin hopping.
    # We limit this to the top 5 candidates to save time.
    
    top_indices = [idx for _, idx in scored_candidates[:5]]
    
    for idx in top_indices:
        if time.time() - start_time > timeout - 10:
            break
            
        x0_points = candidates[idx]
        # Initial t estimate: log ratio of current points
        pts_init = x0_points.reshape(N, D)
        sq_dists = get_pairwise_sq_dists(pts_init)
        min_d = np.min(sq_dists)
        max_d = np.max(sq_dists)
        if min_d > 1e-12:
            t_init = np.log(max_d / min_d) + 0.1 # Add small margin
        else:
            t_init = 5.0
            
        x0_bh = np.concatenate([x0_points, [t_init]])
        
        try:
            # Define the local minimizer for basin hopping
            # We use L-BFGS-B with the epigraph objective
            res_bh = basinhopping(
                objective_epigraph,
                x0_bh,
                minimizer_kwargs={'method': 'L-BFGS-B', 'args': (20.0,), 'options': {'maxiter': 20, 'ftol': 1e-6}},
                niter=10,
                stepsize=0.5,
                T=100.0,
                accept_test=None,
                seed=42
            )
            
            if res_bh.x is not None:
                # Extract points from result
                pts_bh = res_bh.x[:N*D].reshape(N, D)
                update_best(pts_bh)
                
                # Update the candidate list with the improved points for potential further refinement
                candidates[idx] = res_bh.x[:N*D]
                
        except Exception:
            pass

    # --- Phase 3: High-Precision Polish ---
    # Take the best points found so far and refine with high tau log-ratio
    # to get the exact min/max ratio.
    
    if best_points is not None:
        x_polish = best_points.flatten()
        # Try a few different high tau values for final precision
        for tau_polish in [100.0, 500.0, 2000.0]:
            if time.time() - start_time > timeout - 2:
                break
            try:
                res = minimize(
                    objective_log_ratio, 
                    x_polish, 
                    method='L-BFGS-B', 
                    args=(tau_polish,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    x_polish = res.x # Use the result as starting point for next higher tau
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
