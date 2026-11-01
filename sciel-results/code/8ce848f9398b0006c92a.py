import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_log(x, tau=50.0):
    """
    Minimizes log(soft_max(sq_dists)) - log(soft_min(sq_dists)).
    This is equivalent to minimizing the log of the ratio of max to min distances.
    Using log-sum-exp for stability.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max approximation: (1/tau) * log(sum(exp(tau * d)))
    # We want to minimize the ratio, so we minimize log(max) - log(min)
    # log(max) approx: log_sum_exp(tau * d) / tau
    # log(min) approx: -log_sum_exp(-tau * d) / tau
    
    # For numerical stability in log-sum-exp:
    # log(sum(exp(tau * d))) = tau * max(d) + log(sum(exp(tau * (d - max(d)))))
    m_max = np.max(sq_dists)
    log_sum_exp_max = tau * m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max))))
    soft_log_max = log_sum_exp_max / tau
    
    m_min = np.min(sq_dists)
    # log(sum(exp(-tau * d))) = -tau * min(d) + log(sum(exp(-tau * (d - min(d)))))
    log_sum_exp_min = -tau * m_min + np.log(np.sum(np.exp(-tau * (sq_dists - m_min))))
    soft_log_min = -log_sum_exp_min / tau
    
    return soft_log_max - soft_log_min

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin for safety
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        # Ensure points are distinct and valid
        if pts is None or len(pts) != N:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedron (12 points) + 2 points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # The icosahedron has 12 vertices. We need 2 more.
    # The "poles" of the icosahedron (along the 5-fold axis) are not vertices.
    # A good guess for the 2 extra points might be near the center of the faces
    # or simply random points on the sphere, which DE/L-BFGS will refine.
    # Let's add 2 random points on the sphere for the initial guess.
    p1 = rng.normal(size=D)
    p1 = p1 / np.linalg.norm(p1)
    p2 = rng.normal(size=D)
    p2 = p2 / np.linalg.norm(p2)
    
    # Ensure they are distinct from ico vertices (highly likely with random)
    pts_ico = np.vstack([ico, p1.reshape(1,3), p2.reshape(1,3)])
    candidates.append(pts_ico.flatten())
    
    # 2. Random points on sphere
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Random points in a box (to allow for non-spherical configurations if needed)
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Optimization Phase 1: Polish initial candidates with L-BFGS-B
    # This is fast and might find a good local minimum immediately.
    for i, init_x in enumerate(candidates):
        if time.time() > deadline - 30: # Leave 30s for DE
            break
        try:
            res = minimize(
                objective_log, 
                init_x, 
                args=(50.0,), # tau
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                update_best(res.x.reshape(N, D))
        except Exception:
            pass

    # Optimization Phase 2: Differential Evolution
    # DE is better for global search. We run it for a few iterations.
    # Bounds: [-5, 5] is generous. The optimal configuration will likely be on a sphere of radius ~1-2.
    bounds = [(-5, 5)] * (N * D)
    
    # Run DE with different seeds to explore different basins
    seeds = [1, 2, 3]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        try:
            # maxiter=50, popsize=15 is a balance between speed and coverage
            # tol=1e-6 is sufficient for DE
            res_de = differential_evolution(
                objective_log, 
                bounds, 
                args=(50.0,),
                maxiter=50, 
                popsize=15, 
                seed=seed,
                tol=1e-6,
                polish=False, # We will polish manually to save time and control it
                update_interval=10, # Check time more frequently
                callback=lambda xk, convergence: (time.time() > deadline - 15) # Stop if time is up
            )
            
            if res_de.x is not None:
                # Polish the DE result
                if time.time() < deadline - 2:
                    res_polish = minimize(
                        objective_log, 
                        res_de.x, 
                        args=(50.0,),
                        method='L-BFGS-B', 
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res_polish.x is not None:
                        update_best(res_polish.x.reshape(N, D))
                else:
                    # If no time for polish, use DE result directly
                    update_best(res_de.x.reshape(N, D))
                    
        except Exception as e:
            pass

    # Fallback
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
