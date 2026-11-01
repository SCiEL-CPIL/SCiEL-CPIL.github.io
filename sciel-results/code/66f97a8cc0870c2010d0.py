import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    # log(sum(exp(tau * d))) / tau
    # We want log(max(d)) approx
    # Let's use the standard log-sum-exp for numerical stability
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    # -log(sum(exp(-tau * d))) / tau
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    # This is equivalent to log(max/min)
    # The raw score is (max/min)^2 = (max_sq/min_sq)
    # Minimizing log(max_sq/min_sq) is equivalent to minimizing the score.
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Initial candidates
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 1.0
    
    # Add center and one point on the sphere (randomly rotated for symmetry breaking)
    center = np.array([0.0, 0.0, 0.0])
    # Pick a random direction for the 14th point
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    
    init_ico = np.vstack([ico, center, p14])
    candidates = [init_ico.flatten()]
    
    # 2. Random points on a sphere
    for i in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        # Scale to roughly unit radius
        pts = pts * 1.0
        candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(5):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # Optimization loop
    n_restarts = 200
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        if i < len(candidates):
            x0 = candidates[i]
        else:
            # Random restart
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization with increasing tau
            # Stage 1: Low tau for broad exploration
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(10.0,),
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if res1.x is not None:
                pts1 = res1.x.reshape(N, D)
                update_best(pts1)
                
                # Stage 2: Medium tau
                if time.time() - start_time < timeout - 2:
                    res2 = minimize(
                        objective_smooth, 
                        res1.x, 
                        method='L-BFGS-B', 
                        args=(50.0,),
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
                        # Stage 3: High tau for precision
                        if time.time() - start_time < timeout - 1:
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(500.0,),
                                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res3.x is not None:
                                pts3 = res3.x.reshape(N, D)
                                update_best(pts3)
                                
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
