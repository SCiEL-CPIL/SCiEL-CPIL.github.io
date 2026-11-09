# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import differential_evolution, minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective(x):
    """
    Objective function for differential_evolution (minimization).
    We want to maximize (d_min / d_max)^2, so we minimize its negative.
    x is a flat array of shape (32,).
    """
    points = x.reshape(-1, 2)
    score = calculate_score(points)
    return -score

def hexagonal_init(n):
    """Generate a hexagonal grid initialization."""
    points = []
    # 4x4 grid is 16 points
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    # Normalize to roughly fit in [-1, 1]
    center = pts.mean(axis=0)
    pts = pts - center
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / max_dist * 0.9
    return pts

def random_init(n, seed=42):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    # Random points in unit circle
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def polish_points(points, time_limit=10.0):
    """
    Use Simulated Annealing to polish the points found by DE.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    temp = 0.05
    min_temp = 1e-7
    cooling_rate = 0.999
    
    rng = np.random.RandomState(int(time.time()))
    
    while time.time() - start_time < time_limit:
        if temp < min_temp:
            break
            
        # Move a random point
        idx = rng.randint(n)
        delta = rng.randn(2) * temp
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Accept with probability
            delta_score = new_score - current_score
            if delta_score < 0:
                if rng.rand() < np.exp(delta_score / temp):
                    current_points = new_points
                    current_score = new_score
        
        temp *= cooling_rate
    
    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580 # Safety margin
    
    # 1. Generate diverse initial candidates
    # We will run DE on a few different initializations or just one robust DE run.
    # DE is population-based, so it explores widely.
    
    # Bounds for DE: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (n * 2)
    
    # Run Differential Evolution
    # popsize: 15 (default) or higher for better exploration
    # maxiter: adjust based on time
    # tol: stop when improvement is small
    
    # We'll try to run DE for a significant portion of the time, then polish.
    # DE with 32 variables and popsize 15 needs ~15 * 32 evaluations per generation.
    # Let's estimate evaluations per second. calculate_score is fast (pdist).
    
    # Strategy: 
    # 1. Run DE for ~300 seconds.
    # 2. Polish the best result for ~100 seconds.
    # 3. Return best.
    
    # To be safe with DE's internal timing, we use maxiter and let it run.
    # We can't easily interrupt DE mid-run without a callback, but we can set a generous maxiter 
    # and rely on the fact that it will converge or hit the limit.
    # However, DE doesn't have a time limit, only maxiter.
    # We need to estimate maxiter.
    
    # Let's do a quick benchmark of calculate_score speed to estimate iterations.
    test_pts = hexagonal_init(n).flatten()
    start = time.time()
    for _ in range(100):
        objective(test_pts)
    end = time.time()
    time_per_eval = (end - start) / 100
    
    # DE evaluations per iteration = popsize * n_vars (roughly, actually popsize * 1 mutation per individual per gen? No, DE evaluates popsize individuals per generation)
    # Standard DE: popsize individuals are evaluated per generation.
    popsize = 20
    evals_per_gen = popsize
    
    # Available time for DE: 300 seconds
    time_for_de = 300
    est_max_iter = int((time_for_de / time_per_eval) / evals_per_gen)
    
    # Cap max_iter to prevent running too long if estimate is off
    est_max_iter = max(100, min(est_max_iter, 1000))
    
    best_result = None
    best_score = -1.0
    
    # Run DE with a few different seeds to increase chance of finding global optimum
    # But DE is stochastic. One run with high popsize is usually enough.
    # Let's do 2 runs with different popsize/seeds if time permits, or just 1 strong run.
    
    # Run 1: Standard DE
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=est_max_iter, 
            popsize=20, 
            seed=42,
            tol=1e-6,
            polish=False # We will polish manually
        )
        if result.success and result.fun < 0:
            points_de = result.x.reshape(-1, 2)
            score_de = -result.fun
            if score_de > best_score:
                best_score = score_de
                best_result = points_de
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't find anything good, or to improve, run a second DE with different settings?
    # Or just polish the result.
    
    if best_result is None:
        # Fallback to hexagonal grid if DE failed completely
        best_result = hexagonal_init(n)
        best_score = calculate_score(best_result)
        
    # Polish the best result found
    if time.time() < deadline - 110:
        polish_time = min(100, deadline - time.time() - 10)
        polished_points = polish_points(best_result, time_limit=polish_time)
        polished_score = calculate_score(polished_points)
        
        if polished_score > best_score:
            best_result = polished_points
            best_score = polished_score
    
    return best_result

# EVOLVE-BLOCK-END
