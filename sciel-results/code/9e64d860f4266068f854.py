import time
import numpy as np
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 148.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_initial_points(seed):
    """
    Generate diverse initial configurations.
    1. Fibonacci sphere
    2. Icosahedron + 2 random
    3. Random Gaussian
    """
    rng = np.random.default_rng(seed)
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts_fib = np.column_stack([x, y, z])
    pts_fib = pts_fib - pts_fib.mean(axis=0)
    inits.append(pts_fib)
    
    # 2. Icosahedron vertices (12) + 2 points
    phi_gold = (1 + np.sqrt(5)) / 2
    ico_pts = np.array([
        [0, 1, phi_gold], [0, -1, phi_gold],
        [0, 1, -phi_gold], [0, -1, -phi_gold],
        [1, phi_gold, 0], [-1, phi_gold, 0],
        [1, -phi_gold, 0], [-1, -phi_gold, 0],
        [phi_gold, 0, 1], [-phi_gold, 0, 1],
        [phi_gold, 0, -1], [-phi_gold, 0, -1]
    ])
    ico_pts = ico_pts / np.linalg.norm(ico_pts, axis=1)[:, None]
    
    # Add 2 points at "holes" or random
    extra = rng.normal(size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1)[:, None]
    pts_ico = np.vstack([ico_pts, extra])
    pts_ico = pts_ico - pts_ico.mean(axis=0)
    inits.append(pts_ico)
    
    # 3. Random
    pts_rand = rng.normal(size=(N, 3))
    pts_rand = pts_rand - pts_rand.mean(axis=0)
    inits.append(pts_rand)
    
    return inits

def _simulated_annealing(pts_init, rng, time_limit):
    """
    Perform Simulated Annealing to minimize the ratio (max_dist / min_dist)^2.
    
    Strategy:
    - Objective: Minimize R = (max_d / min_d)^2.
    - Equivalent to maximizing min_d subject to max_d being small, or vice versa.
    - We use a geometric temperature schedule.
    - Moves:
        1. Displace one point randomly.
        2. "Repulsion" move: Push two closest points apart.
        3. "Attraction" move: Pull two farthest points together.
    """
    best_pts = pts_init.copy()
    best_score = _true_score(best_pts)
    current_pts = pts_init.copy()
    current_score = best_score
    
    # Initial temperature: High enough to accept bad moves
    # Score range is roughly [1, 100+]. Let's start T around 5.0
    T = 5.0
    cooling_rate = 0.995  # Geometric cooling
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - _t0 < time_limit:
        # Check time every few iterations to avoid overhead
        if iterations % 100 == 0 and time.time() - start_time > (time_limit - (time.time() - _t0)):
            break
            
        # Generate a move
        move_type = rng.integers(0, 3)
        new_pts = current_pts.copy()
        
        if move_type == 0:
            # Displace one point
            idx = rng.integers(0, N)
            # Step size decreases with temperature? Or fixed? 
            # Let's use a step size that scales with T to allow large jumps early
            step_size = 0.2 + 0.5 * T
            new_pts[idx] += rng.normal(scale=step_size, size=D)
            
        elif move_type == 1:
            # Repel closest pair
            d = cdist(current_pts, current_pts)
            np.fill_diagonal(d, np.inf)
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx
            # Push them apart
            vec = current_pts[j] - current_pts[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            # Move i away from j, and j away from i
            step = 0.05 + 0.1 * T
            new_pts[i] -= vec * step
            new_pts[j] += vec * step
            
        else:
            # Attract farthest pair
            d = cdist(current_pts, current_pts)
            np.fill_diagonal(d, np.inf)
            max_idx = np.unravel_index(np.argmax(d), d.shape)
            i, j = max_idx
            # Pull them together
            vec = current_pts[j] - current_pts[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            step = 0.05 + 0.1 * T
            new_pts[i] += vec * step
            new_pts[j] -= vec * step
            
        # Evaluate new point
        new_score = _true_score(new_pts)
        
        # Metropolis criterion
        if new_score < current_score:
            # Always accept improvement
            current_pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept bad move with probability exp(-delta / T)
            delta = new_score - current_score
            if delta < 1e-6: # Avoid log(0) or overflow issues if T is tiny
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_pts = new_pts
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        
        # If T is very small, we might be stuck. 
        # Restart with a higher T if no improvement for a long time?
        # For now, simple geometric cooling.
        
        iterations += 1
        
        # Safety check
        if time.time() - _t0 > time_limit:
            break
            
    return best_pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Use multiple random seeds for diversity in initializations
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if time.time() - _t0 > DEADLINE - 15:
            break
            
        inits = _generate_initial_points(seed)
        
        for pts_init in inits:
            if time.time() - _t0 > DEADLINE - 10:
                break
                
            rng = np.random.default_rng(seed + int(time.time() * 1000) % 10000)
            
            # Run SA for a fixed duration or until time runs out
            # We want to run SA for as long as possible for each init, but we need to finish all inits.
            # Strategy: Run SA for a chunk of time.
            # If we have 148s, and 8 seeds * 3 inits = 24 runs.
            # That's ~6s per run. That's a bit short for SA to converge well.
            # Let's reduce the number of inits per seed or seeds.
            
            # Let's do 4 seeds, 3 inits each = 12 runs. ~12s per run. Better.
            # Or 2 seeds, 3 inits each = 6 runs. ~24s per run.
            
            # Let's dynamically allocate time.
            # Actually, the loop above iterates through seeds. 
            # Let's just run SA with a time limit that allows finishing the current batch.
            
            # A better approach: Run SA for a fixed number of iterations or time slice.
            # Let's give each SA run a slice of the remaining time, but ensure we finish at least one.
            
            # Simplified: Run SA for ~10-15 seconds per configuration if time allows.
            # But we have 24 configs. 148 / 24 ~ 6s.
            # Let's prioritize the first few configs which are usually better (Fibonacci).
            
            # Revised Plan:
            # 1. Run SA on the best initializations (Fibonacci) for longer.
            # 2. Run SA on others for shorter.
            
            # For simplicity in this implementation, let's just run SA with a time limit 
            # that is a fraction of the remaining time, ensuring we don't run out.
            
            # Let's say we have R remaining time. We have K configs left.
            # Time per config = R / K.
            
            # We don't know K exactly inside the loop without counting.
            # Let's just use a fixed time per SA run, say 5 seconds, and see how many we fit.
            
            time_per_sa = 5.0
            sa_end_time = time.time() + time_per_sa
            
            pts_opt = _simulated_annealing(pts_init, rng, sa_end_time)
            score_opt = _true_score(pts_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                
            # If we find a very good score, we can spend more time refining it?
            # Or just let the next SA run start from a perturbed version of the best?
            
    # Fallback
    if best_pts is None:
        # Return Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        
    return best_pts
