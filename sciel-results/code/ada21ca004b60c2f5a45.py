import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _equalize_ratio(pts):
    """
    Analytically refine the configuration to minimize the ratio.
    The ratio is (max_d / min_d)^2.
    This function attempts to find a scaling/translation (though translation doesn't affect ratio)
    and potentially a non-uniform adjustment? No, Euclidean distance is invariant to rotation/translation.
    However, the *shape* of the point set matters.
    
    A simple heuristic: 
    1. Identify the pair with min distance (i_min, j_min) and max distance (i_max, j_max).
    2. If they are the same pair (impossible for N>2), do nothing.
    3. We can try to "push" the min pair apart and "pull" the max pair together slightly?
       This is effectively local gradient descent on the ratio.
       
    Let's use a simple iterative scheme:
    For a few steps, compute gradients of log(max_d) and log(min_d).
    Actually, let's just do a few steps of Nelder-Mead on the ratio itself,
    starting from the SA result. Nelder-Mead is robust to non-smoothness.
    """
    if _t0 is None:
        return pts
    
    if time.time() - _t0 > DEADLINE - 2:
        return pts

    # Flatten for optimization
    x0 = pts.flatten()
    
    def obj(x):
        p = x.reshape(N, D)
        return _true_score(p)
    
    try:
        # Use Nelder-Mead with tight tolerances and limited iterations
        # This is a "polish" step.
        res = minimize(
            obj, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        pts_opt = res.x.reshape(N, D)
        
        # Check if it improved
        if obj(res.x) < obj(x0):
            return pts_opt
    except Exception:
        pass
        
    return pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(123)
    
    # 1. Initial Configuration: Fibonacci Sphere
    def get_fib_sphere():
        i = np.arange(N)
        # Slight perturbation to avoid symmetries that might trap SA
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    pts = get_fib_sphere()
    # Normalize to unit sphere radius 1 (arbitrary scale, ratio is scale invariant)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    current_score = _true_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 0.5
    T_end = 0.001
    alpha = 0.995 # Cooling rate per step
    steps = 5000
    
    # Precompute indices for efficiency? Not really needed for N=14.
    
    for step in range(steps):
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        T = T_start * (alpha ** step)
        
        # Generate a perturbation
        # Strategy: Mix of local moves and global moves
        move_type = rng.random()
        
        if move_type < 0.6:
            # Local: Displace one point
            idx = rng.integers(0, N)
            # Gaussian noise scaled by T
            noise = rng.normal(scale=T * 0.5, size=D)
            new_pts = pts.copy()
            new_pts[idx] += noise
            
        elif move_type < 0.8:
            # Global: Swap two points
            i, j = rng.choice(N, size=2, replace=False)
            new_pts = pts.copy()
            new_pts[i], new_pts[j] = new_pts[j].copy(), new_pts[i].copy()
            
        elif move_type < 0.9:
            # Local: Rotate a pair around their midpoint? Or just perturb two points towards each other
            i, j = rng.choice(N, size=2, replace=False)
            new_pts = pts.copy()
            # Move i towards j and j towards i slightly? Or random direction
            # Let's just perturb both
            noise_i = rng.normal(scale=T * 0.5, size=D)
            noise_j = rng.normal(scale=T * 0.5, size=D)
            new_pts[i] += noise_i
            new_pts[j] += noise_j
            
        else:
            # Global: Random restart of a subset? Or just a big kick
            idx = rng.integers(0, N)
            new_pts = pts.copy()
            # Replace point with a random point on the sphere (or near it)
            # Let's just add a large noise
            noise = rng.normal(scale=T * 2.0, size=D)
            new_pts[idx] += noise
            
        # Evaluate new point
        new_score = _true_score(new_pts)
        
        # Acceptance criterion
        if new_score < current_score:
            # Improvement, always accept
            pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Worse, accept with probability exp(-delta / T)
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta < 100 * T: 
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    current_score = new_score
            
    # 3. Final Polish
    # Use the best points found
    final_pts = best_pts.copy()
    
    # Try to polish from the current SA state as well, in case it's better but not the global best so far?
    # No, best_pts is the global best.
    
    # Run a few independent polish attempts with different initial "jitters" to escape local minima of the polish
    # But primarily rely on the best found.
    
    # One high-quality polish on the best found
    final_pts = _equalize_ratio(best_pts)
    
    # Verify score
    final_score = _true_score(final_pts)
    
    # If the polish made it worse (due to numerical issues or bad local min), revert
    if final_score > best_score:
        final_pts = best_pts
    else:
        best_score = final_score
        best_pts = final_pts

    return best_pts

if __name__ == "__main__":
    # Simple test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Score: {_true_score(pts):.4f}")
    print(pts)
