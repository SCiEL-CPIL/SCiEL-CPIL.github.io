import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian coordinates.
    w = 1 - u - v
    A=(0,0), B=(1,0), C=(0.5, H)
    P = u*A + v*B + w*C
    P = v*(1,0) + (1-u-v)*(0.5, H)
    P = (v + 0.5 - 0.5u - 0.5v, H - Hu - Hv)
    P = (0.5 + 0.5v - 0.5u, H(1 - u - v))
    """
    x = 0.5 + 0.5 * v - 0.5 * u
    y = H * (1 - u - v)
    return np.array([x, y])

def decode(params):
    """
    Decode a flat array of 2*N barycentric coordinates (u, v) into N Cartesian points.
    """
    points = []
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        # Ensure feasibility: u>=0, v>=0, u+v<=1
        # If not, clip to boundary? 
        # For DE, we can just let it explore and penalize in objective.
        # But for decoding to valid points for area calculation, we should clip or reject.
        # Let's clip to valid region to ensure we always return a valid point set.
        if u < 0: u = 0
        if v < 0: v = 0
        if u + v > 1:
            s = u + v
            u /= s
            v /= s
        points.append(bary_to_cart(u, v))
    return np.array(points)

def compute_areas(points):
    """
    Compute areas of all triangles formed by 3 points.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def min_area(points):
    areas = compute_areas(points)
    return np.min(areas)

def smooth_min_area(points, alpha=100.0):
    """
    Smooth approximation of min area using log-sum-exp.
    We want to maximize min(areas).
    Equivalent to minimizing -min(areas).
    Smooth min: -1/alpha * log(sum(exp(-alpha * areas)))
    So we minimize this smooth function.
    """
    areas = compute_areas(points)
    # Prevent overflow in exp
    # max_val = np.max(areas)
    # exp_terms = np.exp(-alpha * (areas - max_val))
    # val = -1/alpha * (np.log(np.sum(exp_terms)) - alpha * max_val)
    # This is -min approx.
    
    # Simpler:
    # f(x) = -1/alpha * ln(sum(exp(-alpha x)))
    # We minimize f(points).
    
    # To avoid numerical issues, shift by max area
    max_a = np.max(areas)
    shifted = areas - max_a
    exp_sum = np.sum(np.exp(-alpha * shifted))
    if exp_sum == 0:
        return 0.0
    return -1.0/alpha * (np.log(exp_sum) - alpha * max_a)

def objective_de(params):
    """
    Objective for Differential Evolution.
    We minimize -min_area.
    If points are invalid (which shouldn't happen with clipping in decode), return large penalty.
    """
    points = decode(params)
    # Check if points are distinct? 
    # If points coincide, area is 0.
    # The clipping in decode ensures points are in triangle.
    # We just need to minimize -min_area.
    a = min_area(points)
    return -a

def objective_smooth(params):
    """
    Objective for Nelder-Mead polishing.
    Minimizes the smooth approximation of -min_area.
    """
    points = decode(params)
    return smooth_min_area(points, alpha=50.0)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s budget, leaving 20s margin
    
    # Bounds for barycentric coordinates u, v
    # u in [0, 1], v in [0, 1]
    # The constraint u+v<=1 is handled by clipping in decode.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -1.0
    
    # Phase 1: Differential Evolution
    # popsize=10, maxiter=100 might be too slow for 22 dims?
    # Let's try popsize=15, maxiter=50.
    # DE is stochastic, so we run it once with a generous budget.
    try:
        result = differential_evolution(
            objective_de, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False # We will polish ourselves
        )
        
        if time.time() < deadline:
            best_params = result.x
            best_score = -result.fun # result.fun is -min_area, so score is min_area
            # Actually result.fun is the value of objective_de, which is -min_area.
            # So min_area = -result.fun
            
    except Exception as e:
        pass
    
    # If DE didn't run or failed, try a random start
    if best_params is None:
        rng = np.random.default_rng(42)
        best_params = rng.uniform(0, 1, size=2*N)
        # Ensure valid
        best_params = decode(best_params) # Wait, decode returns cart. We need params.
        # Just use random valid barycentric
        u = rng.uniform(0, 1, N)
        v = rng.uniform(0, 1, N)
        # Scale to ensure u+v <= 1?
        # Simple random valid points:
        # Generate random points in triangle
        best_params = []
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            best_params.extend([r1, r2])
        best_params = np.array(best_params)
        best_score = -objective_de(best_params)

    # Phase 2: Polishing with Nelder-Mead using smooth objective
    # We can run multiple polishing steps from the best DE result.
    # Or just one strong polish.
    
    current_params = best_params.copy()
    current_score = -objective_de(current_params)
    
    # Run Nelder-Mead
    try:
        # Use the smooth objective for the optimizer
        # Initial simplex size
        # xatol, fatol
        res = minimize(
            objective_smooth, 
            current_params, 
            method='Nelder-Mead', 
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 2000}
        )
        
        if time.time() < deadline:
            # Evaluate the actual (non-smooth) objective for the result
            new_score = -objective_de(res.x)
            if new_score > current_score:
                current_params = res.x
                current_score = new_score
                
    except Exception as e:
        pass

    # Phase 3: Local Nudge Search (Discrete refinement)
    # Try small perturbations to break symmetry or fine-tune.
    # This is optional if NM was good, but NM can get stuck in local mins of the smooth approx.
    # A simple local search:
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        step_size = 0.05
        for _ in range(500):
            if time.time() > deadline - 5: break
            # Pick a random point and nudge it
            idx = rng.integers(0, N)
            u_idx = 2*idx
            v_idx = 2*idx+1
            
            old_u = current_params[u_idx]
            old_v = current_params[v_idx]
            
            new_u = old_u + rng.uniform(-step_size, step_size)
            new_v = old_v + rng.uniform(-step_size, step_size)
            
            # Clip to bounds
            new_u = np.clip(new_u, 0, 1)
            new_v = np.clip(new_v, 0, 1)
            if new_u + new_v > 1:
                s = new_u + new_v
                new_u /= s
                new_v /= s
                
            trial_params = current_params.copy()
            trial_params[u_idx] = new_u
            trial_params[v_idx] = new_v
            
            trial_score = -objective_de(trial_params)
            
            if trial_score > current_score:
                current_params = trial_params
                current_score = trial_score

    # Final check
    if time.time() > deadline:
         # Return whatever we have
         pass

    best_points = decode(current_params)
    return best_points

# Test
if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Min Area:", min_area(pts))
