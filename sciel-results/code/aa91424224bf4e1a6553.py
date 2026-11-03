import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

# Ring structure: 1 center, then rings of 6, 12, 18, 13
RING_COUNTS = [1, 6, 12, 18, 13]
RING_RADII_INIT = [0.0, 0.25, 0.5, 0.75, 0.95]
RING_OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]

def _compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def _params_to_points(params):
    """Convert ring parameters (radii, offsets) to (N, 2) points array."""
    points = np.zeros((N, 2))
    idx = 0
    
    # Center
    points[0] = [0.0, 0.0]
    idx = 1
    
    for i, count in enumerate(RING_COUNTS[1:]):
        r = params[i]
        offset = params[i + len(RING_COUNTS) - 1]
        
        for k in range(count):
            angle = offset + 2 * np.pi * k / count
            points[idx, 0] = r * np.cos(angle)
            points[idx, 1] = r * np.sin(angle)
            idx += 1
            
    return points

def _objective(params):
    """Objective function for Differential Evolution.
    We want to maximize r, so we minimize -r.
    Add a small penalty for boundary violations to guide DE towards valid regions."""
    points = _params_to_points(params)
    r = _compute_radius(points)
    
    # If points are outside, r is 0. We want to avoid this.
    # A simple penalty: sum of squared boundary violations
    norms = np.linalg.norm(points, axis=1)
    violations = np.sum(np.maximum(0, norms - 1.0)**2)
    
    return -r + 10.0 * violations

def _local_polish(points, time_limit=10.0):
    """Perform local gradient-based optimization on the 100D space to fine-tune."""
    start_time = time.time()
    x0 = points.flatten()
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        pts = x.reshape(N, 2)
        # Compute soft-min objective for smoothness
        beta = 100.0
        norms = np.linalg.norm(pts, axis=1)
        f = 1.0 - norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        g = dists / 2.0
        
        all_margins = np.concatenate([f.ravel(), g.ravel()])
        
        # Log-sum-exp trick
        max_margin = np.max(all_margins)
        shifted = beta * (max_margin - all_margins)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_shifted))
        
        obj = max_margin - (1.0 / beta) * log_sum_exp
        
        # Gradient
        p = exp_shifted / np.sum(exp_shifted)
        grad = np.zeros_like(x)
        
        # Boundary
        for i in range(N):
            w = p[i]
            if norms[i] > 1e-10:
                grad[2*i:2*i+2] += w * (-pts[i] / norms[i])
        
        # Pairwise
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = p[N + i*N + j]
                if w < 1e-15:
                    continue
                d = dists[i, j]
                if d > 1e-10:
                    vec = pts[i] - pts[j]
                    grad[2*i:2*i+2] += w * (vec / (2.0 * d))
                    grad[2*j:2*j+2] += w * (-vec / (2.0 * d))
                    
        return -obj, -grad

    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x.reshape(N, 2)
    except Exception:
        pass
    
    return points

def construct():
    deadline = time.time() + DEADLINE
    
    # Bounds for DE:
    # Radii: [0.1, 1.0] for rings 1-4 (index 0-3 in params)
    # Offsets: [0, 2*pi] for rings 1-4 (index 4-7 in params)
    # Note: Center is fixed at 0,0.
    # We optimize 4 radii and 4 offsets = 8 parameters.
    # Wait, RING_COUNTS has 5 entries: 1, 6, 12, 18, 13.
    # So we have 4 rings to optimize (indices 1, 2, 3, 4 in RING_COUNTS).
    # Params: [r1, r2, r3, r4, o1, o2, o3, o4]
    
    bounds = [
        (0.1, 1.0),  # r1
        (0.2, 1.0),  # r2
        (0.4, 1.0),  # r3
        (0.6, 1.0),  # r4
        (0.0, 2*np.pi), # o1
        (0.0, 2*np.pi), # o2
        (0.0, 2*np.pi), # o3
        (0.0, 2*np.pi), # o4
    ]
    
    best_r = 0.0
    best_points = None
    
    # 1. Global Optimization with Differential Evolution
    # DE is robust and handles the low-dimensional space well.
    # We limit the time to leave room for polishing.
    de_time_limit = 60.0
    
    # DE callback to check time
    def de_callback(xk, convergence):
        if time.time() > time.time() + de_time_limit: # This logic is flawed in callback, use wrapper
             return True
        return False
    
    # Better approach: Run DE with a limited number of iterations/population
    # and check time after.
    
    print("Starting Differential Evolution...")
    de_start = time.time()
    
    try:
        # popsize=15, maxiter=100 is usually fast for 8D
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-8,
            polish=False # We will do our own polish
        )
        
        if result.x is not None:
            de_points = _params_to_points(result.x)
            de_r = _compute_radius(de_points)
            
            if de_r > best_r:
                best_r = de_r
                best_points = de_points
            
            print(f"DE Result: r={de_r:.10f}, time={time.time()-de_start:.2f}s")
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't run or was slow, we have a fallback initialization
    if best_points is None:
        # Fallback to a reasonable ring structure
        fallback_params = [0.25, 0.5, 0.75, 0.95, 0, np.pi/6, np.pi/12, np.pi/18]
        best_points = _params_to_points(fallback_params)
        best_r = _compute_radius(best_points)
        
    # 2. Local Polish with L-BFGS-B on full 100D space
    # This allows the points to break the rigid ring structure slightly if it helps,
    # or simply fine-tune the positions.
    
    polish_time_limit = min(40.0, deadline - time.time() - 5)
    
    if polish_time_limit > 5:
        print(f"Starting Local Polish (limit {polish_time_limit:.2f}s)...")
        polished_points = _local_polish(best_points, time_limit=polish_time_limit)
        polished_r = _compute_radius(polished_points)
        
        if polished_r > best_r:
            best_r = polished_r
            best_points = polished_points
            
        print(f"Polished Result: r={polished_r:.10f}")
        
    # 3. Final Validation and Safety
    if best_points is None:
        # Extreme fallback
        best_points = np.zeros((N, 2))
        # Spread them out roughly
        for i in range(1, N):
            angle = 2 * np.pi * i / (N-1)
            r = 0.8 * (i / (N-1))
            best_points[i] = [r * np.cos(angle), r * np.sin(angle)]
            
    # Ensure all points are strictly inside
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
            
    final_r = _compute_radius(best_points)
    print(f"Final Radius: {final_r:.10f}")
    
    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = _compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
