import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
H = np.sqrt(3) / 2

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute indices for all triples
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)
n_triples = len(idx_i)

def compute_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    points: (N, 2) array
    Returns: array of shape (n_triples,) with areas
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def min_area(points):
    """
    Returns the minimum area of any triangle formed by 3 points.
    """
    return np.min(compute_areas(points))

def smooth_min_area(points, gamma=100.0):
    """
    A smooth approximation of the minimum area using the log-sum-exp trick.
    This is differentiable and can be used with gradient-based methods.
    """
    areas = compute_areas(points)
    # Avoid log of zero
    areas = np.maximum(areas, 1e-12)
    # Smooth minimum: -1/gamma * log(sum(exp(-gamma * areas)))
    # This is approximately min(areas) when gamma is large
    max_val = np.max(areas)
    # For numerical stability, subtract max
    # Actually, for min, we want to be careful.
    # logsumexp(-gamma * areas) = log(sum(exp(-gamma * areas)))
    # min(areas) ≈ -1/gamma * logsumexp(-gamma * areas)
    
    # Let's use a stable logsumexp
    vals = -gamma * areas
    max_vals = np.max(vals)
    log_sum_exp = max_vals + np.log(np.sum(np.exp(vals - max_vals)))
    
    return -1.0 / gamma * log_sum_exp

def params_to_points(params):
    """
    Maps a flat array of 2*N parameters (u_i, v_i) to points in the triangle.
    Uses normalization to ensure u+v <= 1.
    """
    params = np.array(params).reshape(N, 2)
    u = params[:, 0].copy()
    v = params[:, 1].copy()
    
    # If u+v > 1, normalize to lie on the edge u+v=1 (mapping square to triangle)
    # This ensures points stay inside the triangle.
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    
    # Points = A + u*(B-A) + v*(C-A)
    # B-A = (1, 0)
    # C-A = (0.5, H)
    x = u * 1.0 + v * 0.5
    y = u * 0.0 + v * H
    
    return np.column_stack((x, y))

def objective(params, gamma=100.0):
    """
    Objective function to minimize (negative of smooth min triangle area).
    """
    points = params_to_points(params)
    return -smooth_min_area(points, gamma)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse initial populations
    # We will run multiple independent local optimizations from different starting points.
    # Starting points:
    # 1. Random uniform in triangle
    # 2. Perturbed grid
    # 3. Perturbed known good configurations (if any, but we don't have them yet, so just random)
    
    start_points = []
    
    # Random starts
    for _ in range(50):
        # Sample u, v such that u>=0, v>=0, u+v<=1
        # One way: sample r1, r2 from uniform(0,1), then u = r1 * r2, v = (1-r1) * r2
        # This gives a uniform distribution over the triangle.
        r1 = rng.random(N)
        r2 = rng.random(N)
        u = r1 * r2
        v = (1 - r1) * r2
        params = np.column_stack((u, v)).flatten()
        start_points.append(params)
    
    # Grid-based starts with perturbation
    # Divide triangle into a grid
    n_grid = 4
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u_center = (i + 0.5) / n_grid
            v_center = (j + 0.5) / n_grid
            if u_center + v_center <= 1.0:
                # Create a grid of points around this center
                params = np.zeros(2 * N)
                for k in range(N):
                    # Spread points in a small circle around the center
                    angle = 2 * np.pi * k / N
                    radius = 0.1
                    u_k = u_center + radius * np.cos(angle)
                    v_k = v_center + radius * np.sin(angle)
                    # Ensure u_k, v_k >= 0 and u_k+v_k <= 1
                    if u_k < 0: u_k = 0
                    if v_k < 0: v_k = 0
                    if u_k + v_k > 1:
                        s = u_k + v_k
                        u_k /= s
                        v_k /= s
                    params[2*k] = u_k
                    params[2*k+1] = v_k
                start_points.append(params)
    
    # Sort start points by initial score (best first) to try good ones early
    # But we want diversity, so maybe just iterate.
    
    # Phase 2: Local optimization using L-BFGS-B on the smooth objective
    # We will try each start point, but limit the number of iterations and total time.
    
    # To speed up, we can pre-compute the Jacobian? No, let's use finite differences first.
    # L-BFGS-B is efficient for this size (22 parameters).
    
    gamma = 50.0  # Start with a moderate gamma, increase later?
    
    for start_idx, start_params in enumerate(start_points):
        if time.time() > deadline - 10.0:
            break
            
        # Try to optimize from this start
        try:
            res = minimize(
                objective,
                start_params,
                args=(gamma,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success:
                # Evaluate the true (non-smooth) objective on the result
                points = params_to_points(res.x)
                score = min_area(points)
                
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
    
    # Phase 3: Refine the best solution with a higher gamma (sharper minimum)
    if best_params is not None and time.time() < deadline - 10.0:
        gamma_refine = 200.0
        try:
            res = minimize(
                objective,
                best_params,
                args=(gamma_refine,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                points = params_to_points(res.x)
                score = min_area(points)
                
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
    
    # Phase 4: Final local search with small random perturbations (simulated annealing style)
    if best_params is not None and time.time() < deadline - 5.0:
        current_params = best_params.copy()
        current_score = best_score
        step_size = 0.01
        no_improve_count = 0
        
        while time.time() < deadline - 2.0 and no_improve_count < 100:
            noise = rng.normal(0, step_size, size=(2 * N,))
            new_params = np.clip(current_params + noise, 0.0, 1.0)
            
            points = params_to_points(new_params)
            new_score = min_area(points)
            
            if new_score > current_score:
                current_score = new_score
                current_params = new_params
                no_improve_count = 0
                # Slowly decrease step size
                step_size *= 0.99
            else:
                no_improve_count += 1
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
    
    # Convert best parameters to points
    if best_params is None:
        # Fallback: random points
        best_params = rng.random(2 * N)
        # Normalize
        u = best_params[0::2]
        v = best_params[1::2]
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        best_params = np.column_stack((u, v)).flatten()
        
    points = params_to_points(best_params)
    
    # Final sanity check: ensure points are within triangle
    # The mapping guarantees this, but let's be safe.
    # Check if any point is outside
    # For each point (x, y):
    # y >= 0
    # x >= 0
    # y <= H * x  (edge AC? No, AC is from (0,0) to (0.5, H), so y = 2*sqrt(3)*x? No.
    # Let's check the barycentric coordinates.
    # A=(0,0), B=(1,0), C=(0.5, H)
    # For a point P=(x,y):
    # y >= 0
    # y <= H * (1 - x) ? No.
    # Line BC: from (1,0) to (0.5, H). Slope = (H-0)/(0.5-1) = -2H.
    # y - 0 = -2H (x - 1) => y = -2Hx + 2H => 2Hx + y = 2H => x + y/H = 1.
    # So y <= H(1-x) or x + y/H <= 1.
    # Line AC: from (0,0) to (0.5, H). Slope = H/0.5 = 2H.
    # y = 2Hx => y <= 2Hx.
    # So conditions: y >= 0, y <= 2Hx, x + y/H <= 1.
    
    # Our mapping u,v >= 0, u+v<=1 ensures:
    # x = u + 0.5v
    # y = H*v
    # y >= 0 since v>=0.
    # y <= 2Hx => H*v <= 2H(u + 0.5v) => v <= 2u + v => 0 <= 2u, which is true.
    # x + y/H <= 1 => u + 0.5v + v <= 1 => u + 1.5v <= 1.
    # Wait, the condition for the triangle is u+v <= 1.
    # Does u+v <= 1 imply x + y/H <= 1?
    # x + y/H = u + 0.5v + v = u + 1.5v.
    # If u+v <= 1, is u+1.5v <= 1?
    # Not necessarily. Example: u=0, v=1. u+v=1. x+1.5v = 1.5 > 1.
    # But the point (x,y) = (0.5, H) is vertex C, which is inside.
    # The condition for being inside the triangle defined by A,B,C in Cartesian coords is:
    # y >= 0
    # y <= 2Hx (below line AC)
    # y <= -2H(x-1) (below line BC) => y <= 2H(1-x)
    
    # Let's re-verify the mapping.
    # P = A + u(B-A) + v(C-A)
    # This is the standard barycentric mapping where u,v are the weights for B and C.
    # The condition for P to be inside the triangle is u>=0, v>=0, u+v<=1.
    # This is correct. The Cartesian inequalities I derived above must be consistent with this.
    # Let's check the vertex C: u=0, v=1. x=0.5, y=H.
    # y <= 2Hx => H <= 2H(0.5) => H <= H. OK.
    # y <= 2H(1-x) => H <= 2H(0.5) => H <= H. OK.
    # Let's check the midpoint of BC: u=0.5, v=0.5.
    # x = 0.5 + 0.25 = 0.75. y = 0.5H.
    # y <= 2Hx => 0.5H <= 1.5H. OK.
    # y <= 2H(1-x) => 0.5H <= 2H(0.25) = 0.5H. OK.
    
    # So the mapping is correct.
    
    return points

if __name__ == "__main__":
    points = construct()
    print(points)
    print("Min Area:", min_area(points))
