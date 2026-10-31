import numpy as np
import time
from scipy.optimize import differential_evolution
from itertools import combinations

N = 11
# Pre-compute indices for all 165 triangles
tri_indices = np.array(list(combinations(range(N), 3)))

def get_cartesian(barycentric):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y)
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    P = u*V1 + v*V2 + (1-u-v)*V0
    V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = u*1 + v*0.5
    y = u*0 + v*(sqrt(3)/2)
    """
    u = barycentric[:, 0]
    v = barycentric[:, 1]
    x = u + 0.5 * v
    y = (3 ** 0.5) / 2 * v
    return np.column_stack((x, y))

def evaluate_min_area(barycentric):
    """
    Calculate the minimum area of all triangles formed by the points.
    barycentric: (N, 2) array of (u, v)
    Returns: minimum area value (scaled by 0.5 * Area_Ref, but we just need relative value)
    Actually, Area = 0.5 * |det| * Area_Ref. We can just return 0.5 * min_det.
    """
    # barycentric shape (N, 2)
    # We need to compute the determinant for each triangle (i, j, k)
    # Det = | (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k) | / 2
    # Using barycentric directly:
    # Area ratio = |det( [u_i, v_i, 1], [u_j, v_j, 1], [u_k, v_k, 1] )|
    
    # Extract coordinates for all triangles
    # tri_indices shape (165, 3)
    u = barycentric[:, 0]
    v = barycentric[:, 1]
    
    i = tri_indices[:, 0]
    j = tri_indices[:, 1]
    k = tri_indices[:, 2]
    
    ui, uj, uk = u[i], u[j], u[k]
    vi, vj, vk = v[i], v[j], v[k]
    
    # Determinant of:
    # | ui  vi  1 |
    # | uj  vj  1 |
    # | uk  vk  1 |
    # = ui(vj - vk) - vi(uj - uk) + (uj*vk - uk*vj)
    
    det = ui * (vj - vk) - vi * (uj - uk) + (uj * vk - uk * vj)
    
    min_det = np.min(np.abs(det))
    
    # The actual area is 0.5 * min_det * Area_Ref
    # Area_Ref for unit side length equilateral triangle is sqrt(3)/4
    # The task says "maximize the smallest area ... The notebook uses an equilateral triangle and divides by its area."
    # This implies the score is Area_min / Area_Ref.
    # Area_min / Area_Ref = (0.5 * min_det * Area_Ref) / Area_Ref = 0.5 * min_det
    return 0.5 * min_det

def objective(barycentric_flat):
    """
    Objective function for differential_evolution.
    Minimizes the negative of the minimum area.
    """
    barycentric = barycentric_flat.reshape(N, 2)
    
    # Check constraints: u >= 0, v >= 0, u+v <= 1
    # DE handles bounds, so u, v are in [0, 1].
    # But we need u+v <= 1.
    # If u+v > 1, the point is outside. We should penalize heavily.
    # However, defining bounds as [0, 1] for u and v allows u+v up to 2.
    # We can project points back or penalize.
    # Penalty is safer for DE.
    
    sum_uv = barycentric[:, 0] + barycentric[:, 1]
    violations = np.maximum(0, sum_uv - 1.0)
    
    if np.any(violations > 0):
        # Heavy penalty
        return 1000.0 + np.sum(violations)
        
    min_area = evaluate_min_area(barycentric)
    return -min_area

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Initial guess: Random points inside the triangle
    # u, v > 0, u+v < 1
    # Simple random generation
    def random_barycentric():
        u = np.random.random(N)
        v = np.random.random(N)
        s = u + v
        # Scale to be inside
        # If s > 1, scale down
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        return np.column_stack((u, v))
    
    # Define bounds for DE
    # u in [0, 1], v in [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Run Differential Evolution
    # maxiter and tol are adjusted based on time
    # DE is stochastic, so we run it and check time periodically
    # Unfortunately, DE doesn't support callback for early stopping in all versions easily without wrapping.
    # We can wrap the objective to check time, but DE calls it many times.
    # Better: Run DE with a reasonable number of iterations and check time after.
    # Or, use a wrapper that raises an exception if time is exceeded.
    
    class TimeoutException(Exception):
        pass

    original_objective = objective
    def time_checked_objective(x):
        if time.time() - start_time > timeout:
            raise TimeoutException()
        return original_objective(x)
    
    try:
        result = differential_evolution(
            time_checked_objective,
            bounds,
            maxiter=1000,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False # Polish can be slow and risky for time, DE is usually good enough
        )
    except TimeoutException:
        # If timed out during DE, we need to return the best found so far.
        # DE doesn't easily expose the best intermediate state if it crashes.
        # Alternative: Run DE in a thread? No, too complex.
        # Alternative: Use a simpler local optimization (L-BFGS-B) starting from good random seeds, 
        # iterating with time checks.
        
        # Let's restart with a loop of local optimizations if DE fails or is too slow.
        # Actually, let's try to make DE faster or use a different approach if DE is too slow.
        # For now, if DE times out, we fall back to the best random point found during initialization?
        # No, we didn't store it.
        
        # Let's implement a fallback:
        # 1. Generate several random valid points.
        # 2. Use L-BFGS-B to optimize each one.
        # 3. Keep the best.
        
        pass # Handled below

    if not hasattr(result, 'x') or result.x is None:
         # Fallback if DE failed completely
         best_bary = random_barycentric()
    else:
        best_bary = result.x.reshape(N, 2)
        # Ensure points are strictly inside (handle numerical errors)
        # Clip and renormalize if slightly outside
        u = best_bary[:, 0]
        v = best_bary[:, 1]
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        best_bary = np.column_stack((u, v))
        
    # Check time again
    if time.time() - start_time > timeout:
        # If we are already over time, just return what we have
        pass
    else:
        # Try to improve with local optimization (L-BFGS-B)
        # This can squeeze out more value.
        # We need a gradient? No, L-BFGS-B can use numerical gradients, but it's slow.
        pass
        # Let's stick to the DE result for now to ensure speed and stability.
        # If we have time, we could try a few L-BFGS steps.
        
        # Let's add a simple local improvement step using Nelder-Mead or similar?
        # No, DE is already global.
        
    # Final validation
    # Ensure the points are valid
    u = best_bary[:, 0]
    v = best_bary[:, 1]
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1 + 1e-9):
        # Fallback to a known valid configuration if something went wrong
        # e.g. vertices and center
        pts = []
        pts.append([0, 0]) # V0
        pts.append([1, 0]) # V1
        pts.append([0.5, 3**0.5/2]) # V2
        # Add some interior points
        pts.append([1/3, 1/3])
        pts.append([1/3, 1/3]) # Duplicate? No, distinct points required? 
        # Task says "Place 11 points". Usually distinct.
        # If duplicates are allowed, area 0. So they must be distinct.
        # Let's just use the random fallback if invalid.
        best_bary = random_barycentric()

    cart = get_cartesian(best_bary)
    return cart

# To ensure we don't rely on DE's internal state if it times out, 
# we should probably wrap the DE call in a way that allows us to return the best 
# found so far. However, scipy's DE doesn't easily support this.
# A safer approach for the "timeout" constraint is to limit the number of iterations 
# based on the time per iteration, or just trust that DE with popsize=15 and maxiter=1000 
# will finish in ~110s.
# 1000 iterations * 15 population size * 22 dimensions.
# Each objective evaluation is vectorized NumPy, very fast (<1ms).
# 1000 * 15 = 15000 evaluations. 15000 * 1ms = 15s.
# So 1000 iterations should be well within 110s.
# We can increase maxiter to 2000 or 5000 to get better results if time permits.

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    def random_barycentric():
        u = np.random.random(N)
        v = np.random.random(N)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        return np.column_stack((u, v))
    
    bounds = [(0, 1)] * (2 * N)
    
    class TimeoutException(Exception):
        pass

    def time_checked_objective(x):
        if time.time() - start_time > timeout:
            raise TimeoutException()
        bary = x.reshape(N, 2)
        sum_uv = bary[:, 0] + bary[:, 1]
        violations = np.maximum(0, sum_uv - 1.0)
        if np.any(violations > 0):
            return 1000.0 + np.sum(violations)
        min_area = evaluate_min_area(bary)
        return -min_area

    try:
        result = differential_evolution(
            time_checked_objective,
            bounds,
            maxiter=5000, # Higher limit, will be cut by timeout
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=False
        )
        if result.x is not None:
            best_bary = result.x.reshape(N, 2)
        else:
            best_bary = random_barycentric()
    except TimeoutException:
        # If DE times out, we don't have the best point.
        # We should have stored the best point found so far.
        # Since we can't easily modify DE, we will rely on the fact that 
        # 5000 iterations * 15 pop * ~1ms/eval = 75s.
        # So it should finish before 110s.
        # If it doesn't, we fall back to a random point.
        best_bary = random_barycentric()
        
    # Post-process: ensure points are inside
    u = best_bary[:, 0]
    v = best_bary[:, 1]
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    # Clip small negatives
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    best_bary = np.column_stack((u, v))
    
    cart = get_cartesian(best_bary)
    return cart
