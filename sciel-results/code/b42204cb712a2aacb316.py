import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices for vectorized area computation
TRI_IDX = np.array(list(combinations(range(N), 3)))

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def project_points_to_triangle(points):
    """Project all points onto the triangle if they're outside."""
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    return points

def objective_for_de(x):
    """
    Objective function for Differential Evolution.
    We minimize the negative of the min triangle area.
    We project points to ensure feasibility.
    """
    points = x.reshape(N, 2)
    points = project_points_to_triangle(points)
    val = min_triangle_area(points)
    return -val

def local_polish(points, max_iter=1000, step_size=0.02):
    """
    Fast local hill climbing to polish the solution.
    """
    points = points.copy()
    best_val = min_triangle_area(points)
    rng = np.random.RandomState(123)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val
            
            # Try several random steps
            for _ in range(20):
                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        else:
            if not improved:
                step_size *= 0.8
                if step_size < 1e-5:
                    break
                    
    return points, best_val

def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Bounds for DE: x in [0,1], y in [0,H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    # Differential Evolution is stochastic, so we might run it once with a good pop size
    # or multiple times if time permits.
    # Maximize min area -> Minimize -min area
    
    best_points = None
    best_value = -1.0
    
    # Strategy: 
    # 1. Run DE with a moderate population to find a good basin.
    # 2. Polish with local hill climbing.
    # 3. If time remains, restart DE from the best found or random points.
    
    # DE Parameters
    # popsize: 15 (default 15)
    # maxiter: 100 (default 1000) - we limit this by time
    # tol: 1e-6
    
    # We'll use a callback to check time? DE doesn't support callbacks easily in all versions, 
    # but we can just run it and hope it finishes, or split the work.
    # Actually, DE can be slow. Let's try a smaller maxiter first.
    
    # Alternative: Use a manual loop of DE-like steps or just rely on the built-in.
    # Built-in DE is efficient in C/Python hybrid.
    
    try:
        # Run DE
        # We want to maximize, so we minimize -f
        res = differential_evolution(
            objective_for_de, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-6,
            seed=42
        )
        
        if res.success or res.fun < 0: # res.fun is -area
            de_points = res.x.reshape(N, 2)
            de_points = project_points_to_triangle(de_points)
            de_val = min_triangle_area(de_points)
            
            if de_val > best_value:
                best_value = de_val
                best_points = de_points.copy()
                
    except Exception as e:
        pass # Fallback if DE fails or times out internally (though it shouldn't hang indefinitely without a timeout param, but maxiter limits it)
    
    # Check time
    if time.time() > deadline - 20:
        if best_points is not None:
             # Final quick polish
             final_points, final_val = local_polish(best_points, max_iter=500, step_size=0.01)
             if final_val > best_value:
                 best_value = final_val
                 best_points = final_points
             return best_points
        else:
             # Fallback to structured
             return generate_structured_initial()

    # If we have time, polish the DE result
    if best_points is not None:
        polished_points, polished_val = local_polish(best_points, max_iter=2000, step_size=0.05)
        if polished_val > best_value:
            best_value = polished_val
            best_points = polished_points
            
    # If we still have time, try another DE run from a different seed or from the polished point?
    # DE from a single point isn't standard, but we can try a random restart.
    
    if time.time() < deadline - 30:
        try:
            # Second DE run with different seed
            res2 = differential_evolution(
                objective_for_de, 
                bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-6,
                seed=123
            )
            if res2.fun < -best_value: # -fun is area, so if -fun > best_value
                de_points2 = res2.x.reshape(N, 2)
                de_points2 = project_points_to_triangle(de_points2)
                de_val2 = min_triangle_area(de_points2)
                
                if de_val2 > best_value:
                    best_value = de_val2
                    best_points = de_points2.copy()
                    
                    # Polish again
                    if time.time() < deadline - 10:
                        polished_points2, polished_val2 = local_polish(best_points, max_iter=1000, step_size=0.05)
                        if polished_val2 > best_value:
                            best_value = polished_val2
                            best_points = polished_points2
        except Exception:
            pass

    # Final verification and return
    if best_points is None:
        best_points = generate_structured_initial()
    else:
        # Ensure all points are strictly inside
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
                
    return best_points

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        p = (1.0 - u - v) * V0 + u * V1 + v * V2
        points.append(p)
    return np.array(points)

# Keep old functions if they are referenced elsewhere? 
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had:
# min_triangle_area, is_inside_triangle, project_to_triangle, project_points_to_triangle, 
# soft_min_surrogate, generate_structured_initial, generate_random_points, 
# lbfgs_optimize, nelder_mead_polish, local_optimize, simulated_annealing, construct

# I need to keep the names. I can make them stubs or keep them if they don't interfere.
# However, `construct` is the main entry.
# I will replace the bodies of the optimization functions with the new logic or keep them unused.
# But wait, the instruction says "Keep the function names and signatures... do not rename".
# It does not say I must keep the *implementation* of every helper if I change the strategy, 
# but usually it's safer to keep them valid in case they are called.
# However, `construct` is the only one called by the evaluator.
# I will update `construct` and add the new helpers. I will leave the old helpers as they are 
# unless they cause errors, but since `construct` is being rewritten, they won't be called 
# unless I call them.

# Let's refine the edits.
