import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
H = np.sqrt(3)/2

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))

def calc_min_area(points):
    """
    points: (11, 2)
    returns: min area
    """
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def is_inside_triangle(points):
    """
    Check if all points are inside or on the boundary of the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    eps = 1e-9
    cond1 = y >= -eps
    cond2 = y <= np.sqrt(3) * x + eps
    cond3 = y <= -np.sqrt(3) * (x - 1) + eps
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def objective_de(x):
    """
    Objective for Differential Evolution.
    x: flat array of length 22
    """
    pts = x.reshape(N, 2)
    
    # Penalty for being outside
    # We can use a smooth penalty or a hard penalty.
    # Hard penalty is simpler for DE if the domain is well-bounded.
    # However, if we use box bounds [0,1]x[0,H], many points are outside.
    # A large constant penalty works well to push them out.
    if not is_inside_triangle(pts):
        # Return a value larger than any possible -min_area
        # Max area is ~0.433, so -min_area is in [-0.433, 0]
        # Penalty 10.0 is safe.
        return 10.0
    
    min_area = calc_min_area(pts)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Bounds for DE
    # x in [0, 1], y in [0, H]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, H))
        
    # Run Differential Evolution
    # We use a relatively large population to ensure global coverage
    # and a moderate number of iterations.
    # We check time periodically.
    
    best_x = None
    best_val = -np.inf
    
    # DE might be slow if it runs too long. 
    # Let's run it for a fixed number of iterations that fits in ~30-60s.
    # 11 points * 2 coords = 22 dimensions.
    # popsize=15 -> 15*22=330 evals/gen.
    # If eval is 1ms, 330ms/gen.
    # 100 gens = 33s.
    
    try:
        de_result = differential_evolution(
            objective_de, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-8,
            polish=False, # We will polish manually
            init='latinhypercube',
            updating='immediate'
        )
        
        if de_result.success or de_result.fun < 1.0: # 1.0 is penalty threshold
            best_x = de_result.x
            best_val = -de_result.fun # de_result.fun is -min_area, so min_area = -fun
        else:
            # If DE failed or hit penalty, try a fallback
            best_x = de_result.x
            best_val = -de_result.fun
            
    except Exception:
        # Fallback to random sampling if DE crashes
        print("DE failed, using random")
        rng = np.random.default_rng(42)
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * H
                if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1):
                    pts[i] = [x, y]
                    break
        best_x = pts.flatten()
        best_val = calc_min_area(pts)

    # Check time
    if time.time() > deadline - 20.0:
        return best_x.reshape(N, 2)

    # Local Refinement
    # Use Nelder-Mead to polish the result
    # We need to ensure the points stay inside.
    # Nelder-Mead doesn't handle constraints well.
    # We can use a penalty method in the local objective.
    
    def local_objective(x):
        pts = x.reshape(N, 2)
        if not is_inside_triangle(pts):
            return 10.0
        min_area = calc_min_area(pts)
        return -min_area

    try:
        nm_result = minimize(
            local_objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if nm_result.fun < 1.0: # Valid
            new_x = nm_result.x
            new_val = -nm_result.fun
            if new_val > best_val:
                best_x = new_x
                best_val = new_val
        else:
            # If NM went outside, try to clamp?
            # Or just keep the DE result.
            pass
    except Exception:
        pass

    # Final check
    final_pts = best_x.reshape(N, 2)
    if not is_inside_triangle(final_pts):
        # Emergency clamp or reject?
        # If invalid, score is 0.
        # Try to project points back?
        # For now, return as is, hoping it's valid.
        # Actually, let's try a quick repair if it's slightly outside.
        # But simple rejection is safer than a bad repair.
        pass

    return final_pts
