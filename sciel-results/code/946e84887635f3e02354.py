"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

DEADLINE = 150.0
TIME_LIMIT = DEADLINE - 5.0  # safety margin

def _perimeter(w, h):
    return 2.0 * (w + h)

def _validity(circles, tol=1e-8):
    """Check if circles are valid: inside bounding box, non-overlapping, radii positive."""
    n = len(circles)
    if n == 0:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    # Bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    w = xmax - xmin
    h = ymax - ymin
    perim = 2.0 * (w + h)
    if perim > 4.0 + tol:
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _objective(circles):
    """Objective: maximize sum of radii. We minimize negative sum."""
    return -np.sum(circles[:, 2])

def _build_constraints(n):
    """Build constraint functions for SLSQP."""
    def bounds():
        # Each circle: x, y, r >= 0
        # r <= some upper bound
        b = []
        for i in range(n):
            b.append((0, 1.0))  # x
            b.append((0, 1.0))  # y
            b.append((0, 1.0))  # r
        return b
    
    def constraints():
        cons = []
        # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # SLSQP needs: g(x) >= 0
        for i in range(n):
            for j in range(i + 1, n):
                def overlap_con(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    return dx*dx + dy*dy - (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': overlap_con})
        
        # Perimeter constraint: 2*(w+h) <= 4
        # w = max(x+r) - min(x-r), h = max(y+r) - min(y-r)
        # This is non-smooth, so we use a soft formulation
        # Actually, for SLSQP, let's use the fact that we can bound the box
        # Instead, let's constrain that all circles fit in [0, W] x [0, H] where 2(W+H) <= 4
        # But W and H depend on the circles. 
        # Alternative: constrain the bounding box explicitly.
        # Let's add constraints that max(x+r) - min(x-r) + max(y+r) - min(y-r) <= 2
        # This is still non-smooth.
        
        # Better approach: use auxiliary variables for the bounding box
        # But that changes the variable space. Let's stick with a simpler approach:
        # Constrain that the sum of extents is bounded.
        # For now, let's just rely on the initial feasible region and local search.
        # We'll add a perimeter constraint using a smooth approximation.
        
        def perim_con(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            xmin = np.min(xs - rs)
            xmax = np.max(xs + rs)
            ymin = np.min(ys - rs)
            ymax = np.max(ys + rs)
            w = xmax - xmin
            h = ymax - ymin
            return 2.0 - (w + h)  # We want 2*(w+h) <= 4 => w+h <= 2
        cons.append({'type': 'ineq', 'fun': perim_con})
        
        return cons
    
    return bounds, cons

def _slsqp_optimize(circles, max_iter=100):
    """Run SLSQP optimization to maximize sum of radii."""
    n = len(circles)
    x0 = circles.flatten()
    bounds, cons = _build_constraints(n)
    
    result = minimize(
        _objective,
        x0,
        method='SLSQP',
        bounds=bounds(),
        constraints=cons(),
        options={'maxiter': max_iter, 'ftol': 1e-8}
    )
    
    if result.success or result.fun < _objective(x0):
        return result.x.reshape(-1, 3)
    return circles

def _local_search(circles, steps=5000, lr=0.01):
    """Simple local search with random perturbations."""
    n = len(circles)
    best = circles.copy()
    best_obj = -_objective(best)
    
    for _ in range(steps):
        # Pick a random circle and perturb it
        i = np.random.randint(n)
        new_circle = best.copy()
        dx = np.random.normal(0, lr)
        dy = np.random.normal(0, lr)
        dr = np.random.normal(0, lr * 0.5)
        
        new_circle[i, 0] += dx
        new_circle[i, 1] += dy
        new_circle[i, 2] += dr
        
        if new_circle[i, 2] <= 0:
            continue
        
        if _validity(new_circle):
            obj = -_objective(new_circle)
            if obj > best_obj:
                best = new_circle
                best_obj = obj
    
    return best

def _expand_circles(circles, max_iter=50):
    """Try to expand all circles slightly while maintaining validity."""
    n = len(circles)
    best = circles.copy()
    best_obj = -_objective(best)
    
    for _ in range(max_iter):
        # Try expanding each circle by a small amount
        improved = False
        for i in range(n):
            new_circle = best.copy()
            new_circle[i, 2] *= 1.001  # Expand by 0.1%
            
            if _validity(new_circle):
                obj = -_objective(new_circle)
                if obj > best_obj:
                    best = new_circle
                    best_obj = obj
                    improved = True
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Initial configuration: grid-based
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    circles = np.array(cells[:N])
    
    # Normalize to fit in perimeter-4 rectangle
    # Current bounding box: [0, 1] x [0, 1], perimeter = 4. Good.
    
    best_circles = circles.copy()
    best_score = -_objective(best_circles)
    
    # Step 1: SLSQP optimization
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 30:
        circles = _slsqp_optimize(circles, max_iter=50)
        if _validity(circles):
            score = -_objective(circles)
            if score > best_score:
                best_circles = circles.copy()
                best_score = score
    
    # Step 2: Local search
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        circles = _local_search(best_circles, steps=3000, lr=0.005)
        if _validity(circles):
            score = -_objective(circles)
            if score > best_score:
                best_circles = circles.copy()
                best_score = score
    
    # Step 3: Expansion
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        circles = _expand_circles(best_circles, max_iter=30)
        if _validity(circles):
            score = -_objective(circles)
            if score > best_score:
                best_circles = circles.copy()
                best_score = score
    
    # Step 4: Final SLSQP polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        circles = _slsqp_optimize(best_circles, max_iter=30)
        if _validity(circles):
            score = -_objective(circles)
            if score > best_score:
                best_circles = circles.copy()
                best_score = score
    
    # Step 5: Final expansion
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 2:
        circles = _expand_circles(best_circles, max_iter=20)
        if _validity(circles):
            score = -_objective(circles)
            if score > best_score:
                best_circles = circles.copy()
                best_score = score
    
    # Ensure validity
    if not _validity(best_circles):
        # Fallback to initial
        best_circles = circles.copy()
    
    return best_circles
# EVOLVE-BLOCK-END
