"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _bounding_box(circles):
    """Compute bounding box of circles: (xmin, ymin, xmax, ymax)."""
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    xmin = np.min(cx - r)
    ymin = np.min(cy - r)
    xmax = np.max(cx + r)
    ymax = np.max(cy + r)
    return xmin, ymin, xmax, ymax

def _perimeter(circles):
    """Perimeter of bounding box."""
    xmin, ymin, xmax, ymax = _bounding_box(circles)
    return 2.0 * ((xmax - xmin) + (ymax - ymin))

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _is_valid(circles, tol=1e-9):
    """Check if circles are disjoint and perimeter <= 4."""
    n = len(circles)
    if n == 0:
        return False
    if _perimeter(circles) > 4.0 + tol:
        return False
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True

def _normalize_circles(circles):
    """Translate circles so bounding box starts at (0,0) and scale so perimeter = 4."""
    xmin, ymin, xmax, ymax = _bounding_box(circles)
    width = xmax - xmin
    height = ymax - ymin
    perim = 2.0 * (width + height)
    if perim <= 0:
        return circles
    scale = 4.0 / perim
    # Translate so min is at 0
    cx = circles[:, 0] - xmin
    cy = circles[:, 1] - ymin
    r = circles[:, 2]
    # Scale
    cx = cx * scale
    cy = cy * scale
    r = r * scale
    return np.column_stack([cx, cy, r])

def _objective(x, N_circles):
    """
    Objective: minimize negative sum of radii.
    x: [cx_0, cy_0, r_0, cx_1, cy_1, r_1, ...]
    We'll work in a unit square [0,1]x[0,1] and then scale.
    Actually, let's work directly with the constraint that bounding box perimeter = 4.
    
    Better approach: parameterize positions in [0,1]x[0,1] and radii, 
    then the bounding box will have some perimeter P. We want P <= 4.
    
    Alternative: work in a fixed aspect ratio rectangle. Let's use W x H where 2(W+H)=4, so W+H=2.
    We can vary the aspect ratio.
    
    Let me use a different approach: place circles in a rectangle of width W and height H 
    where W + H = 2 (so perimeter = 4). The aspect ratio W/H is a variable.
    
    Actually, let's simplify: work in normalized coordinates [0,1]x[0,1].
    The bounding box of the circles in this space will have some width w and height h.
    The actual perimeter is 2*(w*scale + h*scale) = 2*scale*(w+h) where scale is the 
    scaling factor. We want 2*scale*(w+h) = 4, so scale = 2/(w+h).
    The actual radii are r_i * scale.
    Sum of actual radii = scale * sum(r_i) = 2*sum(r_i)/(w+h).
    
    So we want to maximize 2*sum(r_i)/(w+h) subject to:
    - Circles are in [0,1]x[0,1] (actually, their centers can be anywhere, but the bounding box is what matters)
    - Circles are disjoint: dist(i,j) >= r_i + r_j
    - Circles are within the bounding box: for each circle, cx - r >= 0, cx + r <= w, etc.
    
    Hmm, this is getting complicated. Let me use a simpler parameterization.
    
    Let's define:
    - For each circle i: (x_i, y_i, r_i) in some coordinate system
    - The bounding box has width W = max(x_i + r_i) - min(x_i - r_i)
    - The bounding box has height H = max(y_i + r_i) - min(y_i - r_i)
    - We require 2*(W + H) <= 4, i.e., W + H <= 2
    - We want to maximize sum(r_i)
    
    This is a non-linear optimization problem with non-smooth constraints (the bounding box depends on max/min).
    
    Let me use scipy.optimize.minimize with a penalty method.
    """
    # x has 3*N_circles elements: [x0, y0, r0, x1, y1, r1, ...]
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Compute bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    W = xmax - xmin
    H = ymax - ymin
    
    # Objective: maximize sum(rs) subject to W + H <= 2
    # Use penalty: minimize -sum(rs) + penalty * max(0, W + H - 2)^2
    sum_r = np.sum(rs)
    perim_violation = max(0.0, W + H - 2.0)
    penalty = 100.0 * perim_violation ** 2
    
    # Also add a small penalty for circles going outside [0, 2]x[0, 2] to keep them bounded
    # Actually, let's add a penalty for radii being too small or negative
    r_penalty = 0.0
    for i in range(N_circles):
        if rs[i] < 1e-6:
            r_penalty += (1e-6 - rs[i]) ** 2
    
    return -sum_r + penalty + r_penalty

def _gradient(x, N_circles):
    """Numerical gradient."""
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_objective(x_plus, N_circles) - _objective(x_minus, N_circles)) / (2 * eps)
    return grad

def _local_optimize(circles, max_iter=500):
    """
    Take initial circle configuration and optimize using scipy.
    circles: (N, 3) array with [x, y, r]
    Returns optimized circles and the objective value.
    """
    N_circles = len(circles)
    x0 = circles.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        _objective,
        x0,
        args=(N_circles,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = result.x
    xs = x_opt[0::3]
    ys = x_opt[1::3]
    rs = x_opt[2::3]
    
    # Ensure radii are positive
    rs = np.maximum(rs, 1e-6)
    
    optimized = np.column_stack([xs, ys, rs])
    return optimized, result.fun

def _random_init(N_circles, rng):
    """Generate a random initial configuration."""
    # Place circles randomly in [0, 2] x [0, 2]
    xs = rng.uniform(0.1, 1.9, size=N_circles)
    ys = rng.uniform(0.1, 1.9, size=N_circles)
    # Random radii
    rs = rng.uniform(0.05, 0.3, size=N_circles)
    return np.column_stack([xs, ys, rs])

def _grid_init(N_circles, rng):
    """Generate a grid-like initial configuration."""
    side = int(np.ceil(np.sqrt(N_circles)))
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N_circles:
                break
            x = (i + 0.5) / side * 2.0
            y = (j + 0.5) / side * 2.0
            r = 0.9 / side
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    # Add some random perturbation
    xs = np.array(xs) + rng.uniform(-0.05, 0.05, size=N_circles)
    ys = np.array(ys) + rng.uniform(-0.05, 0.05, size=N_circles)
    rs = np.array(rs)
    return np.column_stack([xs, ys, rs])

def _evaluate(circles):
    """Evaluate a configuration: return (sum_radii, is_valid)."""
    if not _is_valid(circles):
        return 0.0, False
    return _sum_radii(circles), True

def construct():
    """
    Construct 21 circles with disjoint interiors in a rectangle of perimeter <= 4,
    maximizing the sum of radii.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_circles = None
    best_value = 0.0
    
    # Try multiple initializations
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate initial configuration
        if attempt == 0:
            circles = _grid_init(N, rng)
        elif attempt == 1:
            circles = _random_init(N, rng)
        else:
            # Mix of random and grid
            if rng.random() < 0.5:
                circles = _grid_init(N, rng)
            else:
                circles = _random_init(N, rng)
        
        # Local optimization
        try:
            optimized, obj_val = _local_optimize(circles, max_iter=300)
        except Exception:
            continue
        
        # Check validity
        value, valid = _evaluate(optimized)
        if valid and value > best_value:
            best_value = value
            best_circles = optimized.copy()
        
        # Also try with different aspect ratios by scaling
        # Actually, the optimizer should handle this, but let's try a few more variants
        
        # Perturb the best solution and re-optimize
        if best_circles is not None and time.time() < deadline - 20.0:
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                perturbed = best_circles + rng.normal(0, 0.05, size=(N, 3))
                # Ensure radii stay positive
                perturbed[:, 2] = np.maximum(perturbed[:, 2], 0.01)
                try:
                    opt2, obj2 = _local_optimize(perturbed, max_iter=200)
                    val2, valid2 = _evaluate(opt2)
                    if valid2 and val2 > best_value:
                        best_value = val2
                        best_circles = opt2.copy()
                except Exception:
                    continue
    
    # Final polish: run more iterations on the best solution
    if best_circles is not None and time.time() < deadline - 10.0:
        try:
            polished, obj_final = _local_optimize(best_circles, max_iter=1000)
            val_final, valid_final = _evaluate(polished)
            if valid_final and val_final > best_value:
                best_value = val_final
                best_circles = polished.copy()
        except Exception:
            pass
    
    # If we still don't have a valid solution, fall back to the grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
