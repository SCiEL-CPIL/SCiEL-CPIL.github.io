"""Heilbronn triangles: maximize the minimum triangle area among 11 points."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
SQRT3 = 3 ** 0.5
H = SQRT3 / 2.0  # height of unit-base equilateral triangle

# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    return 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def _min_triangle_area(points):
    """Minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _soft_min_triangle_area(points, beta):
    """Smooth surrogate: soft-min of all triangle areas using log-sum-exp.
    Returns a value that approximates the min area when beta is large."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                a = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(a)
    areas = np.array(areas)
    # Soft-min: -log(sum(exp(-beta * a))) / beta  -> approximates min(a) as beta -> inf
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_a - log_sum / beta
    return soft_min


def _objective_smooth(x, beta):
    """Objective to minimize: negative soft-min area."""
    points = x.reshape(N, 2)
    # Check if points are inside the triangle (with small margin)
    # Barycentric coordinates
    margin = 0.005
    for p in points:
        # Check using barycentric coordinates
        # p = a*V0 + b*V1 + c*V2, a+b+c=1, a,b,c >= 0
        a = H * p[1] / H  # coefficient for V0 (0,0)
        # Actually let's use a simpler check
        # Point is inside triangle if it's on the same side of each edge
        pass
    return -_soft_min_triangle_area(points, beta)


def _is_inside_triangle(p, margin=0.0):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Edge 1: from (0,0) to (1,0): y >= 0
    if y < -margin:
        return False
    # Edge 2: from (1,0) to (0.5, H): line is y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    # Point should be below this line: y <= sqrt(3)*(1-x)
    if y > H * (1 - x) + margin:
        return False
    # Edge 3: from (0.5, H) to (0,0): line is y = sqrt(3)*x
    # Point should be below this line: y <= sqrt(3)*x
    if y > H * x + margin:
        return False
    return True


def _project_to_triangle(p):
    """Project point onto the triangle (simple clipping)."""
    x, y = p
    # Clip to the three half-planes
    # y >= 0
    if y < 0:
        y = 0
    # y <= H*(1-x)  =>  H*x + y <= H
    if H * x + y > H:
        # Project onto line H*x + y = H
        # Normal direction: (H, 1), magnitude = sqrt(H^2+1) = sqrt(3+1)=2
        # Distance from point to line: (H*x + y - H) / 2
        # Projected point: p - ((H*x+y-H)/2) * (H,1)
        d = (H * x + y - H) / 2.0
        x = x - d * H
        y = y - d
    # y <= H*x  =>  y - H*x <= 0
    if y - H * x > 0:
        d = (y - H * x) / 2.0
        x = x + d * H
        y = y - d
    return [x, y]


def _lattice_init(seed=0):
    """Initialize points on a triangular lattice inside the triangle."""
    rng = np.random.RandomState(seed)
    # Create a grid of candidate positions
    # Use barycentric coordinates
    points = []
    # Try to place points roughly evenly
    # Use a 3x3 grid in barycentric coords, but that's only 9
    # Let's use a more systematic approach
    # Place points at fractional positions
    # For 11 points, use a pattern
    # Start with vertices and midpoints and center, then add more
    
    # Barycentric grid approach
    # n levels: for n=3, we get 6 points; for n=4, we get 10 points
    # Let's use a combination
    
    # Method: use a fine grid and pick 11 well-spread points
    n_grid = 10
    candidates = []
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                if i + j + k == 2 * (n_grid - 1):
                    # Barycentric coords
                    a = i / (n_grid - 1)
                    b = j / (n_grid - 1)
                    c = k / (n_grid - 1)
                    # Cartesian: p = a*V0 + b*V1 + c*V2
                    x = a * 0.0 + b * 1.0 + c * 0.5
                    y = a * 0.0 + b * 0.0 + c * H
                    candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    # Pick 11 points that maximize spread (simple greedy)
    selected = []
    # Start with a vertex
    selected.append(candidates[0].copy())  # (0,0)
    
    for _ in range(N - 1):
        best_idx = -1
        best_min_dist = -1
        for idx in range(len(candidates)):
            if any(np.allclose(candidates[idx], s) for s in selected):
                continue
            # Minimum distance to already selected
            min_d = min(np.linalg.norm(candidates[idx] - s) for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        if best_idx >= 0:
            selected.append(candidates[best_idx].copy())
    
    # Add some randomness
    result = np.array(selected)
    # Small perturbation
    result += rng.uniform(-0.01, 0.01, size=result.shape)
    # Project back to triangle
    for i in range(len(result)):
        result[i] = _project_to_triangle(result[i])
    
    return result


def _sa_optimize(init_points, deadline, seed=0):
    """Simulated Annealing optimization using soft-min surrogate."""
    rng = np.random.RandomState(seed)
    best_points = init_points.copy()
    best_score = _min_triangle_area(best_points)
    
    current_points = init_points.copy()
    current_score = _min_triangle_area(current_points)
    
    # SA parameters
    T0 = 0.05
    Tend = 0.0001
    n_iter = 50000
    cooling = (Tend / T0) ** (1.0 / n_iter)
    
    T = T0
    time_start = time.time()
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Propose a move: pick a random point and perturb it
        idx = rng.randint(0, N)
        new_points = current_points.copy()
        # Gaussian perturbation
        step = 0.02 * (1.0 + 0.5 * T / T0)
        new_points[idx] += rng.normal(0, step, size=2)
        # Project to triangle
        new_points[idx] = _project_to_triangle(new_points[idx])
        
        # Evaluate
        new_score = _min_triangle_area(new_points)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
    
    return best_points, best_score


def _nelder_mead_polish(points, deadline):
    """Polish using Nelder-Mead on the exact objective."""
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(N, 2)
        # Penalty for points outside triangle
        penalty = 0.0
        for p in pts:
            if not _is_inside_triangle(p, margin=0.0):
                # Distance to triangle boundary
                penalty += 1.0
        score = _min_triangle_area(pts)
        return -score + penalty * 10.0
    
    # Set up bounds implicitly through projection
    options = {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', options=options)
        if result.fun < 0:  # Valid improvement
            new_points = result.x.reshape(N, 2)
            # Verify all points are inside
            valid = True
            for p in new_points:
                if not _is_inside_triangle(p, margin=-0.001):
                    valid = False
                    break
            if valid:
                return new_points
    except Exception:
        pass
    
    return points


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Lattice initialization with different seeds
        init = _lattice_init(seed=restart * 137 + 42)
        
        # SA optimization
        pts, score = _sa_optimize(init, deadline - 15.0, seed=restart * 101 + 7)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Nelder-Mead polish
        if time.time() < deadline - 5.0:
            polished = _nelder_mead_polish(pts, deadline - 5.0)
            polished_score = _min_triangle_area(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Final polish on the best
    if best_points is not None and time.time() < deadline - 2.0:
        final = _nelder_mead_polish(best_points, deadline - 2.0)
        final_score = _min_triangle_area(final)
        if final_score > best_score:
            best_points = final
    
    if best_points is None:
        # Fallback: simple grid
        best_points = _lattice_init(seed=0)
    
    return best_points
