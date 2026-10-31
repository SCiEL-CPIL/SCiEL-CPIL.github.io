"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
# Unit area equilateral triangle: vertices (0,0), (1,0), (0.5, sqrt(3)/2)
# Area = sqrt(3)/4
TRI_AREA = SQRT3 / 4.0
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

DEADLINE_FRACTION = 0.85  # safety margin


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _soft_min_triangle_area(pts, beta):
    """Soft-min of all triangle areas using log-sum-exp (negative for min)."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas[i, j, k] = area
    # Soft min: -log(sum(exp(-beta * area))) / beta
    # Use log-sum-exp trick for stability
    m = np.min(areas[areas > 0]) if np.any(areas > 0) else 0.0
    if m < 1e-15:
        m = 1e-15
    exp_neg = np.exp(-beta * (areas - m))
    # Only consider valid triangles (i<j<k)
    valid = np.zeros_like(areas, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                valid[i, j, k] = True
    s = np.sum(exp_neg[valid])
    if s < 1e-300:
        return m
    soft_min = m - np.log(s) / beta
    return max(soft_min, 0.0)


def _negative_soft_min(pts, beta):
    """Objective for minimization: negative soft-min area."""
    return -_soft_min_triangle_area(pts, beta)


def _in_triangle(pts, margin=0.0):
    """Check if all points are inside or on the triangle with a margin."""
    # Barycentric coordinate check
    # For point p, barycentric coords w.r.t. V0, V1, V2
    # p = w0*V0 + w1*V1 + w2*V2, w0+w1+w2=1, wi >= -margin
    for p in pts:
        # Solve for barycentric coords
        # x = w1*1 + w2*0.5
        # y = w2*sqrt(3)/2
        # w0 = 1 - w1 - w2
        w2 = p[1] / (SQRT3 / 2.0)
        w1 = p[0] - 0.5 * w2
        w0 = 1.0 - w1 - w2
        if w0 < -margin or w1 < -margin or w2 < -margin:
            return False
    return True


def _project_to_triangle(pts):
    """Project points that are outside the triangle back inside."""
    pts = pts.copy()
    for idx in range(len(pts)):
        p = pts[idx]
        w2 = p[1] / (SQRT3 / 2.0)
        w1 = p[0] - 0.5 * w2
        w0 = 1.0 - w1 - w2
        if w0 < 0 or w1 < 0 or w2 < 0:
            # Clamp barycentric coords
            w0 = max(w0, 0.0)
            w1 = max(w1, 0.0)
            w2 = max(w2, 0.0)
            s = w0 + w1 + w2
            if s < 1e-15:
                w0, w1, w2 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
            else:
                w0 /= s
                w1 /= s
                w2 /= s
            pts[idx] = w1 * V1 + w2 * V2  # w0*V0 = 0
    return pts


def _initial_placement():
    """Generate a structured initial placement using a grid in barycentric coords."""
    rng = np.random.default_rng(42)
    # Use a structured grid in barycentric coordinates
    # 11 points: try a few patterns
    # Pattern 1: corners + edges + center
    points = []
    # Corners (slightly inset)
    eps = 0.05
    points.append([eps, eps])
    points.append([1 - eps, eps])
    points.append([0.5 - 0.5 * eps, (SQRT3 / 2) * (1 - eps)])
    # Edge midpoints
    points.append([0.5, eps])
    points.append([0.5 + 0.5 * (1 - eps), (SQRT3 / 2) * (1 - eps) / 2])
    points.append([0.5 * (1 - eps), (SQRT3 / 2) * (1 - eps) / 2])
    # Interior points
    # Centroid
    points.append([0.5, SQRT3 / 6.0])
    # Additional interior points
    points.append([0.3, 0.15])
    points.append([0.7, 0.15])
    points.append([0.5, 0.3])
    points.append([0.35, 0.35])
    
    points = np.array(points)
    # Add small random perturbation
    points += rng.normal(0, 0.01, size=points.shape)
    points = _project_to_triangle(points)
    return points


def _sa_phase(pts, time_budget, rng_seed=0):
    """Simulated annealing phase with adaptive step size."""
    rng = np.random.default_rng(rng_seed)
    start_time = time.time()
    best_pts = pts.copy()
    best_val = _min_triangle_area(pts)
    current_pts = pts.copy()
    current_val = best_val
    
    T = 0.1  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    
    step_size = 0.1
    n_iter = 0
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        n_iter += 1
        # Adaptive step size
        if no_improve > 500:
            step_size *= 0.9
            no_improve = 0
        elif no_improve > 100:
            step_size *= 0.95
            no_improve = 0
        if step_size < 1e-6:
            step_size = 1e-6
        
        # Random perturbation
        new_pts = current_pts + rng.normal(0, step_size, size=current_pts.shape)
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _min_triangle_area(new_pts)
        
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = current_pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_val


def _gradient_polish(pts, time_budget, beta_start=100.0, beta_end=1000.0):
    """Gradient-based polishing using scipy minimize with smooth objective."""
    start_time = time.time()
    best_pts = pts.copy()
    best_val = _min_triangle_area(pts)
    
    # Flatten to 1D
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 2)
        p = _project_to_triangle(p)
        return -_soft_min_triangle_area(p, beta_start)
    
    def gradient(x):
        # Numerical gradient
        p = x.reshape(-1, 2)
        p = _project_to_triangle(p)
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            fp = objective(xp)
            xm = x.copy()
            xm[i] -= eps
            fm = objective(xm)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Run minimize with time limit
    best_x = x0.copy()
    best_obj = objective(x0)
    
    n_iter = 0
    while time.time() - start_time < time_budget:
        n_iter += 1
        try:
            res = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
                p = _project_to_triangle(best_x.reshape(-1, 2))
                val = _min_triangle_area(p)
                if val > best_val:
                    best_val = val
                    best_pts = p.copy()
            if not res.success:
                break
        except Exception:
            break
    
    return best_pts, best_val


def construct():
    """Construct 11 points in the unit equilateral triangle maximizing min triangle area."""
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    # Phase 1: Initial structured placement
    pts = _initial_placement()
    best_pts = pts.copy()
    best_val = _min_triangle_area(pts)
    
    # Phase 2: SA global exploration
    remaining = deadline - time.time()
    if remaining > 10:
        sa_budget = min(remaining * 0.4, 40.0)
        pts, val = _sa_phase(pts, sa_budget, rng_seed=42)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Phase 3: Gradient polish
    remaining = deadline - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.3, 20.0)
        pts, val = _gradient_polish(best_pts, polish_budget, beta_start=200.0)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Phase 4: SA re-explore from best
    remaining = deadline - time.time()
    if remaining > 10:
        sa_budget = min(remaining * 0.4, 30.0)
        pts, val = _sa_phase(best_pts, sa_budget, rng_seed=123)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.5, 15.0)
        pts, val = _gradient_polish(best_pts, polish_budget, beta_start=500.0)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Ensure all points are inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    return np.array(best_pts)
