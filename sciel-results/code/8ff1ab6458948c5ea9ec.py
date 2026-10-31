"""Heilbronn triangles: maximize the smallest triangle area among 11 points in a unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
# Equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, SQRT3 / 2.0])
TRI_AREA = 0.5 * SQRT3 / 2.0  # area of the triangle

# Deadline guard
DEADLINE = 120.0  # seconds
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(N,3) triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _smooth_min_triangle_area(points, beta=50.0):
    """
    Smooth surrogate for the minimum triangle area using log-sum-exp.
    We want to maximize the min area, so we minimize -min_area.
    Using: -min_i a_i ≈ (1/beta) * log(sum(exp(-beta * a_i)))
    """
    n = len(points)
    # Collect all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                areas.append(area)
    areas = np.array(areas)
    # Smooth min: min(a) ≈ (1/beta)*log(sum(exp(-beta*a)))
    # To avoid overflow, shift by max
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return (1.0 / beta) * log_sum


def _penalty_outside(points, margin=0.01):
    """Penalty for points outside the triangle (with margin)."""
    # Check each point is inside the triangle
    # Using barycentric coordinates
    penalty = 0.0
    for p in points:
        # Barycentric coordinates relative to A, B, C
        # p = a*A + b*B + c*C, a+b+c=1, a,b,c >= 0
        # Solve: p = b*(B-A) + c*(C-A)
        # B-A = (1,0), C-A = (0.5, sqrt(3)/2)
        # p_x = b + 0.5*c
        # p_y = (sqrt(3)/2)*c
        # c = p_y / (sqrt(3)/2) = 2*p_y/sqrt(3)
        # b = p_x - 0.5*c
        # a = 1 - b - c
        c_coord = 2.0 * p[1] / SQRT3
        b_coord = p[0] - 0.5 * c_coord
        a_coord = 1.0 - b_coord - c_coord
        # Penalize if any coordinate < margin
        if a_coord < margin:
            penalty += (margin - a_coord) ** 2
        if b_coord < margin:
            penalty += (margin - b_coord) ** 2
        if c_coord < margin:
            penalty += (margin - c_coord) ** 2
    return penalty


def _objective(x, beta=50.0):
    """Objective function to minimize: -smooth_min_area + penalty."""
    points = x.reshape(-1, 2)
    smooth_min = _smooth_min_triangle_area(points, beta=beta)
    penalty = _penalty_outside(points)
    # We want to maximize min area, so minimize -min_area
    # But smooth_min is an approximation of min_area (positive)
    # So we minimize: -smooth_min + penalty
    return -smooth_min + 100.0 * penalty


def _objective_gradient(x, beta=50.0):
    """Gradient of the objective function."""
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Gradient of smooth min triangle area
    # d/dx [ (1/beta)*log(sum(exp(-beta*a_i))) ]
    # = (1/beta) * sum_i [-beta * da_i/dx * exp(-beta*a_i)] / sum_j exp(-beta*a_j)
    # = -sum_i [da_i/dx * exp(-beta*a_i)] / sum_j exp(-beta*a_j)
    
    # First compute all areas and their gradients
    areas = []
    area_grads = []  # list of (2n,) gradients
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # area = 0.5 * |cross(p2-p1, p3-p1)|
                # For smoothness, assume orientation is consistent
                # area = 0.5 * ( (p2_x - p1_x)*(p3_y - p1_y) - (p2_y - p1_y)*(p3_x - p1_x) )
                cross_val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross_val)
                areas.append(area)
                
                # Gradient with respect to each point
                # d(area)/d(p1) = -0.5 * (cross(p2-p1, p3-p1) sign) ... let's use the signed version
                # For the signed area: A = 0.5 * cross(p2-p1, p3-p1)
                # dA/dp1 = -0.5 * cross(p2-p1, p3-p1) / (p2-p1) ... 
                # Actually, let's compute numerically or use the formula:
                # dA/dp1 = -0.5 * (cross(p2-p1, p3-p1) / |...|) -- this is getting complex
                # Let's use: A = 0.5 * |det|, and for the gradient, use the subgradient
                # For simplicity, use numerical gradient for the smooth part
                
                areas[-1] = area
    
    areas = np.array(areas)
    
    # Compute weights
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    # Now compute gradient numerically for simplicity (165 triangles, 22 variables)
    # Use central differences for the smooth objective
    eps = 1e-7
    grad = np.zeros_like(x)
    for idx in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (_objective(x_plus, beta) - _objective(x_minus, beta)) / (2 * eps)
    
    return grad


def _random_points_in_triangle(rng, n=N, margin=0.02):
    """Generate n random points uniformly inside the triangle with margin."""
    points = []
    for _ in range(n):
        while True:
            u = rng.uniform(margin, 1.0 - margin)
            v = rng.uniform(margin, 1.0 - margin)
            if u + v < 1.0 - 2 * margin:
                # Barycentric: a = 1-u-v, b = u, c = v
                # Point = a*A + b*B + c*C
                p = u * B + v * C
                points.append(p)
                break
    return np.array(points)


def _perturb_points(points, sigma, rng):
    """Perturb points with Gaussian noise, keeping them inside the triangle."""
    perturbed = points.copy()
    for i in range(len(perturbed)):
        noise = rng.normal(0, sigma, 2)
        new_p = perturbed[i] + noise
        # Check if inside triangle
        c_coord = 2.0 * new_p[1] / SQRT3
        b_coord = new_p[0] - 0.5 * c_coord
        a_coord = 1.0 - b_coord - c_coord
        if a_coord > 0.01 and b_coord > 0.01 and c_coord > 0.01:
            perturbed[i] = new_p
    return perturbed


def _optimize_from_start(points, beta=50.0, method='L-BFGS-B', maxiter=200):
    """Run optimization from a starting point."""
    x0 = points.flatten()
    
    bounds = [(0.0, 1.0), (0.0, SQRT3 / 2.0)] * N
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(beta,),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points


def _slsqp_polish(points, beta=50.0, maxiter=100):
    """Polish with SLSQP."""
    x0 = points.flatten()
    bounds = [(0.0, 1.0), (0.0, SQRT3 / 2.0)] * N
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(beta,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        if result.success:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points


def _local_search(points, n_iterations=50, step_size=0.05, rng=None):
    """Simple local search: try small random moves and keep improvements."""
    if rng is None:
        rng = np.random.default_rng()
    
    current = points.copy()
    current_score = _min_triangle_area(current)
    
    for _ in range(n_iterations):
        if _time_left() < 5.0:
            break
        # Try a random move
        idx = rng.integers(0, N)
        move = rng.normal(0, step_size, 2)
        new_points = current.copy()
        new_points[idx] += move
        
        # Check bounds
        if new_points[idx][0] < 0.01 or new_points[idx][0] > 0.99:
            continue
        if new_points[idx][1] < 0.01 or new_points[idx][1] > SQRT3 / 2 - 0.01:
            continue
        
        # Check inside triangle
        c_coord = 2.0 * new_points[idx][1] / SQRT3
        b_coord = new_points[idx][0] - 0.5 * c_coord
        a_coord = 1.0 - b_coord - c_coord
        if a_coord < 0.01 or b_coord < 0.01 or c_coord < 0.01:
            continue
        
        new_score = _min_triangle_area(new_points)
        if new_score > current_score:
            current = new_points
            current_score = new_score
    
    return current


def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Multiple random restarts with L-BFGS-B + SLSQP polish
    n_restarts = 15
    for r in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        # Generate diverse starting points
        if r == 0:
            # Symmetric-ish start: points near the vertices and center
            points = np.array([
                [0.15, 0.08],      # near A
                [0.85, 0.08],      # near B
                [0.5, 0.45],       # near C
                [0.3, 0.2],        # interior
                [0.7, 0.2],        # interior
                [0.5, 0.25],       # center-ish
                [0.25, 0.35],      # interior
                [0.75, 0.35],      # interior
                [0.4, 0.15],       # interior
                [0.6, 0.15],       # interior
                [0.5, 0.35],       # interior
            ])
        elif r < 5:
            # Perturbed symmetric start
            base = np.array([
                [0.15, 0.08],
                [0.85, 0.08],
                [0.5, 0.45],
                [0.3, 0.2],
                [0.7, 0.2],
                [0.5, 0.25],
                [0.25, 0.35],
                [0.75, 0.35],
                [0.4, 0.15],
                [0.6, 0.15],
                [0.5, 0.35],
            ])
            points = _perturb_points(base, sigma=0.05, rng=rng)
        else:
            # Random start
            points = _random_points_in_triangle(rng, margin=0.03)
        
        # Optimize with L-BFGS-B
        optimized = _optimize_from_start(points, beta=50.0, method='L-BFGS-B', maxiter=150)
        
        # Polish with SLSQP
        polished = _slsqp_polish(optimized, beta=50.0, maxiter=80)
        
        # Local search
        final = _local_search(polished, n_iterations=30, step_size=0.03, rng=rng)
        
        # Ensure points are inside the triangle
        for i in range(N):
            c_coord = 2.0 * final[i][1] / SQRT3
            b_coord = final[i][0] - 0.5 * c_coord
            a_coord = 1.0 - b_coord - c_coord
            if a_coord < 0.005:
                final[i] = [0.05, 0.03]
            elif b_coord < 0.005:
                final[i] = [0.95, 0.03]
            elif c_coord < 0.005:
                final[i] = [0.5, 0.4]
        
        score = _min_triangle_area(final)
        if score > best_score:
            best_score = score
            best_points = final.copy()
    
    # Strategy 2: If we have time, try more restarts with different betas
    if _time_left() > 15.0 and best_points is not None:
        for beta in [30.0, 80.0]:
            if _time_left() < 10.0:
                break
            for r in range(5):
                if _time_left() < 5.0:
                    break
                points = _random_points_in_triangle(rng, margin=0.03)
                optimized = _optimize_from_start(points, beta=beta, method='L-BFGS-B', maxiter=100)
                polished = _slsqp_polish(optimized, beta=beta, maxiter=60)
                final = _local_search(polished, n_iterations=20, step_size=0.03, rng=rng)
                
                for i in range(N):
                    c_coord = 2.0 * final[i][1] / SQRT3
                    b_coord = final[i][0] - 0.5 * c_coord
                    a_coord = 1.0 - b_coord - c_coord
                    if a_coord < 0.005:
                        final[i] = [0.05, 0.03]
                    elif b_coord < 0.005:
                        final[i] = [0.95, 0.03]
                    elif c_coord < 0.005:
                        final[i] = [0.5, 0.4]
                
                score = _min_triangle_area(final)
                if score > best_score:
                    best_score = score
                    best_points = final.copy()
    
    if best_points is None:
        # Fallback: return random points
        best_points = _random_points_in_triangle(rng, margin=0.05)
    
    return best_points


# Test at module level to ensure it works
if __name__ == "__main__":
    points = construct()
    print("Points:")
    for p in points:
        print(f"  ({p[0]:.4f}, {p[1]:.4f})")
    print(f"Min triangle area: {_min_triangle_area(points):.6f}")
    print(f"Normalized by triangle area: {_min_triangle_area(points) / TRI_AREA:.6f}")
