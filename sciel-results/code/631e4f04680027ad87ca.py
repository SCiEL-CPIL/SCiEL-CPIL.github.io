"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _remaining():
    return DEADLINE - time.time()


def _max_radius_for_circle(x, y, others):
    """Max radius for a circle at (x,y) given other circles (array of (x,y,r))."""
    r = min(x, y, 1.0 - x, 1.0 - y)
    if others is not None and len(others) > 0:
        dists = np.sqrt((x - others[:, 0]) ** 2 + (y - others[:, 1]) ** 2)
        r = min(r, np.min(dists - others[:, 2]))
    return max(r, 0.0)


def _compute_score(circles):
    """Compute sum of radii if all circles are valid (disjoint, inside square)."""
    n = len(circles)
    if n == 0:
        return 0.0
    # Check bounds
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -1e-9 or y - r < -1e-9 or x + r > 1.0 + 1e-9 or y + r > 1.0 + 1e-9:
            return -np.inf
    # Check disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if d < ri + rj - 1e-9:
                return -np.inf
    return np.sum(circles[:, 2])


def _penalty(circles, penalty_weight=100.0):
    """Penalty for constraint violations."""
    n = len(circles)
    pen = 0.0
    # Bounds penalty
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            pen += penalty_weight * (1.0 - r) ** 2
        # Distance to walls
        d_left = x - r
        d_right = 1.0 - x - r
        d_bottom = y - r
        d_top = 1.0 - y - r
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < 0:
                pen += penalty_weight * d * d
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            overlap = ri + rj - d
            if overlap > 0:
                pen += penalty_weight * overlap * overlap
    return pen


def _objective(params, penalty_weight=100.0):
    """Objective: negative sum of radii + penalty."""
    circles = params.reshape(-1, 3)
    radii = circles[:, 2]
    # Ensure radii are positive
    radii = np.maximum(radii, 0.0)
    neg_sum = -np.sum(radii)
    pen = _penalty(circles, penalty_weight)
    return neg_sum + pen


def _delaunay_pairs(circles):
    """Get Delaunay edges as index pairs for constraint reduction."""
    n = len(circles)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        points = circles[:, :2]
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        # Also add boundary edges (circles near walls)
        # For simplicity, just return the Delaunay edges
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _hex_initial():
    """Hexagonal close-packed initial configuration."""
    circles = []
    # Try to fit hexagonal packing
    # For 26 circles, use a pattern
    # Row 0: 5 circles, Row 1: 5, Row 2: 5, Row 3: 5, Row 4: 4, Row 5: 2
    # Let's use a simpler approach: grid with perturbation
    side = 5
    r = 0.18
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    x0 = r + 0.01
    y0 = r + 0.01
    count = 0
    for row in range(6):
        for col in range(6):
            if count >= N:
                break
            x = x0 + col * dx + (row % 2) * dx / 2
            y = y0 + row * dy
            if x + r > 1.0:
                x = 1.0 - r
            if y + r > 1.0:
                y = 1.0 - r
            circles.append([x, y, r])
            count += 1
    return np.array(circles[:N])


def _grid_initial():
    """Grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _random_initial():
    """Random initial configuration."""
    circles = np.random.uniform(0.05, 0.95, size=(N, 2))
    radii = np.full(N, 0.1)
    return np.column_stack([circles, radii])


def _optimize_from_initial(init, penalty_weight=50.0, max_iter=200):
    """Run L-BFGS-B optimization from initial configuration."""
    x0 = init.flatten()
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    result = minimize(
        _objective,
        x0,
        args=(penalty_weight,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(-1, 3)


def _polish(circles, penalty_weight=1000.0, max_iter=100):
    """High-precision polish pass."""
    x0 = circles.flatten()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
    
    result = minimize(
        _objective,
        x0,
        args=(penalty_weight,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    return result.x.reshape(-1, 3)


def construct():
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [_grid_initial(), _hex_initial()]
    # Add random inits
    for _ in range(3):
        inits.append(_random_initial())
    
    for init in inits:
        if _remaining() < 5.0:
            break
        try:
            result = _optimize_from_initial(init, penalty_weight=50.0, max_iter=200)
            # Polish
            result = _polish(result, penalty_weight=1000.0, max_iter=100)
            score = _compute_score(result)
            if score > best_score:
                best_score = score
                best_circles = result
        except Exception:
            continue
    
    # If no valid result, return grid
    if best_circles is None or best_score < 0:
        return _grid_initial()
    
    return best_circles
# EVOLVE-BLOCK-END
