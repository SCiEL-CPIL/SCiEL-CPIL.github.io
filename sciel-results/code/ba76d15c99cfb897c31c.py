"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
H = np.sqrt(3) / 2.0

# Precompute vertex indices for the 165 triangles
from itertools import combinations
_vertex_indices = list(combinations(range(N), 3))
_n_tri = len(_vertex_indices)

# Precompute coefficients for area = 0.5 * |det|
# det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
# = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
# So area = 0.5 * |x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)|

def _build_coeff_matrix():
    """Build matrix A (n_tri x 2N) such that A @ x_flat gives det values for all triangles."""
    A = np.zeros((_n_tri, 2 * N))
    for idx, (i, j, k) in enumerate(_vertex_indices):
        # det = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
        A[idx, i] = 1.0  # coefficient of x_i
        A[idx, N + j] = -1.0  # coefficient of y_i (negative because y_i appears in x_k*(y_i - y_j) with +, but let's re-derive)
        # Let me re-derive:
        # det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        # = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
        # = x_j*y_k - x_j*y_i - x_i*y_k - y_j*x_k + y_j*x_i + y_i*x_k
        # Group by x_i, x_j, x_k:
        # x_i: -y_k + y_j
        # x_j: y_k - y_i
        # x_k: y_i - y_j
        # So det = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
        # In terms of flat vector [x_0, y_0, x_1, y_1, ...]:
        # A[idx, i] = 1.0 (multiplied by x_i, but we need the y terms too)
        # Actually, this is bilinear in x and y, so it's not linear in the flat vector!
        # We need to handle this differently.
        pass
    return A

# Since the area is bilinear (product of x and y differences), it's not a linear function of the flattened coordinates.
# Instead, we'll use the fact that for fixed other points, the area is linear in each point's coordinates.
# For L-BFGS-B, we'll compute gradients numerically or analytically.

def _min_area_squared(points):
    """Compute the squared minimum triangle area."""
    n = len(points)
    min_sq = np.inf
    for (i, j, k) in _vertex_indices:
        det = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
              (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
        sq = det * det * 0.25
        if sq < min_sq:
            min_sq = sq
    return min_sq

def _neg_min_area_squared(x_flat):
    """Negative of squared min area for minimization."""
    points = x_flat.reshape(N, 2)
    return -_min_area_squared(points)

def _gradient_neg_min_area_sq(x_flat):
    """Numerical gradient of negative min squared area."""
    eps = 1e-7
    n = len(x_flat)
    grad = np.zeros(n)
    f0 = _neg_min_area_squared(x_flat)
    for i in range(n):
        xp = x_flat.copy()
        xp[i] += eps
        fp = _neg_min_area_squared(xp)
        grad[i] = (fp - f0) / eps
    return grad

def _check_inside_triangle(points):
    """Check if all points are inside or on the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, H)
    # For a point to be inside, it must satisfy:
    # y >= 0
    # y <= H * x  (below the left edge from (0,0) to (0.5, H)... wait)
    # Actually, the edges are:
    # Bottom: y = 0, x in [0, 1]
    # Left: from (0,0) to (0.5, H): y = H * x / 0.5 = 2*H*x... no
    # Line from (0,0) to (0.5, H): slope = H/0.5 = 2H, so y = 2H*x
    # Right: from (1,0) to (0.5, H): slope = (H-0)/(0.5-1) = H/(-0.5) = -2H, so y = -2H*(x-1) = 2H*(1-x)
    # So constraints:
    # y >= 0
    # y <= 2*H*x  (left of the left edge? No, inside means below the left edge line extended)
    # Wait, for the triangle with vertices (0,0), (1,0), (0.5, H):
    # A point (x,y) is inside if:
    # y >= 0
    # y <= H * (x / 0.5) = 2*H*x  for x <= 0.5? No.
    # Let's use barycentric or half-plane tests.
    # Half-plane for edge (0,0)-(1,0): y >= 0
    # Half-plane for edge (1,0)-(0.5,H): the line is y = -2H*(x-1) = 2H*(1-x). Inside is below this line for x in [0.5, 1], but actually the interior is on the side toward (0,0).
    # The line through (1,0) and (0.5, H): 2H*x + y - 2H = 0. At (0,0): 0 + 0 - 2H = -2H < 0. So inside is 2H*x + y - 2H <= 0, i.e., y <= 2H*(1-x).
    # Half-plane for edge (0.5,H)-(0,0): line through (0,0) and (0.5, H): H*x - 0.5*y = 0. At (1,0): H*1 - 0 = H > 0. So inside is H*x - 0.5*y >= 0, i.e., y <= 2*H*x.
    # So: y >= 0, y <= 2*H*x, y <= 2*H*(1-x).
    tol = -1e-9
    for i in range(N):
        x, y = points[i]
        if y < tol:
            return False
        if y > 2 * H * x + 1e-9:
            return False
        if y > 2 * H * (1 - x) + 1e-9:
            return False
    return True

def _project_to_triangle(points):
    """Project points back into the triangle if they're outside."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        # Clip to triangle
        if y < 0:
            y = 0
        if y > 2 * H * x:
            # Project onto line y = 2H*x
            # Minimize (x'-x)^2 + (y'-y)^2 subject to y' = 2H*x'
            # Optimal: x' = (x + 2H*y) / (1 + 4H^2)
            denom = 1 + 4 * H * H
            x = (x + 2 * H * y) / denom
            y = 2 * H * x
        if y > 2 * H * (1 - x):
            # Project onto line y = 2H*(1-x) = 2H - 2H*x
            # Minimize (x'-x)^2 + (y'-y)^2 subject to y' = 2H - 2H*x'
            # Let g(x') = (x'-x)^2 + (2H - 2H*x' - y)^2
            # dg/dx' = 2(x'-x) + 2(2H - 2H*x' - y)*(-2H) = 0
            # (x'-x) - 2H*(2H - 2H*x' - y) = 0
            # x' - x - 4H^2 + 4H^2*x' + 2H*y = 0
            # x'(1 + 4H^2) = x + 4H^2 - 2H*y
            denom = 1 + 4 * H * H
            x = (x + 4 * H * H - 2 * H * y) / denom
            y = 2 * H * (1 - x)
        # Final clip for numerical safety
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > 2 * H * x + 1e-9:
            y = 2 * H * x
        if y > 2 * H * (1 - x) + 1e-9:
            y = 2 * H * (1 - x)
        points[i] = [x, y]
    return points

def _initial_points(rng_seed=42):
    """Generate initial points inside the triangle."""
    rng = np.random.RandomState(rng_seed)
    points = np.zeros((N, 2))
    # Place points at a grid-like pattern inside the triangle
    # Use a simple approach: random sampling with rejection
    for i in range(N):
        for attempt in range(1000):
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = H * v
                if y >= 0 and y <= 2 * H * x and y <= 2 * H * (1 - x):
                    points[i] = [x, y]
                    break
    return points

def construct():
    deadline = time.time() + 110  # 110s to be safe within 120s
    
    best_points = None
    best_val = -np.inf
    
    # Try multiple random initializations
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        points = _initial_points(seed)
        x_flat = points.flatten()
        
        # Use L-BFGS-B to maximize min squared area
        result = minimize(
            _neg_min_area_squared,
            x_flat,
            method='L-BFGS-B',
            jac=_gradient_neg_min_area_sq,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < -best_val:
            pts = _project_to_triangle(result.x.reshape(N, 2))
            val = _min_area_squared(pts)
            if val > best_val:
                best_val = val
                best_points = pts
        
        if time.time() > deadline - 10:
            break
    
    if best_points is None:
        best_points = _initial_points(0)
    
    return best_points
