"""Delaunay Active-Set Heilbronn-11: minimize over Delaunay triangles only, with periodic re-triangulation."""
import numpy as np
import time
import itertools
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

DEADLINE = 138.0  # Safety margin from 150s limit

# Precompute all triplets for N=11 points (for verification)
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]


def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _all_min_area(points):
    """Calculate the minimum area of all triangles formed by the points."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()

    # Clip to triangle boundaries
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)

    # Re-clip x after y adjustments
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)

    return np.column_stack([x, y])


def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and
            np.all(y <= S3 * x + tol) and
            np.all(S3 * x + y <= S3 + tol))


def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r


def _denser_lattice_init():
    """Initialize with a denser lattice for better starting configuration."""
    # Use a grid-like initialization in barycentric space
    bary = np.zeros((N, 3))
    # Place points on a grid in barycentric coordinates
    # For N=11, use a 4x4 grid minus corners, or similar
    # Actually, let's use a more structured approach
    # Place points at regular intervals
    positions = []
    # Corner-ish points (slightly inside)
    eps = 0.1
    positions.append([1-2*eps, eps, eps])
    positions.append([eps, 1-2*eps, eps])
    positions.append([eps, eps, 1-2*eps])
    # Edge midpoints
    positions.append([0.5, 0.5, 0])
    positions.append([0.5, 0, 0.5])
    positions.append([0, 0.5, 0.5])
    # Center
    positions.append([1/3, 1/3, 1/3])
    # Additional points
    positions.append([0.2, 0.3, 0.5])
    positions.append([0.3, 0.5, 0.2])
    positions.append([0.5, 0.2, 0.3])
    positions.append([0.4, 0.4, 0.2])
    # Normalize
    bary = np.array(positions)
    bary /= bary.sum(axis=1, keepdims=True)
    return bary


def _delaunay_triangles(points):
    """Get Delaunay triangle indices for the point set."""
    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception:
        # Fallback to all triangles if Delaunay fails
        return np.array(_TRIPLETS)


def _delaunay_min_area(points, tri_indices):
    """Calculate minimum area over Delaunay triangles only."""
    if len(tri_indices) == 0:
        return 0.0
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _soft_delaunay_min_area(points, tri_indices, alpha=100):
    """Soft minimum over Delaunay triangles."""
    if len(tri_indices) == 0:
        return 0.0
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    np.random.seed(42)

    # Initial best using denser lattice
    best_bary = _denser_lattice_init()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _all_min_area(best_cart)

    # Get initial Delaunay triangulation
    best_tri = _delaunay_triangles(best_cart)

    # Phase 1: Soft-min annealing with Delaunay active set
    alphas = [5, 20, 50, 100, 200, 500]

    for alpha in alphas:
        if time.time() > deadline:
            break

        num_restarts = 8
        for r in range(num_restarts):
            if time.time() > deadline:
                break

            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            else:
                x0 = _random_bary().flatten()

            def bary_objective(flat_bary, alpha=alpha):
                bary = flat_bary.reshape(N, 3)
                bary = np.maximum(0, bary)
                bary_sum = bary.sum(axis=1, keepdims=True)
                bary_sum = np.where(bary_sum < 1e-10, 1.0, bary_sum)
                bary /= bary_sum

                cart = _bary_to_cart(bary)
                cart = _project_cartesian(cart)

                # Get Delaunay triangles for current configuration
                try:
                    tri = Delaunay(cart)
                    tri_idx = tri.simplices
                except Exception:
                    tri_idx = np.array(_TRIPLETS)

                # Soft minimum over Delaunay triangles
                soft_min = _soft_delaunay_min_area(cart, tri_idx, alpha)

                # Penalty for points outside triangle
                x = cart[:, 0]
                y = cart[:, 1]
                pen1 = np.sum(np.maximum(0, -y)**2)
                pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
                pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
                pen = pen1 + pen2 + pen3

                return -soft_min + 100 * pen

            res = minimize(bary_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-6})

            # Evaluate the result
            new_bary = res.x.reshape(N, 3)
            new_bary = np.maximum(0, new_bary)
            new_bary_sum = new_bary.sum(axis=1, keepdims=True)
            new_bary_sum = np.where(new_bary_sum < 1e-10, 1.0, new_bary_sum)
            new_bary /= new_bary_sum

            new_cart = _bary_to_cart(new_bary)
            new_cart = _project_cartesian(new_cart)

            # Verify with all triangles
            new_score = _all_min_area(new_cart)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
                best_cart = new_cart
                best_tri = _delaunay_triangles(best_cart)

    # Phase 2: Hard minimum refinement with periodic re-triangulation
    if time.time() < deadline - 15:
        # Use the Delaunay triangles as active set for hard minimum optimization
        current_tri = best_tri.copy()
        current_cart = best_cart.copy()
        current_score = best_score

        def hard_objective(flat_cart, tri_indices=current_tri):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            min_area = _delaunay_min_area(cart, tri_indices)
            # Small penalty for boundary violations
            x = cart[:, 0]
            y = cart[:, 1]
            pen = np.sum(np.maximum(0, -y)**2) + np.sum(np.maximum(0, y - S3 * x)**2) + np.sum(np.maximum(0, S3 * x + y - S3)**2)
            return -min_area + 10 * pen

        # Iterate with re-triangulation
        for it in range(50):
            if time.time() > deadline - 5:
                break

            x0 = current_cart.flatten()
            res = minimize(hard_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7})

            new_cart = res.x.reshape(N, 2)
            new_cart = _project_cartesian(new_cart)

            # Verify with all triangles
            new_score = _all_min_area(new_cart)

            # Re-triangulate
            new_tri = _delaunay_triangles(new_cart)

            if new_score > current_score:
                current_score = new_score
                current_cart = new_cart
                current_tri = new_tri

                if current_score > best_score:
                    best_score = current_score
                    best_cart = current_cart
                    best_bary = _cart_to_bary(best_cart)
                    best_tri = current_tri

    # Final verification
    final_cart = _project_cartesian(best_cart)
    final_score = _all_min_area(final_cart)

    return final_cart
# EVOLVE-BLOCK-END
