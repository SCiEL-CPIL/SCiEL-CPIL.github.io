"""Heilbronn-11: Barycentric optimization with soft-min annealing."""
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import time

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of equilateral triangle with side 1

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian.
    Point i has barycentric coords (u_i, v_i, 1-u_i-v_i) relative to V0, V1, V2.
    cart_i = u_i*V1 + v_i*V2  (since V0 is origin)
    """
    uv = uv_flat.reshape(N, 2)
    # cart = u*V1 + v*V2
    cart = uv[:, 0:1] * V1 + uv[:, 1:2] * V2
    return cart


def _cart_to_bary(cart):
    """Convert Cartesian points to barycentric (u, v) where w = 1-u-v >= 0."""
    # Solve: cart = u*V1 + v*V2
    # x = u + 0.5*v
    # y = S3H*v
    # v = y / S3H
    # u = x - 0.5*v
    v = cart[:, 1] / S3H
    u = cart[:, 0] - 0.5 * v
    return np.column_stack([u, v]).flatten()


def _vectorized_min_area_cart(points):
    """Compute min area of triangles from N Cartesian points (N,2)."""
    # points shape (N, 2)
    pi = points[_T_I]  # (165, 2)
    pj = points[_T_J]
    pk = points[_T_K]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _soft_min_area_cart(points, beta=1.0):
    """Soft minimum of triangle areas using log-sum-exp."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    # Soft min: -log(sum(exp(-beta*areas)))/beta
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
    return soft_min


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(N, 2).copy()
    for i in range(N):
        u, v = uv[i]
        if u < 0:
            u = 0
        if v < 0:
            v = 0
        if u + v > 1:
            # Scale down
            s = 1.0 / (u + v)
            u *= s
            v *= s
        uv[i] = [u, v]
    return uv.flatten()


def _is_valid_cart(points, eps=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(N):
        x, y = points[i]
        if y < -eps or x < -eps or x > 1 + eps:
            return False
        if y > S3 * x + eps:
            return False
        if y > S3 * (1 - x) + eps:
            return False
    return True


def _neg_soft_min(x, beta):
    """Objective: negative soft-min area in barycentric space."""
    uv = _project_to_simplex(x)
    cart = _bary_to_cart(uv)
    return -_soft_min_area_cart(cart, beta)


def _neg_min_area(x):
    """Objective: negative exact min area in barycentric space."""
    uv = _project_to_simplex(x)
    cart = _bary_to_cart(uv)
    return -_vectorized_min_area_cart(cart)


def _structured_init():
    """Triangular lattice-like initial configuration in barycentric coords."""
    points_cart = [
        [0.0, 0.0],      # V0
        [1.0, 0.0],      # V1
        [0.5, S3H],      # V2
        [0.5, 0.0],      # midpoint of V0-V1
        [1.0/6, S3/6],   # row 1 left
        [0.5, S3/6],     # row 1 mid
        [5.0/6, S3/6],   # row 1 right
        [1.0/3, S3/3],   # row 2 left
        [2.0/3, S3/3],   # row 2 right
        [0.4, S3/4],
        [0.6, S3/4],
    ]
    cart = np.array(points_cart)
    return _cart_to_bary(cart)


def _random_init_bary(seed):
    """Random points in barycentric simplex."""
    rng = np.random.RandomState(seed)
    uv = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random() * (1 - u)
        uv[i] = [u, v]
    return uv.flatten()


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin from 150s limit

    best_area = -np.inf
    best_uvs = None

    # Initial configurations
    inits = []
    try:
        inits.append(_structured_init())
    except:
        pass
    for seed in range(8):
        try:
            inits.append(_random_init_bary(seed))
        except:
            pass

    for init_uvs in inits:
        if time.time() - start_time > deadline - 15:
            break

        try:
            x0 = _project_to_simplex(init_uvs.copy())

            # Phase 1: soft-min annealing with Nelder-Mead
            # Start with moderate beta, anneal to high beta
            betas = [5.0, 20.0, 100.0, 500.0]
            current_x = x0.copy()

            for beta in betas:
                if time.time() - start_time > deadline - 10:
                    break
                try:
                    res = minimize(
                        _neg_soft_min,
                        current_x,
                        args=(beta,),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-10}
                    )
                    current_x = _project_to_simplex(res.x)
                except:
                    pass

            # Phase 2: exact min area polish with tighter tolerance
            if time.time() - start_time > deadline - 5:
                break
            try:
                res = minimize(
                    _neg_min_area,
                    current_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12}
                )
                final_uvs = _project_to_simplex(res.x)
            except:
                final_uvs = current_x

            # Convert to Cartesian and check validity
            cart = _bary_to_cart(final_uvs)
            if _is_valid_cart(cart):
                area = _vectorized_min_area_cart(cart)
                if area > best_area:
                    best_area = area
                    best_uvs = final_uvs.copy()

        except Exception:
            pass

    # Fallback
    if best_uvs is None:
        best_uvs = _structured_init()
        best_uvs = _project_to_simplex(best_uvs)

    cart = _bary_to_cart(best_uvs)
    # Final projection to ensure validity
    for i in range(N):
        x, y = cart[i]
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > S3 * x:
            y = S3 * x if x > 0 else 0
        if y > S3 * (1 - x):
            y = S3 * (1 - x) if x < 1 else 0
        cart[i] = [x, y]

    return cart


# EVOLVE-BLOCK-END
