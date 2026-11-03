"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, root

N = 11

# Equilateral triangle vertices (side length 1, area = sqrt(3)/4)
_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3) / 2.0])
_TRIG_AREA = np.sqrt(3) / 4.0  # area of the container triangle

# All C(N,3) = 165 triangle index triplets
_TRIPLES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])


def _barycentric_to_cart(bary):
    """Convert barycentric coords (relative to triangle ABC) to Cartesian.
    bary shape (N,3) -> cart shape (N,2)"""
    return bary[:, 0:1] * _A + bary[:, 1:2] * _B + bary[:, 2:3] * _C


def _cart_to_barycentric(cart):
    """Convert Cartesian to barycentric. cart shape (N,2) -> bary shape (N,3)"""
    # Solve: P = a*A + b*B + c*C, a+b+c=1
    # Using the standard formula with A=(0,0), B=(1,0), C=(0.5, h)
    h = np.sqrt(3) / 2.0
    x = cart[:, 0]
    y = cart[:, 1]
    # c = y / h
    c = y / h
    # b = x - 0.5*c
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _project_bary(bary):
    """Project barycentric coordinates to the feasible region (all coords in [0,1], sum=1)."""
    bary = bary.copy()
    # Normalize sum to 1
    bary /= bary.sum(axis=1, keepdims=True)
    # Clip to [0,1] and re-normalize iteratively
    for _ in range(10):
        bary = np.clip(bary, 0.0, 1.0)
        s = bary.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        bary /= s
    return bary


def _min_area_from_bary(bary):
    """Compute minimum triangle area (normalized by container area) from barycentric coords."""
    cart = _barycentric_to_cart(bary)
    areas = np.zeros(len(_TRIPLES))
    for idx, (i, j, k) in enumerate(_TRIPLES):
        p1, p2, p3 = cart[i], cart[j], cart[k]
        areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return areas / _TRIG_AREA  # normalized


def _soft_min(areas, beta):
    """Soft minimum: (sum(exp(-beta*a)))^(-1/beta) approximates min(areas)."""
    exp_neg = np.exp(-beta * areas)
    s = exp_neg.sum()
    if s < 1e-300:
        return 0.0
    return s ** (-1.0 / beta)


def _objective_and_grad(bary, beta):
    """Objective (negative soft-min) and gradient for optimization in barycentric space."""
    cart = _barycentric_to_cart(bary)
    areas_raw = np.zeros(len(_TRIPLES))
    # Compute areas
    for idx, (i, j, k) in enumerate(_TRIPLES):
        p1, p2, p3 = cart[i], cart[j], cart[k]
        areas_raw[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    areas = areas_raw / _TRIG_AREA

    exp_neg = np.exp(-beta * areas)
    s = exp_neg.sum()
    if s < 1e-300:
        return 0.0, np.zeros_like(bary)
    val = s ** (-1.0 / beta)

    # Gradient: d/dx of s^(-1/beta) = (-1/beta) * s^(-1/beta - 1) * ds/dx
    # ds/dx = sum_k exp(-beta*a_k) * (-beta) * da_k/dx
    # So dval/dx = (-1/beta) * s^(-1/beta-1) * (-beta) * sum_k exp(-beta*a_k)*da_k/dx
    #           = s^(-1/beta-1) * sum_k exp(-beta*a_k) * da_k/dx
    coeff = s ** (-1.0 / beta - 1.0)
    weights = coeff * exp_neg  # shape (165,)

    # da_k/d(point_m) for triangle (i,j,k):
    # area = 0.5 * |cross(p2-p1, p3-p1)|
    # We need the signed area to get gradient. Let's use signed area.
    # Actually, for the gradient of |f|, it's sign(f)*grad(f).
    # Let's compute signed areas and their gradients.

    grad = np.zeros_like(bary)  # (N,3) but we only use 2 independent coords; barycentric constraint handled by projection
    # Actually, let's work in Cartesian for gradient then map back.
    # Simpler: compute gradient w.r.t. Cartesian coords, then the barycentric gradient is just the mapping.
    # Since bary -> cart is linear: cart = bary @ M where M is 3x2 (columns are A,B,C)
    # d cart_m / d bary_n = M[n, :]
    # So grad_bary[n, :] = grad_cart[m, :] for the point m, mapped through the linear relation.
    # Actually: cart = sum_n bary[n] * V[n] where V is (N,2) = [_A,_B,_C] broadcast... 
    # No. cart[i] = bary[i,0]*A + bary[i,1]*B + bary[i,2]*C
    # So d cart[i] / d bary[i,:] = [A, B, C] (3x2 matrix)
    # d cart[i] / d bary[j,:] = 0 for j != i

    # Compute signed areas and their gradients w.r.t. Cartesian points
    signed_areas = np.zeros(len(_TRIPLES))
    # grad_cart[m, dim] for each triangle
    grad_cart = np.zeros((N, 2))

    for idx, (i, j, k) in enumerate(_TRIPLES):
        p1, p2, p3 = cart[i], cart[j], cart[k]
        # signed area = 0.5 * ((p2-p1) x (p3-p1))
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        signed_area = 0.5 * cross
        signed_areas[idx] = signed_area

        # d signed_area / d p1 = 0.5 * (-(p3[1]-p1[1]), (p3[0]-p1[0]))  ... wait
        # signed_area = 0.5 * (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
        # d/dx1: 0.5 * (-(y3-y1) - (y2-y1)) = 0.5 * (y1-y3 + y1-y2) = y1 - 0.5*(y2+y3)
        # d/dy1: 0.5 * ((x2-x1) - (x3-x1)) = 0.5 * (x2-x3)
        # d/dx2: 0.5 * (y3-y1)
        # d/dy2: -0.5 * (x3-x1)
        # d/dx3: -0.5 * (y2-y1)
        # d/dy3: 0.5 * (x2-x1)

        da_d_p1 = np.array([p1[1] - 0.5 * (p2[1] + p3[1]), 0.5 * (p2[0] - p3[0])])
        da_d_p2 = np.array([0.5 * (p3[1] - p1[1]), -0.5 * (p3[0] - p1[0])])
        da_d_p3 = np.array([-0.5 * (p2[1] - p1[1]), 0.5 * (p2[0] - p1[0])])

        # Gradient of |signed_area| w.r.t. each point:
        # d|f|/dx = sign(f) * df/dx
        sign = np.sign(signed_area)
        if sign == 0:
            sign = 1.0

        w = weights[idx] * sign
        grad_cart[i] += w * da_d_p1
        grad_cart[j] += w * da_d_p2
        grad_cart[k] += w * da_d_p3

    # Map Cartesian gradient to barycentric gradient
    # cart[i] = bary[i,0]*A + bary[i,1]*B + bary[i,2]*C
    # d cart[i] / d bary[i,0] = A, d cart[i] / d bary[i,1] = B, d cart[i] / d bary[i,2] = C
    # grad_bary[i,0] = grad_cart[i] . A
    # grad_bary[i,1] = grad_cart[i] . B
    # grad_bary[i,2] = grad_cart[i] . C
    grad_bary = np.zeros((N, 3))
    for i in range(N):
        grad_bary[i, 0] = np.dot(grad_cart[i], _A)
        grad_bary[i, 1] = np.dot(grad_cart[i], _B)
        grad_bary[i, 2] = np.dot(grad_cart[i], _C)

    # The objective is -soft_min, so we negate the gradient
    return -val, -grad_bary


def _optimize_from_bary(bary_init, beta, maxiter=200):
    """Optimize from a barycentric starting point."""
    def obj(x):
        bary = _project_bary(x)
        val, _ = _objective_and_grad(bary, beta)
        return val

    def grad(x):
        bary = _project_bary(x)
        _, g = _objective_and_grad(bary, beta)
        return g

    try:
        res = minimize(obj, bary_init.copy(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        bary_opt = _project_bary(res.x)
        return bary_opt
    except Exception:
        return _project_bary(bary_init)


def _evaluate(points_cart):
    """Compute the raw score (normalized min area)."""
    bary = _cart_to_barycentric(points_cart)
    areas = _min_area_from_bary(bary)
    return float(np.min(areas))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    best_score = 0.0
    best_bary = None

    # Generate diverse initial configurations
    rng = np.random.default_rng(42)

    # Strategy 1: structured grid-like points
    # Use a 3x3 grid plus 2 extra points
    structured = []
    for i in range(3):
        for j in range(3):
            # Barycentric coordinates for a 3x3 grid
            b = i / 2.0
            c = j / 2.0
            a = 1.0 - b - c
            if a >= -0.01 and b >= -0.01 and c >= -0.01:
                structured.append([max(0, a), max(0, b), max(0, c)])
    # Normalize
    for s in structured:
        s_sum = sum(s)
        if s_sum > 0:
            s = [x / s_sum for x in s]
    # Add 2 more points (center and one more)
    structured.append([1/3, 1/3, 1/3])
    structured.append([0.5, 0.25, 0.25])
    # Take first 11
    structured = structured[:11]
    structured = np.array(structured)
    structured = _project_bary(structured)

    # Strategy 2: random
    random_bary = rng.random((N, 3))
    random_bary = _project_bary(random_bary)

    # Strategy 3: vertices + midpoints + center
    verts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1/3, 1/3, 1/3],
        [0.25, 0.25, 0.5],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.75, 0.125, 0.125],
    ])
    verts = _project_bary(verts)

    # Strategy 4: more random
    random_bary2 = rng.random((N, 3))
    random_bary2 = _project_bary(random_bary2)

    # Strategy 5: concentric-like
    concentric = []
    # Inner triangle
    for t in [0.3, 0.6]:
        for ang in [0, 2*np.pi/3, 4*np.pi/3]:
            # Point at distance t from center in direction ang
            cx, cy = 0.5, np.sqrt(3)/6  # centroid
            px = cx + t * 0.3 * np.cos(ang)
            py = cy + t * 0.3 * np.sin(ang)
            bary = _cart_to_barycentric(np.array([[px, py]]))[0]
            concentric.append(bary)
    # Center
    concentric.append([1/3, 1/3, 1/3])
    # Two more
    concentric.append([0.5, 0.25, 0.25])
    concentric.append([0.25, 0.5, 0.25])
    concentric = np.array(concentric[:11])
    concentric = _project_bary(concentric)

    initial_sets = [structured, random_bary, verts, random_bary2, concentric]

    # Beta-annealing schedule
    betas = [20, 50, 100, 200, 400]

    # Multi-start with annealing
    for init_idx, init_bary in enumerate(initial_sets):
        if time.time() > deadline:
            break

        current_bary = init_bary.copy()

        for beta in betas:
            if time.time() > deadline:
                break

            current_bary = _optimize_from_bary(current_bary, beta, maxiter=150)

            # Evaluate
            cart = _barycentric_to_cart(current_bary)
            score = _evaluate(cart)

            if score > best_score:
                best_score = score
                best_bary = current_bary.copy()

            # Add small perturbation for next beta
            if beta < betas[-1]:
                perturbation = rng.normal(0, 0.05, size=(N, 3))
                current_bary = current_bary + perturbation
                current_bary = _project_bary(current_bary)

        # Additional restarts from best found so far
        if best_bary is not None and time.time() < deadline:
            for restart in range(3):
                if time.time() > deadline:
                    break
                perturbed = best_bary + rng.normal(0, 0.1, size=(N, 3))
                perturbed = _project_bary(perturbed)
                for beta in [100, 200, 400]:
                    if time.time() > deadline:
                        break
                    perturbed = _optimize_from_bary(perturbed, beta, maxiter=100)
                cart = _barycentric_to_cart(perturbed)
                score = _evaluate(cart)
                if score > best_score:
                    best_score = score
                    best_bary = perturbed.copy()

    # Final polish with high beta
    if best_bary is not None and time.time() < deadline:
        for beta in [500, 1000]:
            if time.time() > deadline:
                break
            best_bary = _optimize_from_bary(best_bary, beta, maxiter=200)
        cart = _barycentric_to_cart(best_bary)
        score = _evaluate(cart)
        if score > best_score:
            best_score = score

    # Ensure we return valid points
    if best_bary is None:
        best_bary = _project_bary(rng.random((N, 3)))

    cart = _barycentric_to_cart(best_bary)
    return np.array(cart)
