"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
# Vertices of equilateral triangle with area 1
# (0,0), (1,0), (0.5, sqrt(3)/2)
S3 = np.sqrt(3.0)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])


def _bary_to_cart(bary):
    """Convert barycentric coords (n,3) to Cartesian (n,2)."""
    # bary[i] = (a, b, c) with a+b+c=1, point = a*V0 + b*V1 + c*V2
    return bary[:, 1:2] * V1 + bary[:, 2:3] * V2


def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3) relative to V0,V1,V2."""
    # V0=(0,0), V1=(1,0), V2=(0.5, S3/2)
    # x = b + 0.5*c, y = (S3/2)*c
    # c = 2*y/S3, b = x - 0.5*c, a = 1-b-c
    c = 2.0 * pts[:, 1] / S3
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _project_to_triangle(pts):
    """Project points onto the triangle (clip barycentric coords to [0,1] and renormalize)."""
    bary = _cart_to_bary(pts)
    # Clip to valid region
    bary = np.clip(bary, 0.0, None)
    # Renormalize
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    bary = bary / s
    return _bary_to_cart(bary)


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a


def _objective(x):
    """Negative of min triangle area for minimization."""
    pts = _bary_to_cart(x)
    return -_min_triangle_area(pts)


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    best_score = -float('inf')
    best_pts = None

    # Initial random points inside triangle
    rng = np.random.RandomState(42)

    def random_points_in_triangle():
        pts = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            # Barycentric: (1-u-v, u, v)
            bary = np.array([1.0 - u - v, u, v])
            pt = bary[1] * V1 + bary[2] * V2
            pts.append(pt)
        return np.array(pts)

    # Multiple restarts with differential evolution + Nelder-Mead polish
    n_restarts = 12

    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break

        # Random initial population for DE
        x0 = random_points_in_triangle()
        x0_bary = _cart_to_bary(x0)

        # Bounds for barycentric coordinates: each in [0,1]
        bounds = [(0.0, 1.0)] * (N * 3)

        # Differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=restart * 1000 + 7,
                maxiter=60,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 0.9),
                recombination=0.7,
                init='sobol',
                polish=True,
                callback=None,
                x0=x0_bary,
            )
            if time.time() > deadline - 10.0:
                break
            score = -result.fun
            if score > best_score:
                best_score = score
                best_pts = _bary_to_cart(result.x)
        except Exception:
            pass

    # Additional Nelder-Mead refinements from best point
    if best_pts is not None:
        if time.time() < deadline - 5.0:
            best_bary = _cart_to_bary(best_pts)
            try:
                nm_result = minimize(
                    _objective,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if time.time() < deadline - 2.0:
                    score = -nm_result.fun
                    if score > best_score:
                        best_score = score
                        best_pts = _bary_to_cart(nm_result.x)
            except Exception:
                pass

    # Ensure points are inside the triangle
    if best_pts is not None:
        best_pts = _project_to_triangle(best_pts)
        # Final verification
        final_score = _min_triangle_area(best_pts)
        if final_score > best_score:
            best_score = final_score

    return best_pts
