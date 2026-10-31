"""Heilbronn 11-point problem: maximize the smallest triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
# Equilateral triangle vertices (unit area: area = sqrt(3)/4)
_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3) / 2.0])
_VERTS = np.array([_A, _B, _C])
_TRI_AREA = np.sqrt(3) / 4.0  # area of the big triangle


def _bary_to_cart(bary):
    """bary shape (n,3) -> cart shape (n,2)."""
    return bary @ _VERTS.T


def _min_area_cart(pts):
    """Minimum area of any triangle formed by 3 of the n points (raw, not divided)."""
    n = len(pts)
    best = np.inf
    for i, j, k in combinations(range(n), 3):
        a = pts[i]
        b = pts[j]
        c = pts[k]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area < best:
            best = area
    return best


def _objective(bary):
    """Return -min area (to be minimized). bary shape (n,3)."""
    pts = _bary_to_cart(bary)
    return -_min_area_cart(pts)


def _constraints(bary):
    """Constraints: each row sums to 1, each coord >= 0 (>= small margin)."""
    n = bary.shape[0]
    cons = []
    for i in range(n):
        cons.append({'type': 'eq', 'fun': lambda b, i=i: np.sum(b[i]) - 1.0})
    for i in range(n):
        for c in range(3):
            cons.append({'type': 'ineq', 'fun': lambda b, i=i, c=c: b[i, c] - 1e-4})
    return cons


def _initial_bary(seed):
    """Generate a random barycentric starting point (uniform in triangle)."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 3))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        bary[i] = [1.0 - u - v, u, v]
    return bary


def _solve_from(bary0):
    """Run SLSQP from bary0, return best bary found."""
    bounds = [(0.0, 1.0)] * (N * 3)
    res = minimize(
        _objective,
        bary0.reshape(-1),
        method='SLSQP',
        bounds=bounds,
        constraints=_constraints(bary0),
        options={'maxiter': 200, 'ftol': 1e-10},
    )
    bary = res.x.reshape(N, 3)
    # normalize rows to sum to 1
    bary = bary / bary.sum(axis=1, keepdims=True)
    bary = np.clip(bary, 0.0, 1.0)
    return bary


def construct():
    deadline = time.time() + 100.0  # safety margin before 120s limit
    best_bary = None
    best_val = -np.inf

    # Start with a few well-chosen seeds
    seeds = [1, 7, 42, 123, 456, 789, 2024, 3141, 2718, 1618]
    for s in seeds:
        if time.time() > deadline:
            break
        bary0 = _initial_bary(s)
        bary = _solve_from(bary0)
        val = -_objective(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()

    # Local perturbation refinement on the best found
    if best_bary is not None:
        rng = np.random.default_rng(99)
        for _ in range(30):
            if time.time() > deadline:
                break
            perturb = best_bary.copy()
            idx = rng.integers(0, N)
            perturb[idx] += rng.normal(0, 0.05, 3)
            perturb = np.clip(perturb, 0.0, None)
            perturb = perturb / perturb.sum(axis=1, keepdims=True)
            bary = _solve_from(perturb)
            val = -_objective(bary)
            if val > best_val:
                best_val = val
                best_bary = bary.copy()

    if best_bary is None:
        # Fallback: simple random points
        bary = _initial_bary(0)
        best_bary = bary

    pts = _bary_to_cart(best_bary)
    return np.array(pts)
