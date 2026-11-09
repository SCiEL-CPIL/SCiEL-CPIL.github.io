"""Optimize inner hexagon translations (fixed orientations) to minimize outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

# The 6 unit outward normals of the outer hexagon (pointing to edge midpoints)
_WALL_NORMALS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
_WALL_ANGLES = np.array([math.radians(a) for a in range(0, 360, 60)])


def _support(angle):
    """Support function of a unit regular hexagon (circumradius 1) at direction `angle` (radians)."""
    k = np.arange(6)
    return float(np.max(np.cos(angle - k * math.pi / 3.0)))


def _compute_outer_side(P, orientations):
    """Exact outer side length for given positions P (N x 2) and orientations (N,)."""
    max_ext = 0.0
    for i in range(N):
        for w in range(6):
            n_w = _WALL_NORMALS[w]
            ext = abs(n_w[0] * P[i, 0] + n_w[1] * P[i, 1]) + _support(orientations[i] - _WALL_ANGLES[w])
            if ext > max_ext:
                max_ext = ext
    return 2.0 * max_ext / math.sqrt(3.0)


def _check_overlap(P, orientations, tol=1e-6):
    """Return True if any two hexagons overlap."""
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = P[i] - P[j]
            r = np.linalg.norm(d)
            if r < 1e-9:
                return True
            theta_rel = math.atan2(d[1], d[0])
            # Minkowski difference of two hexagons with orientations θ_i, θ_j
            # has support h_diff(φ) = h(φ - θ_i) + h(φ - θ_j)
            # They overlap iff for all φ: |d·e(φ)| ≤ h_diff(φ)
            # Equivalent: max_φ (|d·e(φ)| - h_diff(φ)) ≤ 0
            # Since d has fixed direction θ_rel, |d·e(φ)| = r·|cos(φ - θ_rel)|
            # So check: max_φ (r·|cos(φ - θ_rel)| - h(φ - θ_i) - h(φ - θ_j)) ≤ 0
            # Sample over φ
            phi_vals = np.linspace(0, 2 * math.pi, 720)
            vals = r * np.abs(np.cos(phi_vals - theta_rel)) - (_support(phi_vals - orientations[i]) + _support(phi_vals - orientations[j]))
            if np.max(vals) > tol:
                return True
    return False


def _objective(x, N, orientations):
    """Objective: outer side length + large overlap penalty."""
    P = x.reshape(N, 2)
    side = _compute_outer_side(P, orientations)
    
    # Overlap penalty
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = P[i] - P[j]
            r = np.linalg.norm(d)
            if r < 1e-9:
                pen += 100.0
                continue
            theta_rel = math.atan2(d[1], d[0])
            # Approximate overlap: check if centers are too close
            # Use a simpler check: distance < 2 (min separation for same-orientation)
            # For different orientations, the minimum separation varies
            # Use: if r < 1.9, definitely overlapping
            if r < 1.9:
                pen += (1.9 - r) ** 2
            else:
                # More careful check with sampled angles
                phi_vals = np.linspace(0, 2 * math.pi, 360)
                vals = r * np.abs(np.cos(phi_vals - theta_rel)) - (_support(phi_vals - orientations[i]) + _support(phi_vals - orientations[j]))
                max_v = np.max(vals)
                if max_v > 0:
                    pen += max_v ** 2
    
    return side + 1000.0 * pen


def _initial_positions(spacing=2.05):
    """Hexagonal lattice initial positions for N hexagons."""
    positions = []
    rows = [4, 3, 4]
    for ri, count in enumerate(rows):
        y = (ri - 1) * spacing * math.sqrt(3) / 2
        for ci in range(count):
            x = (ci - (count - 1) / 2.0) * spacing
            positions.append([x, y])
    while len(positions) < N:
        positions.append([len(positions) * spacing, 0.0])
    return np.array(positions[:N], dtype=float)


def construct():
    global _START, _DEADLINE
    N = 11
    orientations = np.zeros(N)  # all axis-aligned

    best_side = float('inf')
    best_P = None

    np.random.seed(42)
    for start_iter in range(8):
        if time.time() > _DEADLINE - 15:
            break
        P0 = _initial_positions(spacing=2.0 + 0.08 * start_iter)
        if start_iter > 0:
            P0 = P0 + np.random.randn(N, 2) * (0.2 + 0.05 * start_iter)
        x0 = P0.flatten()

        res = minimize(lambda x: _objective(x, N, orientations), x0,
                       method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 40})

        if res.fun < best_side:
            P = res.x.reshape(N, 2)
            if not _check_overlap(P, orientations, tol=1e-5):
                side = _compute_outer_side(P, orientations)
                if side < best_side:
                    best_side = side
                    best_P = P.copy()

    # Final polish
    if best_P is not None and time.time() < _DEADLINE - 5:
        x0 = best_P.flatten()
        res = minimize(lambda x: _objective(x, N, orientations), x0,
                       method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 80})
        P = res.x.reshape(N, 2)
        if not _check_overlap(P, orientations, tol=1e-6):
            side = _compute_outer_side(P, orientations)
            if side < best_side:
                best_side = side
                best_P = P.copy()

    if best_P is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = [[float(best_P[i, 0]), float(best_P[i, 1]), 0.0] for i in range(N)]
    outer_side = _compute_outer_side(best_P, orientations)
    return inner, (0.0, 0.0), float(outer_side), 0.0
