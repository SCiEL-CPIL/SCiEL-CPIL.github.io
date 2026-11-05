"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Correct "flower" arrangement:
    # 1 center + 6 inner ring (distance sqrt(3), angles 30,90,150,210,270,330)
    # + 4 outer ring (distance 3, angles 0,60,120,180)
    # All inner hexagons have rotation 0.
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])

    # Inner ring: 6 hexagons at distance sqrt(3) + tiny epsilon
    r1 = math.sqrt(3) + eps
    for k in range(6):
        angle = math.radians(30 + k * 60)
        centers = np.vstack([centers, [r1 * math.cos(angle), r1 * math.sin(angle)]])

    # Outer ring: 4 hexagons at distance 3.0 + tiny epsilon
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = math.radians(k * 60)
        centers = np.vstack([centers, [r2 * math.cos(angle), r2 * math.sin(angle)]])

    n_hex = len(centers)

    # Base vertices of unit hexagon (pointy-right, angle 0)
    base_vertices = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
    ])

    def compute_side(theta_rel):
        """
        Compute minimum outer hexagon side length for given relative rotation.
        theta_rel = rotation of outer hexagon (inner hexagons fixed at 0).
        """
        theta_out = theta_rel
        # Outer hexagon face normals at angles 30 + k*60 + theta_out
        normals = np.array([
            [math.cos(math.radians(30 + k * 60) + theta_out),
             math.sin(math.radians(30 + k * 60) + theta_out)]
            for k in range(6)
        ])

        max_proj = 0.0
        for i in range(n_hex):
            verts = base_vertices + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, float(np.max(np.abs(dots))))

        # Apothem = S * sqrt(3)/2, so S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    # Coarse grid search over [0, pi/6] (periodic with period pi/3, symmetric)
    best_score = float('inf')
    best_theta = 0.0

    thetas = np.linspace(0, math.pi / 6, 2000)
    for t in thetas:
        if time.time() > deadline - 10:
            break
        s = compute_side(float(t))
        if s < best_score:
            best_score = s
            best_theta = float(t)

    # Refine with minimize_scalar around the best point
    step = (math.pi / 6) / 2000
    lo = max(0.0, best_theta - 10 * step)
    hi = min(math.pi / 6, best_theta + 10 * step)

    try:
        res = minimize_scalar(compute_side, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-14, 'maxiter': 2000})
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
    except Exception:
        pass

    # Final ultra-tight polish
    if time.time() < deadline - 5:
        lo2 = max(0.0, best_theta - 1e-10)
        hi2 = min(math.pi / 6, best_theta + 1e-10)
        try:
            res = minimize_scalar(compute_side, bounds=(lo2, hi2), method='bounded',
                                  options={'xatol': 1e-16, 'maxiter': 500})
            if res.fun < best_score:
                best_score = res.fun
                best_theta = res.x
        except Exception:
            pass

    # Build output
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], 0.0])

    side = best_score
    theta_out_deg = math.degrees(best_theta)

    return inner, (0.0, 0.0), side, theta_out_deg

# EVOLVE-BLOCK-END
