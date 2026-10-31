"""Rewritten program for Heilbronn triangles task."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = H / 2  # area of unit equilateral triangle = sqrt(3)/4

# Precompute triangle indices
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)


def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas


def _min_area(pts):
    """Minimum area of any triangle formed by three points."""
    return float(np.min(_all_areas(pts)))


def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c

    eps = 1e-10
    a = max(eps, min(1 - 2 * eps, a))
    b = max(eps, min(1 - 2 * eps, b))
    c = max(eps, min(1 - 2 * eps, c))

    s = a + b + c
    a /= s
    b /= s
    c /= s

    return a * V0 + b * V1 + c * V2


def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])


def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    shifted -= np.max(shifted)  # stability
    log_sum = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    soft_min_val = m - log_sum / beta
    return soft_min_val


def _objective_smooth(flat_pts, beta=50.0):
    """Smooth objective for L-BFGS-B."""
    pts = _clamp_all(flat_pts.reshape(-1, 2))
    return -_soft_min_area(pts, beta)


def _objective_hard(flat_pts):
    """Hard objective for Nelder-Mead / Powell."""
    pts = _clamp_all(flat_pts.reshape(-1, 2))
    return -_min_area(pts)


def _generate_hexagonal_init():
    """Generate initial configuration using hexagonal lattice pattern."""
    pts = []
    # Center point
    cx, cy = 1/3, H/3
    pts.append([cx, cy])

    # Ring 1: 6 points around center
    r1 = 0.2
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        px = cx + r1 * np.cos(angle)
        py = cy + r1 * np.sin(angle)
        pts.append([px, py])

    # Ring 2: 4 more points (total 11)
    r2 = 0.35
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        pts.append([px, py])

    return _clamp_all(np.array(pts))


def _generate_symmetric_init():
    """Symmetric initial placement with 3-fold symmetry."""
    pts = []
    # Center
    cx, cy = 1/3, H/3
    pts.append([cx, cy])

    # 3 points at 120 degrees, radius r1
    r1 = 0.18
    for angle in [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]:
        px = cx + r1 * np.cos(angle)
        py = cy + r1 * np.sin(angle)
        pts.append([px, py])

    # 3 points at 120 degrees, radius r2, offset
    r2 = 0.32
    for angle in [np.pi/2 + np.pi/3, np.pi/2 + np.pi/3 + 2*np.pi/3, np.pi/2 + np.pi/3 + 4*np.pi/3]:
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        pts.append([px, py])

    # 3 more points near vertices
    m = 0.12
    for v in [V0, V1, V2]:
        # Move inward from each vertex
        inward = np.array([cx, cy]) - v
        pts.append(v + m * inward / np.linalg.norm(inward))

    return _clamp_all(np.array(pts))


def _generate_random_init(seed=None):
    """Random initial placement inside triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        # Random barycentric coordinates
        u = rng.rand()
        v = rng.rand()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        a, b, c = 1 - u - v, u, v
        p = a * V0 + b * V1 + c * V2
        pts.append(p)
    return _clamp_all(np.array(pts))


def _force_directed_relax(pts, iterations=200, target_min_dist=None):
    """
    Force-directed relaxation: treat points as particles with repulsive forces
    proportional to 1/d^2, and also push apart any close triple.
    This is a heuristic to spread points out.
    """
    pts = pts.copy().astype(float)
    n = len(pts)

    if target_min_dist is None:
        # Estimate from area: each point should 'own' area/11
        target_min_dist = np.sqrt(AREA_EQUI / n) * 0.8

    for it in range(iterations):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                d = pts[j] - pts[i]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                # Repulsive force: k / dist^2
                f = 0.05 / (dist * dist)
                d_hat = d / dist
                forces[i] -= f * d_hat
                forces[j] += f * d_hat

        # Also add a gentle attraction to center to prevent drift
        center = np.array([1/3, H/3])
        for i in range(n):
            to_center = center - pts[i]
            forces[i] += 0.01 * to_center

        # Update positions with damping
        learning_rate = 0.01 * (1 - it / iterations)
        pts += learning_rate * forces
        pts = _clamp_all(pts)

    return pts


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def time_left():
        return deadline - (time.time() - start_time)

    # Bounds for optimization
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N

    # Generate multiple initial configurations
    inits = [
        _generate_hexagonal_init(),
        _generate_symmetric_init(),
        _generate_random_init(42),
        _generate_random_init(123),
        _generate_random_init(999),
    ]

    # Score all inits
    best_pts = None
    best_val = -1
    for init in inits:
        val = _min_area(init)
        if val > best_val:
            best_val = val
            best_pts = init

    # Also try force-directed relaxation on the best init
    if time_left() > 30:
        relaxed = _force_directed_relax(best_pts, iterations=150)
        val_relax = _min_area(relaxed)
        if val_relax > best_val:
            best_val = val_relax
            best_pts = relaxed

    # Phase 1: L-BFGS-B on smooth objective
    if time_left() > 20:
        try:
            res = minimize(
                lambda x: _objective_smooth(x, beta=80.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            polished = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(polished)
            if val > best_val:
                best_val = val
                best_pts = polished
        except Exception:
            pass

    # Phase 2: Nelder-Mead on hard objective
    if time_left() > 15:
        try:
            res = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            polished = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(polished)
            if val > best_val:
                best_val = val
                best_pts = polished
        except Exception:
            pass

    # Phase 3: Powell on hard objective
    if time_left() > 10:
        try:
            res = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Powell',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            polished = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(polished)
            if val > best_val:
                best_val = val
                best_pts = polished
        except Exception:
            pass

    # Phase 4: Second round of L-BFGS-B with higher beta
    if time_left() > 8:
        try:
            res = minimize(
                lambda x: _objective_smooth(x, beta=150.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(polished)
            if val > best_val:
                best_val = val
                best_pts = polished
        except Exception:
            pass

    # Phase 5: Final Nelder-Mead polish
    if time_left() > 3:
        try:
            res = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            polished = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(polished)
            if val > best_val:
                best_val = val
                best_pts = polished
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
