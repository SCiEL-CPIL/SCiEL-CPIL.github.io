import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates (sum to 1).
    Returns the minimum triangle area (normalized: 0.5 * |det| of barycentric matrix).
    """
    min_area = np.inf
    for i in range(N - 2):
        pi = p[i]
        for j in range(i + 1, N - 1):
            pj = p[j]
            ps_k = p[j+1:]
            dets = (pi[0]*(pj[1]*ps_k[:,2] - pj[2]*ps_k[:,1])
                   - pi[1]*(pj[0]*ps_k[:,2] - pj[2]*ps_k[:,0])
                   + pi[2]*(pj[0]*ps_k[:,1] - pj[1]*ps_k[:,0]))
            areas = 0.5 * np.abs(dets)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _bary_to_cart(bary):
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _objective_flat(x):
    """Objective for optimizer: negative min area (we maximize)."""
    p = x.reshape(N, 3)
    p = _project_bary(p)
    return -_min_area_bary(p)

def _make_hexagonal_init():
    """Hexagonal lattice-like arrangement in barycentric coords."""
    bary = np.zeros((N, 3))
    # 3 corners
    bary[0] = [0.85, 0.075, 0.075]
    bary[1] = [0.075, 0.85, 0.075]
    bary[2] = [0.075, 0.075, 0.85]
    # 3 edge midpoints (slightly inward)
    bary[3] = [0.425, 0.425, 0.15]
    bary[4] = [0.15, 0.425, 0.425]
    bary[5] = [0.425, 0.15, 0.425]
    # Center
    bary[6] = [1/3, 1/3, 1/3]
    # 4 inner ring points (hexagonal positions)
    r = 0.2
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi/4
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle)
        w = 1 - u - v
        bary[7+i] = [max(u,0), max(v,0), max(w,0)]
    # Normalize to ensure sum=1
    for i in range(N):
        s = np.sum(bary[i])
        if s > 0:
            bary[i] /= s
        else:
            bary[i] = [1/3, 1/3, 1/3]
    return bary

def _make_ring_init():
    """Two concentric rings + center."""
    bary = np.zeros((N, 3))
    bary[0] = [1/3, 1/3, 1/3]  # center
    # Inner ring: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.18
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle)
        w = 1 - u - v
        bary[1+i] = [max(u,0), max(v,0), max(w,0)]
    # Outer ring: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi/5
        r = 0.38
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle)
        w = 1 - u - w
        bary[6+i] = [max(u,0), max(v,0), max(w,0)]
    for i in range(N):
        s = np.sum(bary[i])
        if s > 0:
            bary[i] /= s
        else:
            bary[i] = [1/3, 1/3, 1/3]
    return bary

def _make_corner_ring_init():
    """Corners + mid-edges + inner cluster."""
    bary = np.zeros((N, 3))
    # Corners
    bary[0] = [0.9, 0.05, 0.05]
    bary[1] = [0.05, 0.9, 0.05]
    bary[2] = [0.05, 0.05, 0.9]
    # Edge midpoints
    bary[3] = [0.45, 0.45, 0.1]
    bary[4] = [0.1, 0.45, 0.45]
    bary[5] = [0.45, 0.1, 0.45]
    # Inner hexagon (5 points around center)
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.15
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle)
        w = 1 - u - v
        bary[6+i] = [max(u,0), max(v,0), max(w,0)]
    for i in range(N):
        s = np.sum(bary[i])
        if s > 0:
            bary[i] /= s
        else:
            bary[i] = [1/3, 1/3, 1/3]
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    best_area = 0.0
    best_bary = None

    def _is_time_up():
        return time.time() > deadline

    # Initial configurations
    inits = [
        _make_hexagonal_init(),
        _make_ring_init(),
        _make_corner_ring_init(),
    ]
    
    # Add a few random inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(_project_bary(bary))

    nm_options = {
        'maxiter': 3000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
        'disp': False,
    }

    for init_bary in inits:
        if _is_time_up():
            break

        x0 = init_bary.flatten()
        current_area = -_objective_flat(x0)
        if current_area > best_area:
            best_area = current_area
            best_bary = init_bary.copy()

        # Nelder-Mead optimization
        try:
            result = minimize(
                _objective_flat,
                x0,
                method='Nelder-Mead',
                options=nm_options
            )
            if result.success or result.fun < _objective_flat(x0):
                opt_bary = _project_bary(result.x.reshape(N, 3))
                opt_area = _min_area_bary(opt_bary)
                if opt_area > best_area:
                    best_area = opt_area
                    best_bary = opt_bary.copy()
        except Exception:
            pass

        # If time remains, do a second pass with slightly perturbed start
        if not _is_time_up():
            for _ in range(2):
                if _is_time_up():
                    break
                perturb = best_bary + rng.normal(0, 0.02, size=(N, 3))
                perturb = _project_bary(perturb)
                x0p = perturb.flatten()
                try:
                    result2 = minimize(
                        _objective_flat,
                        x0p,
                        method='Nelder-Mead',
                        options=nm_options
                    )
                    if result2.fun < _objective_flat(x0p):
                        opt_bary2 = _project_bary(result2.x.reshape(N, 3))
                        opt_area2 = _min_area_bary(opt_bary2)
                        if opt_area2 > best_area:
                            best_area = opt_area2
                            best_bary = opt_bary2.copy()
                except Exception:
                    pass

    if best_bary is None:
        # Fallback: uniform random
        bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(bary)

    cart = _bary_to_cart(best_bary)
    return cart
