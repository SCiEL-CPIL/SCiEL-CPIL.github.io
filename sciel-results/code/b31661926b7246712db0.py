"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np

N = 26


def _clamp_positions(pos, radii):
    """Keep circles inside [0,1]^2 by clamping centers to [r, 1-r]."""
    r = radii
    lo = np.maximum(r, 0.0)
    hi = np.minimum(1.0 - r, 1.0)
    # If radius > 0.5, degenerate; clamp to valid range
    hi = np.maximum(hi, lo)
    pos[:, 0] = np.clip(pos[:, 0], lo, hi)
    pos[:, 1] = np.clip(pos[:, 1], lo, hi)
    return pos


def _is_valid(pos, radii, eps=1e-9):
    """Check disjoint interiors and inside unit square."""
    if np.any(radii <= 0.0):
        return False
    if np.any(pos[:, 0] < -eps) or np.any(pos[:, 0] > 1.0 + eps):
        return False
    if np.any(pos[:, 1] < -eps) or np.any(pos[:, 1] > 1.0 + eps):
        return False
    if np.any(radii > 0.5 + eps):
        return False
    # Wall: center must be at least r from each wall
    if np.any(pos[:, 0] + radii > 1.0 + eps) or np.any(pos[:, 0] - radii < -eps):
        return False
    if np.any(pos[:, 1] + radii > 1.0 + eps) or np.any(pos[:, 1] - radii < -eps):
        return False
    # Pairwise: distance >= r_i + r_j
    d = pos[:, 0:1] - pos[0:1, 0]
    dy = pos[:, 1:2] - pos[0:2, 1]
    dist = np.sqrt(d * d + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_gap = np.min(dist - radii[:, None] - radii[None, :])
    return min_gap >= -eps


def _score(pos, radii):
    return float(np.sum(radii))


def _make_grid(n):
    """Initial grid placement of n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side
            cells.append([x, y, r])
    cells = cells[:n]
    pos = np.array([c[:2] for c in cells], dtype=float)
    radii = np.array([c[2] for c in cells], dtype=float)
    return pos, radii


def _pbd_step(pos, radii, dt, dt_r, w_overlap, w_wall, max_iter=8):
    """One physics step: resolve overlaps and grow radii, keep inside box."""
    n = len(radii)
    for _ in range(max_iter):
        # Grow radii a little
        radii = radii * (1.0 + dt_r)
        radii = np.minimum(radii, 0.5)
        # Clamp positions to walls
        pos = _clamp_positions(pos, radii)
        # Resolve pairwise overlaps by pushing apart
        for it in range(3):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    req = radii[i] + radii[j]
                    if dist < req and dist > 1e-12:
                        overlap = (req - dist) * 0.5 * w_overlap
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= nx * overlap
                        pos[i, 1] -= ny * overlap
                        pos[j, 0] += nx * overlap
                        pos[j, 1] += ny * overlap
                    elif dist <= 1e-12:
                        # nudge apart
                        pos[i, 0] -= 1e-4
                        pos[j, 0] += 1e-4
        pos = _clamp_positions(pos, radii)
    return pos, radii


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s

    best_pos, best_radii = None, None
    best_val = -np.inf

    def consider(pos, radii):
        nonlocal best_pos, best_radii, best_val
        if _is_valid(pos, radii):
            v = _score(pos, radii)
            if v > best_val:
                best_val = v
                best_pos = pos.copy()
                best_radii = radii.copy()

    # Initial grid
    pos, radii = _make_grid(N)
    consider(pos, radii)

    # Adaptive growth schedule
    dt_r = 0.002
    w_overlap = 0.9
    n = N

    it = 0
    while time.time() < deadline:
        pos2, radii2 = _pbd_step(pos, radii, dt=0.0, dt_r=dt_r,
                                 w_overlap=w_overlap, w_wall=1.0, max_iter=6)
        pos, radii = pos2, radii2
        consider(pos, radii)
        it += 1

        # Slowly reduce growth rate to converge
        if it % 500 == 0:
            dt_r *= 0.97
            w_overlap = min(w_overlap * 0.99, 0.999)

        # If we're stuck (no improvement), try a small perturbation
        if it % 2000 == 0 and best_val > 0:
            # small random jitter to escape local minima
            jitter = np.random.normal(0, 0.001, size=(n, 2))
            pos_pert = pos + jitter
            radii_pert = radii.copy()
            pos_pert = _clamp_positions(pos_pert, radii_pert)
            # try to resolve overlaps after jitter
            for _ in range(4):
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = pos_pert[j, 0] - pos_pert[i, 0]
                        dy = pos_pert[j, 1] - pos_pert[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        req = radii_pert[i] + radii_pert[j]
                        if dist < req and dist > 1e-12:
                            overlap = (req - dist) * 0.5
                            nx = dx / dist
                            ny = dy / dist
                            pos_pert[i, 0] -= nx * overlap
                            pos_pert[i, 1] -= ny * overlap
                            pos_pert[j, 0] += nx * overlap
                            pos_pert[j, 1] += ny * overlap
                pos_pert = _clamp_positions(pos_pert, radii_pert)
            if _is_valid(pos_pert, radii_pert):
                v = _score(pos_pert, radii_pert)
                if v > best_val:
                    pos, radii = pos_pert, radii_pert
                    consider(pos, radii)
                else:
                    # keep original
                    pass

    if best_pos is None:
        # Fallback to trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    out = np.zeros((N, 3), dtype=float)
    out[:, 0] = best_pos[:, 0]
    out[:, 1] = best_pos[:, 1]
    out[:, 2] = best_radii
    return out
