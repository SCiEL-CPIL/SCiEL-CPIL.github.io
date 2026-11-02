"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Physics-based discrete Lagrangian relaxation.
- Represent circles as particles.
- Apply repulsive forces proportional to overlap (like springs).
- Iteratively: (1) grow radii to max feasible, (2) relax positions via force-based
  displacement, (3) repeat until convergence or time budget.
- Use multiple diverse initializations and keep the best valid packing.
"""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 26
DEADLINE = 142.0  # safety margin below 150s kill
TOL = 1e-8


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _force_relax(pos, rad, steps=200, dt=0.01, k=1.0):
    """
    Physics-based relaxation: apply repulsive forces for overlapping circles
    and attractive forces toward the center for circles too close to walls.
    Returns new positions.
    """
    n = len(rad)
    pos = pos.copy()
    for step in range(steps):
        forces = np.zeros((n, 2))
        for i in range(n):
            # Wall forces: push away from walls
            if pos[i, 0] < rad[i]:
                forces[i, 0] += k * (rad[i] - pos[i, 0])
            elif pos[i, 0] > 1.0 - rad[i]:
                forces[i, 0] -= k * (pos[i, 0] - (1.0 - rad[i]))
            if pos[i, 1] < rad[i]:
                forces[i, 1] += k * (rad[i] - pos[i, 1])
            elif pos[i, 1] > 1.0 - rad[i]:
                forces[i, 1] -= k * (pos[i, 1] - (1.0 - rad[i]))

            # Pairwise repulsive forces
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx, dy = 0.01, 0.01
                    d = np.hypot(dx, dy)
                overlap = rad[i] + rad[j] - d
                if overlap > 0:
                    fx = k * overlap * dx / d
                    fy = k * overlap * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Update positions
        pos += dt * forces
        # Clip to [0,1]
        pos = np.clip(pos, 0.0, 1.0)

    return pos


def _relax_and_grow(pos, rad, time_budget=5.0):
    """
    Alternate between force-based relaxation and radius growth.
    """
    t0 = time.time()
    for it in range(200):
        if time.time() - t0 > time_budget:
            break
        # Grow radii to max
        rad = _max_radius(pos, rad)
        # Relax positions
        pos = _force_relax(pos, rad, steps=50, dt=0.02, k=2.0)
    # Final grow
    rad = _max_radius(pos, rad)
    return pos, rad


def _polish(pos, rad, time_budget=3.0):
    """
    Final polish: smaller steps for fine-tuning.
    """
    t0 = time.time()
    for it in range(100):
        if time.time() - t0 > time_budget:
            break
        rad = _max_radius(pos, rad)
        pos = _force_relax(pos, rad, steps=20, dt=0.005, k=1.0)
    rad = _max_radius(pos, rad)
    return pos, rad


def _init_grid(side):
    """Grid initialization."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    rad = np.full(N, 0.5 / side - 1e-4)
    for _ in range(5):
        rad = _max_radius(pos, rad)
    return pos, rad


def _init_hex():
    """Hexagonal lattice initialization."""
    hex_pos = []
    r_hex = 0.13
    dx = 2 * r_hex * 0.95  # slightly less to allow growth
    dy = r_hex * np.sqrt(3) * 0.95
    for row in range(10):
        for col in range(10):
            x = col * dx + (row % 2) * dx / 2 + 0.12
            y = row * dy + 0.12
            if 0.03 < x < 0.97 and 0.03 < y < 0.97:
                hex_pos.append((x, y))
    hex_pos = hex_pos[:N]
    if len(hex_pos) < N:
        # Fill remaining with random
        rng = np.random.default_rng(123)
        while len(hex_pos) < N:
            hex_pos.append((rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1))
    pos = np.array(hex_pos, dtype=float)
    rad = np.full(N, 0.1)
    rad = _max_radius(pos, rad)
    return pos, rad


def _init_random(rng, n=26):
    """Random initialization."""
    pos = rng.random((n, 2)) * 0.7 + 0.15
    rad = np.full(n, 0.05)
    rad = _max_radius(pos, rad)
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            # Try to fix by reducing radii
            rad = np.maximum(rad * 0.998, 0)
            arr = np.column_stack([pos, rad])
            if not _valid_check(arr):
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr.copy()

    # ---- Phase A: Generate diverse initializations ----
    starts = []

    # Grids
    for side in range(3, 9):
        if time.time() > DEADLINE - 40:
            break
        pos, rad = _init_grid(side)
        starts.append((pos.copy(), rad.copy()))
        # Perturbed grids
        for k in range(3):
            if time.time() > DEADLINE - 40:
                break
            p0 = pos + 0.04 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = _max_radius(p0, np.full(N, 0.5 / side))
            starts.append((p0, r0))

    # Hexagonal
    if time.time() <= DEADLINE - 40:
        pos, rad = _init_hex()
        starts.append((pos, rad))

    # Random
    for k in range(10):
        if time.time() > DEADLINE - 40:
            break
        pos, rad = _init_random(rng)
        starts.append((pos, rad))

    # ---- Phase B: Relax and grow from each start ----
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 25:
            break
        remaining = DEADLINE - 25 - time.time()
        budget = min(6.0, max(1.5, remaining / max(1, len(starts) - idx)))
        pos_r, rad_r = _relax_and_grow(pos0, rad0, time_budget=budget)
        consider(pos_r, rad_r)

    # ---- Phase C: Polish the best solution ----
    if best is not None:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        pos_p, rad_p = _polish(pos_b, rad_b, time_budget=4.0)
        consider(pos_p, rad_p)

    # ---- Phase D: Local search perturbation on best ----
    if best is not None and time.time() < DEADLINE - 12:
        for k in range(15):
            if time.time() > DEADLINE - 6:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            n_perturb = rng.integers(3, 9)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.015 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.03, 0.97)
            rad_p = _max_radius(pos_p, rad_p)
            pos_f, rad_f = _polish(pos_p, rad_p, time_budget=2.0)
            consider(pos_f, rad_f)

    # ---- Phase E: If still time, do a global restart with best as seed ----
    if best is not None and time.time() < DEADLINE - 15:
        # Use best as a perturbed seed for a fresh relaxation
        for k in range(5):
            if time.time() > DEADLINE - 8:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            # Larger perturbation
            pos_p += 0.03 * (rng.random((N, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.05, 0.95)
            rad_p = _max_radius(pos_p, rad_p)
            pos_f, rad_f = _relax_and_grow(pos_p, rad_p, time_budget=3.0)
            consider(pos_f, rad_f)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
