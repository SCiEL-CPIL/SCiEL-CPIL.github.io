"""Circle packing in unit square; EVOLVE-BLOCK rewritten with repulsion-based force-directed optimization."""
import numpy as np
import time

N = 26
DEADLINE = 150.0
SAFETY = 12.0          # stop this many seconds before the hard limit
START_TIME = time.time()


def _remaining():
    return DEADLINE - SAFETY - (time.time() - START_TIME)


def _max_radius_for_circle(i, pos, rad):
    """Max radius for circle i given positions pos (n,2) and radii rad (n,)."""
    xi, yi = pos[i]
    r = min(xi, yi, 1.0 - xi, 1.0 - yi)
    for j in range(N):
        if j == i:
            continue
        d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
        r = min(r, d - rad[j])
    return max(r, 0.0)


def _eval_packing(pos, rad):
    """Return (sum_radii, valid)."""
    for i in range(N):
        if rad[i] < -1e-9 or pos[i, 0] < -1e-9 or pos[i, 1] < -1e-9 \
                or pos[i, 0] > 1.0 + 1e-9 or pos[i, 1] > 1.0 + 1e-9:
            return 0.0, False
        if pos[i, 0] + rad[i] > 1.0 + 1e-7 or pos[i, 1] + rad[i] > 1.0 + 1e-7 \
                or pos[i, 0] - rad[i] < -1e-7 or pos[i, 1] - rad[i] < -1e-7:
            return 0.0, False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-7:
                return 0.0, False
    return float(np.sum(rad)), True


def _grid_init():
    """Dense structured grid starting layout."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            pos[idx, 0] = (i + 0.5) / side
            pos[idx, 1] = (j + 0.5) / side
            idx += 1
    rad = np.array([_max_radius_for_circle(i, pos, np.zeros(N)) for i in range(N)])
    return pos, rad


def _repulsion_optimize(pos, rad, deadline_check, max_iters=5000, lr_init=0.02, decay=0.999):
    """
    Force-directed optimization: circles repel each other, walls attract to keep in bounds.
    Uses momentum to escape local minima.
    """
    pos = pos.copy()
    rad = rad.copy()
    vel = np.zeros_like(pos)
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_sum = float(np.sum(rad))
    
    lr = lr_init
    momentum = 0.9
    
    for it in range(max_iters):
        if deadline_check():
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Wall forces: push circles away from boundaries
        for i in range(N):
            x, y = pos[i]
            # Left wall
            if x < 0.05:
                forces[i, 0] += (0.05 - x) * 50.0
            # Right wall
            if x > 0.95:
                forces[i, 0] -= (x - 0.95) * 50.0
            # Bottom wall
            if y < 0.05:
                forces[i, 1] += (0.05 - y) * 50.0
            # Top wall
            if y > 0.95:
                forces[i, 1] -= (y - 0.95) * 50.0
        
        # Repulsion forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    # Random nudge if exactly overlapping
                    dx = np.random.randn() * 1e-3
                    dy = np.random.randn() * 1e-3
                    d = np.hypot(dx, dy)
                
                # Repulsion strength increases when circles are close
                # Use a soft repulsion potential
                r_i = rad[i]
                r_j = rad[j]
                min_dist = r_i + r_j
                
                if d < min_dist * 1.5:
                    # Strong repulsion when overlapping or very close
                    strength = 100.0 * (min_dist / max(d, 1e-6)) ** 2
                    fx = strength * dx / d
                    fy = strength * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                else:
                    # Weaker long-range repulsion to spread circles
                    strength = 10.0 * (min_dist / d) ** 4
                    fx = strength * dx / d
                    fy = strength * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update velocities with momentum
        vel = momentum * vel + lr * forces
        pos += vel
        
        # Clamp positions to [0, 1]
        pos = np.clip(pos, 0.0, 1.0)
        
        # Recompute radii based on new positions
        for i in range(N):
            rad[i] = _max_radius_for_circle(i, pos, rad)
        
        # Decay learning rate
        lr *= decay
        
        # Track best
        current_sum = float(np.sum(rad))
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_rad = rad.copy()
    
    return best_pos, best_rad, best_sum


def _gradient_polish(pos, rad, deadline_check):
    """Polish radii by repeatedly recomputing max radius (radii depend on positions)."""
    pos = pos.copy()
    rad = rad.copy()
    for _ in range(50):
        if deadline_check():
            break
        changed = False
        for i in range(N):
            new_r = _max_radius_for_circle(i, pos, rad)
            if abs(new_r - rad[i]) > 1e-12:
                rad[i] = new_r
                changed = True
        if not changed:
            break
    return pos, rad


def construct():
    deadline_check = lambda: _remaining() < 0.0

    # Start from grid
    pos, rad = _grid_init()
    best_pos, best_rad, best_sum = pos.copy(), rad.copy(), float(np.sum(rad))

    # Phase 1: Repulsion-based optimization with multiple restarts
    np.random.seed(42)
    for restart in range(3):
        if _remaining() < 30.0:
            break
        
        # Jitter initial positions for diversity
        jittered_pos = pos.copy() + np.random.randn(N, 2) * 0.02
        jittered_pos = np.clip(jittered_pos, 0.0, 1.0)
        jittered_rad = np.array([_max_radius_for_circle(i, jittered_pos, np.zeros(N)) for i in range(N)])
        
        # Run repulsion optimization
        new_pos, new_rad, new_sum = _repulsion_optimize(
            jittered_pos, jittered_rad, deadline_check,
            max_iters=3000, lr_init=0.03, decay=0.9995
        )
        
        if new_sum > best_sum:
            best_pos, best_rad, best_sum = new_pos.copy(), new_rad.copy(), new_sum
        
        # Use best as starting point for next restart
        pos = best_pos.copy()
        rad = best_rad.copy()

    # Phase 2: Gradient polish
    pos, rad = _gradient_polish(best_pos, best_rad, deadline_check)
    s = float(np.sum(rad))
    if s > best_sum:
        best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s

    # Phase 3: Another repulsion pass with smaller steps for fine-tuning
    if _remaining() > 15.0:
        new_pos, new_rad, new_sum = _repulsion_optimize(
            best_pos, best_rad, deadline_check,
            max_iters=2000, lr_init=0.01, decay=0.9998
        )
        if new_sum > best_sum:
            best_pos, best_rad, best_sum = new_pos.copy(), new_rad.copy(), new_sum
        pos, rad = _gradient_polish(best_pos, best_rad, deadline_check)
        s = float(np.sum(rad))
        if s > best_sum:
            best_pos, best_rad, best_sum = pos.copy(), rad.copy(), s

    # Final validation & clip
    out = np.zeros((N, 3))
    for i in range(N):
        x = float(np.clip(best_pos[i, 0], 0.0, 1.0))
        y = float(np.clip(best_pos[i, 1], 0.0, 1.0))
        r = float(best_rad[i])
        # ensure inside
        r = min(r, x, y, 1.0 - x, 1.0 - y)
        r = max(r, 0.0)
        out[i] = (x, y, r)

    # verify validity; if invalid, fall back to safe grid
    _, valid = _eval_packing(out[:, :2], out[:, 2])
    if not valid:
        # shrink radii slightly to guarantee validity
        for i in range(N):
            out[i, 2] *= 0.98
        _, valid2 = _eval_packing(out[:, :2], out[:, 2])
        if not valid2:
            # ultimate fallback: trivial grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            out = np.array(cells[:N])

    return out
