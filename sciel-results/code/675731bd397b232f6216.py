"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Force-directed (molecular dynamics) energy minimization with
simulated-annealing-style temperature schedule.  Circles repel each other
and walls via soft potentials; a spring drives each radius toward its
current maximum clearance.  Velocity Verlet integration + adaptive step
size.  Multiple restarts with different seeds and initial configurations.
"""

import numpy as np
import time
import random

N = 26


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side, n=N):
    """Create a grid of up to n circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.003
                positions.append([x, y, r])
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:n])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _compute_forces(xy, r, k_repel=50.0, k_wall=100.0, k_spring=10.0, soft=0.02):
    """
    Compute forces on each circle.
    
    Forces:
      1. Pairwise repulsion: F = k_repel * (r_i + r_j - dist) in direction i->j
         (only when overlapping or close, softened)
      2. Wall repulsion: pushes circle away from walls when too close
      3. Radius spring: drives r_i toward its max clearance (computed separately)
    
    Returns:
      F_xy: (n, 2) force on positions
      F_r:  (n,) force on radii (drives toward max clearance)
    """
    n = len(xy)
    F_xy = np.zeros((n, 2))
    F_r = np.zeros(n)

    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-12)
            min_dist = r[i] + r[j]
            # Soft repulsion: strong when overlapping, decays with distance
            overlap = min_dist - dist
            if overlap > -soft:
                # Linear spring-like repulsion
                force_mag = k_repel * max(0.0, overlap)
                if dist > 1e-10:
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                else:
                    fx = 0.0
                    fy = 0.0
                F_xy[i, 0] += fx
                F_xy[i, 1] += fy
                F_xy[j, 0] -= fx
                F_xy[j, 1] -= fy

    # Wall repulsion
    for i in range(n):
        ri = r[i]
        # Left wall
        d_left = xy[i, 0] - ri
        if d_left < soft:
            F_xy[i, 0] += k_wall * max(0.0, soft - d_left)
        # Right wall
        d_right = 1.0 - xy[i, 0] - ri
        if d_right < soft:
            F_xy[i, 0] -= k_wall * max(0.0, soft - d_right)
        # Bottom wall
        d_bot = xy[i, 1] - ri
        if d_bot < soft:
            F_xy[i, 1] += k_wall * max(0.0, soft - d_bot)
        # Top wall
        d_top = 1.0 - xy[i, 1] - ri
        if d_top < soft:
            F_xy[i, 1] -= k_wall * max(0.0, soft - d_top)

    # Radius spring: drive r_i toward max clearance
    for i in range(n):
        # Max clearance from walls
        wall_clear = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        # Max clearance from other circles
        min_circle_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            clearance = dist - r[j]
            if clearance < min_circle_dist:
                min_circle_dist = clearance
        max_r = min(wall_clear, min_circle_dist)
        max_r = max(max_r, 0.0)
        # Spring force toward max_r
        F_r[i] = k_spring * (max_r - r[i])

    return F_xy, F_r


def _md_minimize(xy0, r0, deadline, seed=0, n_steps=5000, dt_init=0.002,
                 k_repel=50.0, k_wall=100.0, k_spring=10.0, soft=0.02,
                 temp_init=0.01, temp_min=1e-6, cooling=0.9995):
    """
    Molecular dynamics minimization with simulated annealing.
    
    Uses velocity Verlet integration. Temperature schedule allows
    occasional uphill moves to escape local minima.
    """
    n = len(xy0)
    rng = np.random.RandomState(seed)
    
    xy = xy0.copy()
    r = r0.copy()
    v_xy = np.zeros((n, 2))
    v_r = np.zeros(n)
    
    dt = dt_init
    temp = temp_init
    
    # Compute initial forces
    F_xy, F_r = _compute_forces(xy, r, k_repel, k_wall, k_spring, soft)
    
    best_total = 0.0
    best_valid = False
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Initial evaluation
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    energy_prev = None
    stall_counter = 0
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Velocity Verlet: half-step velocity
        v_xy += 0.5 * dt * F_xy
        v_r += 0.5 * dt * F_r
        
        # Full-step position
        xy += dt * v_xy
        r += dt * v_r
        
        # Keep within bounds (hard constraints)
        for i in range(n):
            # Clamp radius
            r[i] = max(r[i], 0.005)
            r[i] = min(r[i], 0.5)
            # Clamp position based on radius
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
        
        # Compute new forces
        F_xy_new, F_r_new = _compute_forces(xy, r, k_repel, k_wall, k_spring, soft)
        
        # Half-step velocity with new forces
        v_xy = 0.5 * (v_xy + dt * F_xy_new)
        v_r = 0.5 * (v_r + dt * F_r_new)
        
        # Damping (viscosity)
        damping = 0.98
        v_xy *= damping
        v_r *= damping
        
        # Simulated annealing: occasionally accept uphill moves
        # Compute a proxy "energy" = -sum(r) + penalty
        proxy_energy = -np.sum(r)
        if energy_prev is not None:
            delta_e = proxy_energy - energy_prev
            if delta_e > 0:  # uphill
                # Accept with probability exp(-delta_e / temp)
                if rng.random() > np.exp(-delta_e / max(temp, 1e-10)):
                    # Revert
                    xy -= dt * v_xy
                    r -= dt * v_r
                    v_xy = -v_xy
                    v_r = -v_r
                    # Re-clamp
                    for i in range(n):
                        r[i] = max(r[i], 0.005)
                        r[i] = min(r[i], 0.5)
                        ri = r[i]
                        xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
                        xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
                    F_xy, F_r = _compute_forces(xy, r, k_repel, k_wall, k_spring, soft)
        
        energy_prev = proxy_energy
        
        # Cool down
        temp = max(temp * cooling, temp_min)
        
        # Adaptive step size: reduce if forces are large
        max_force = np.max(np.abs(F_xy_new))
        if max_force > 100.0:
            dt = max(dt * 0.9, 1e-5)
        elif max_force < 1.0:
            dt = min(dt * 1.01, 0.01)
        
        # Update forces for next iteration
        F_xy, F_r = F_xy_new, F_r_new
        
        # Evaluate periodically
        if step % 50 == 0:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total + 1e-8:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
                stall_counter = 0
            elif valid:
                stall_counter += 1
            else:
                # If invalid, nudge back toward validity
                for i in range(n):
                    ri = r[i]
                    xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
                    xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
                # Shrink radii slightly to help validity
                r *= 0.99
                stall_counter += 1
            
            if stall_counter > 50:
                # Restart with small perturbation
                pert = 0.02 * rng.uniform(-1, 1, size=xy.shape)
                xy += pert
                for i in range(n):
                    ri = r[i]
                    xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
                    xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
                v_xy = np.zeros((n, 2))
                v_r = np.zeros(n)
                temp = max(temp, 0.005)
                stall_counter = 0
    
    # Final evaluation
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total + 1e-8:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _local_refine(xy, r, deadline, grid_size=60):
    """
    Quick local refinement: for each circle, find best position on a fine grid
    given others fixed. This serves as a post-MD polish.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 5
    for sweep in range(max_sweeps):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 1.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initializations + MD
    for side in [5, 6, 7]:
        if time.time() > deadline - 30:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _md_minimize(
            xy, r, deadline, seed=side * 100, n_steps=3000)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + MD
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _md_minimize(
            xy, r, deadline, seed=seed * 1000 + 1, n_steps=2000)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Local refinement on best
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _local_refine(
            best_xy.copy(), best_r.copy(), deadline, grid_size=80)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Phase 4: Another MD round from refined state
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        md2_xy, md2_r, md2_total, md2_valid = _md_minimize(
            best_xy.copy(), best_r.copy(), deadline, seed=99999, n_steps=2000)
        if md2_valid and md2_total > best_total + 1e-10:
            best_total = md2_total
            best_xy = md2_xy
            best_r = md2_r

    # Phase 5: Final local refinement
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        ls2_xy, ls2_r, ls2_total, ls2_valid = _local_refine(
            best_xy.copy(), best_r.copy(), deadline, grid_size=100)
        if ls2_valid and ls2_total > best_total + 1e-10:
            best_total = ls2_total
            best_xy = ls2_xy
            best_r = ls2_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
