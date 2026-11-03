"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

            # Refinement: if we moved, do a finer local search around the new position
            if new_r > old_r + 1e-10:
                # Create a finer grid around the best position
                step = 1.0 / grid_size
                fine_gx = np.linspace(max(0, new_x - 3*step), min(1, new_x + 3*step), 20)
                fine_gy = np.linspace(max(0, new_y - 3*step), min(1, new_y + 3*step), 20)
                FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
                fine_candidates = np.column_stack([FGX.ravel(), FGY.ravel()])
                FM = len(fine_candidates)
                
                if FM > 0:
                    fine_wall_clear = np.minimum(np.minimum(fine_candidates[:, 0], 1.0 - fine_candidates[:, 0]),
                                                 np.minimum(fine_candidates[:, 1], 1.0 - fine_candidates[:, 1]))
                    if len(others_xy) > 0:
                        fdx = fine_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                        fdy = fine_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                        fdists = np.sqrt(fdx * fdx + fdy * fdy)
                        fcircle_clear = np.min(fdists - others_r[np.newaxis, :], axis=1)
                        fr_candidates = np.minimum(fine_wall_clear, fcircle_clear)
                    else:
                        fr_candidates = fine_wall_clear.copy()
                    
                    fidx = int(np.argmax(fr_candidates))
                    fnew_x, fnew_y = fine_candidates[fidx, 0], fine_candidates[fidx, 1]
                    fnew_r = fr_candidates[fidx]
                    
                    if fnew_r > new_r + 1e-10:
                        xy[i, 0] = fnew_x
                        xy[i, 1] = fnew_y
                        r[i] = fnew_r

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _sa_packing(xy, r, deadline, seed=42):
    """
    Simulated annealing for circle packing.
    State: (xy, r). Moves: perturb position/radius of 1-2 circles.
    Accepts worse moves with probability exp(-delta/T).
    """
    n = len(xy)
    rng = np.random.RandomState(seed)
    
    current_xy = xy.copy()
    current_r = r.copy()
    current_total, current_valid = _eval_packing(current_xy, current_r)
    if not current_valid:
        current_total = 0.0
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total
    best_valid = current_valid
    
    T0 = 0.05  # initial temperature
    T_end = 0.001
    cooling = 0.995
    
    # Precompute indices for pairwise checks
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = T0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and T > T_end and no_improve_count < max_no_improve:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Perturb position of one circle
            i = rng.randint(n)
            dx = rng.normal(0, T * 2)
            dy = rng.normal(0, T * 2)
            new_xy = current_xy.copy()
            new_xy[i, 0] += dx
            new_xy[i, 1] += dy
            new_r = current_r.copy()
        elif move_type < 0.7:
            # Perturb radius of one circle
            i = rng.randint(n)
            dr = rng.normal(0, T * 0.5)
            new_xy = current_xy.copy()
            new_r = current_r.copy()
            new_r[i] = max(0.001, min(0.49, current_r[i] + dr))
        elif move_type < 0.9:
            # Perturb position of two circles
            i, j = rng.choice(n, size=2, replace=False)
            dx_i = rng.normal(0, T * 2)
            dy_i = rng.normal(0, T * 2)
            dx_j = rng.normal(0, T * 2)
            dy_j = rng.normal(0, T * 2)
            new_xy = current_xy.copy()
            new_xy[i, 0] += dx_i
            new_xy[i, 1] += dy_i
            new_xy[j, 0] += dx_j
            new_xy[j, 1] += dy_j
            new_r = current_r.copy()
        else:
            # Swap positions of two circles (keep radii)
            i, j = rng.choice(n, size=2, replace=False)
            new_xy = current_xy.copy()
            new_xy[[i, j]] = current_xy[[j, i]]
            new_r = current_r.copy()
        
        # Clip to valid range
        for k in range(n):
            new_xy[k, 0] = np.clip(new_xy[k, 0], new_r[k], 1.0 - new_r[k])
            new_xy[k, 1] = np.clip(new_xy[k, 1], new_r[k], 1.0 - new_r[k])
        
        new_total, new_valid = _eval_packing(new_xy, new_r)
        if not new_valid:
            new_total = -10.0  # heavy penalty
        
        delta = new_total - current_total
        
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_xy = new_xy
            current_r = new_r
            current_total = new_total
            current_valid = new_valid
            
            if current_total > best_total:
                best_total = current_total
                best_valid = current_valid
                best_xy = current_xy.copy()
                best_r = current_r.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        T *= cooling
    
    return best_xy, best_r, best_total, best_valid


def _refine_with_sa(xy, r, deadline):
    """
    Run SA from multiple random restarts and return the best result.
    """
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    n_restarts = 5
    per_restart_time = (deadline - time.time()) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create perturbed version of input
        rng = np.random.RandomState(1000 + restart)
        perturbed_xy = xy.copy()
        perturbed_r = r.copy()
        
        # Add small random perturbation
        for i in range(len(xy)):
            perturbed_xy[i, 0] += rng.normal(0, 0.02)
            perturbed_xy[i, 1] += rng.normal(0, 0.02)
            perturbed_r[i] = max(0.001, perturbed_r[i] * (1 + rng.normal(0, 0.1)))
        
        # Clip
        for i in range(len(perturbed_xy)):
            perturbed_xy[i, 0] = np.clip(perturbed_xy[i, 0], perturbed_r[i], 1.0 - perturbed_r[i])
            perturbed_xy[i, 1] = np.clip(perturbed_xy[i, 1], perturbed_r[i], 1.0 - perturbed_r[i])
        
        sub_deadline = min(deadline, time.time() + per_restart_time * 0.9)
        sa_xy, sa_r, sa_total, sa_valid = _sa_packing(perturbed_xy, perturbed_r, sub_deadline, seed=2000 + restart)
        
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_valid = True
            best_xy = sa_xy
            best_r = sa_r
    
    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + local search
    for seed in range(8):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Simulated annealing from multiple restarts
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            sa_xy, sa_r, sa_total, sa_valid = _refine_with_sa(best_xy, best_r, deadline - 5)
            if sa_valid and sa_total > best_total:
                best_total = sa_total
                best_xy = sa_xy
                best_r = sa_r

        # Phase 4: Final local search polish from SA result
        if best_valid and best_xy is not None and time.time() < deadline - 3:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
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
# EVOLVE-BLOCK-END
