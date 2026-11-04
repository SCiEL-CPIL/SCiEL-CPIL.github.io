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

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

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


def _force_directed_packing(xy, r, deadline, seed=42):
    """
    Force-directed optimization:
    1. Define a smooth potential energy function for overlaps and wall violations.
    2. Use gradient descent with adaptive step size to minimize energy.
    3. Simultaneously grow radii as the system stabilizes.
    
    This treats the packing as a physical system where circles repel each other
    and are confined by walls, seeking an equilibrium state of maximum total radius.
    """
    rng = np.random.RandomState(seed)
    n = len(xy)
    cur_xy = xy.copy()
    cur_r = r.copy()
    
    best_xy = cur_xy.copy()
    best_r = cur_r.copy()
    best_total, best_valid = _eval_packing(cur_xy, cur_r)
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Parameters
    k_repel = 10.0      # Repulsion strength
    k_wall = 10.0       # Wall repulsion strength
    lr = 0.01           # Learning rate
    growth_rate = 0.0001 # How fast we try to grow radii
    max_iter = 5000
    cooldown = 0.999    # Learning rate decay

    T = 1.0 # Temperature for acceptance of worse states (if we add SA flavor)
    
    # Initial energy calculation helper
    def _energy(xy_, r_):
        e = 0.0
        # Wall energy
        for i in range(n):
            x, y, ri = xy_[i, 0], xy_[i, 1], r_[i]
            # Soft violation: how much is it outside?
            out_x = max(0, ri - x, ri - (1.0 - x))
            out_y = max(0, ri - y, ri - (1.0 - y))
            e += k_wall * (out_x**2 + out_y**2)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_[i, 0] - xy_[j, 0]
                dy = xy_[i, 1] - xy_[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r_[i] + r_[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    e += k_repel * overlap**2
        return e

    cur_e = _energy(cur_xy, cur_r)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute gradients numerically or analytically?
        # Analytical is faster.
        grad_xy = np.zeros_like(cur_xy)
        grad_r = np.zeros_like(cur_r)
        
        # Wall gradients
        for i in range(n):
            x, y, ri = cur_xy[i, 0], cur_xy[i, 1], cur_r[i]
            # dE/dx for wall
            if x < ri: # Left wall violation
                grad_xy[i, 0] -= 2 * k_wall * (ri - x)
            elif x > 1.0 - ri: # Right wall
                grad_xy[i, 0] += 2 * k_wall * (x - (1.0 - ri))
            
            if y < ri: # Bottom wall
                grad_xy[i, 1] -= 2 * k_wall * (ri - y)
            elif y > 1.0 - ri: # Top wall
                grad_xy[i, 1] += 2 * k_wall * (y - (1.0 - ri))
            
            # dE/dr for wall
            # If x < ri: E ~ (ri-x)^2 -> dE/dr = 2(ri-x)
            # If x > 1-ri: E ~ (x-(1-ri))^2 = (ri + x - 1)^2 -> dE/dr = 2(ri + x - 1)
            # Note: usually only one wall is violated at a time significantly, but let's sum.
            if x < ri:
                grad_r[i] += 2 * k_wall * (ri - x)
            if x > 1.0 - ri:
                grad_r[i] += 2 * k_wall * (x - (1.0 - ri))
            if y < ri:
                grad_r[i] += 2 * k_wall * (ri - y)
            if y > 1.0 - ri:
                grad_r[i] += 2 * k_wall * (y - (1.0 - ri))
                
        # Pairwise gradients
        for i in range(n):
            for j in range(i + 1, n):
                dx = cur_xy[i, 0] - cur_xy[j, 0]
                dy = cur_xy[i, 1] - cur_xy[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    dist = 1e-6
                    dx = 1e-6
                    dy = 0
                else:
                    dist = np.sqrt(dist_sq)
                
                min_dist = cur_r[i] + cur_r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # E = k * overlap^2
                    # dE/dx_i = -2 k overlap * (dx/dist)
                    # dE/dx_j = +2 k overlap * (dx/dist)
                    factor = 2 * k_repel * overlap / dist
                    grad_xy[i, 0] -= factor * dx
                    grad_xy[i, 1] -= factor * dy
                    grad_xy[j, 0] += factor * dx
                    grad_xy[j, 1] += factor * dy
                    
                    # dE/dr_i = 2 k overlap
                    # dE/dr_j = 2 k overlap
                    grad_r[i] += 2 * k_repel * overlap
                    grad_r[j] += 2 * k_repel * overlap
        
        # Update
        # We want to minimize energy, but also maximize radii.
        # Objective: Minimize E - lambda * sum(r)? 
        # Or simpler: Gradient descent on E, then increase r slightly.
        
        # Step 1: Move positions to reduce energy
        cur_xy -= lr * grad_xy
        cur_r -= lr * grad_r
        
        # Step 2: Grow radii (the "packing" part)
        # We grow radii uniformly by a small amount, relying on the repulsion
        # to push them apart or shrink them if they hit walls.
        # Actually, growing r increases energy if overlapping. 
        # So we do a projected gradient step on the objective:
        # f(xy, r) = E(xy, r) - mu * sum(r)
        # Grad_r = grad_r_E - mu
        # Let's use mu = 1.0 (weight of radius maximization)
        
        # Re-compute effective gradient for r including the -sum(r) term
        # We already computed grad_r from energy.
        # Effective grad for r: grad_r - 1.0
        # We already applied the energy grad to r above. 
        # Let's do a separate step for the "growth" objective.
        # This is a bit ad-hoc. Better:
        # Update xy: xy -= lr * grad_xy_E
        # Update r: r -= lr * (grad_r_E - 1.0)  <-- Note: we want to maximize r, so minimize -r.
        
        # Let's redo the update properly.
        # We already subtracted lr * grad_xy from xy.
        # We already subtracted lr * grad_r from r.
        # We need to add the -1.0 term to the r gradient.
        # So effectively: r += lr * 1.0
        
        cur_r += lr * 1.0
        
        # Clamp
        cur_xy = np.clip(cur_xy, 0.0, 1.0)
        cur_r = np.clip(cur_r, 0.001, 0.5)
        
        # Decay learning rate
        lr *= cooldown
        
        # Check validity and best
        total, valid = _eval_packing(cur_xy, cur_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
            
        # If energy is very low, we might be stuck. Add noise?
        # For now, keep it simple.
        
    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n, beta=50.0):
    """
    Smooth objective for final gradient polish.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n, beta=50.0):
    """Gradient of smooth objective."""
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        if min_idx == 0:
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-10:
                continue
            factor = 100.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0
    return grad


def _gradient_polish(xy, r, deadline):
    """Final gradient-based polish."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    beta = 100.0

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, beta)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search (fast, good starting point)
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Force-Directed Optimization from multiple seeds
    # This is the new structural change.
    fd_seeds = [42, 123, 777, 2024, 555, 999]
    
    if best_valid and best_xy is not None:
        # FD from the best grid solution
        out_xy, out_r, total, valid = _force_directed_packing(best_xy.copy(), best_r.copy(), deadline, seed=42)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    for seed in fd_seeds:
        if time.time() > deadline - 20:
            break
        # Random init
        rng = np.random.RandomState(seed)
        xy = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.03, 0.08, size=N)
        
        out_xy, out_r, total, valid = _force_directed_packing(xy, r, deadline, seed=seed)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Local search polish on the FD result
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline, grid_size=100)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

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
