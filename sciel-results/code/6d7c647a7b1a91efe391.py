"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    # Check radii positive
    if np.any(r <= 0):
        return None
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    # Check disjoint interiors (vectorized)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.02 + 0.03 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _penalty_objective(circles):
    """Objective to minimize: -sum(r) + penalty for violations."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(r)
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    boundary_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    obj += 100 * np.sum(boundary_viol)
    
    # Penalty for circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = max(0, r[i] + r[j] - dist)
            obj += 50 * overlap
    
    return obj


def _local_optimize(circles, time_limit=10, seed=0):
    """Fast local optimization using vectorized pair computations.
    
    Uses a combination of:
    1. Repulsion-based gradient for overlaps
    2. Attraction toward center for boundary
    3. Radius growth when feasible
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    lr = 0.005
    decay = 0.99995
    it = 0
    while True:
        if time.time() - start_time > time_limit:
            break
        if it % 5000 > 0 and time.time() - start_time > time_limit:
            break
        
        # Compute all pairwise distances
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        
        # Overlap amount
        overlap = r[i_idx] + r[j_idx] - dist
        
        # Gradient for positions: push apart when overlapping
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Repulsion from overlaps
        repulsion = np.where(overlap > 0, 50.0 * overlap / dist, 0.0)
        gx[i_idx] -= repulsion * dx / dist
        gy[i_idx] -= repulsion * dy / dist
        gx[j_idx] += repulsion * dx / dist
        gy[j_idx] += repulsion * dy / dist
        
        # Boundary repulsion
        # Push away from walls
        gx += np.where(x < r, 100.0, 0.0) - np.where(1 - x < r, 100.0, 0.0)
        gy += np.where(y < r, 100.0, 0.0) - np.where(1 - y < r, 100.0, 0.0)
        
        # Radius gradient: want to grow, but penalize overlaps
        gr += 1.0  # Want to grow
        # Penalty for boundary violations
        gr -= np.where(x < r, 100.0, 0.0)
        gr -= np.where(1 - x < r, 100.0, 0.0)
        gr -= np.where(y < r, 100.0, 0.0)
        gr -= np.where(1 - y < r, 100.0, 0.0)
        # Penalty for overlaps (each circle contributes to its overlaps)
        overlap_pen_i = np.zeros(n)
        overlap_pen_j = np.zeros(n)
        np.add.at(overlap_pen_i, i_idx, np.where(overlap > 0, 50.0, 0.0))
        np.add.at(overlap_pen_j, j_idx, np.where(overlap > 0, 50.0, 0.0))
        gr -= overlap_pen_i + overlap_pen_j
        
        # Update
        x += lr * gx
        y += lr * gy
        r += lr * gr
        
        # Clamp
        r = np.maximum(r, 0.001)
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        
        # Ensure within bounds
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        lr *= decay
        it += 1
        
        # Check score periodically
        if it % 200 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                best_score = score
                best_circles = current_circles.copy()
    
    # Final check
    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles
    
    return best_circles, best_score


def _simulated_annealing(circles, max_iter=50000, t_init=0.05, t_min=0.001,
                          time_limit=80, seed=0):
    """Simulated annealing with fast scoring to optimize circle packing."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _fast_score(cx, cy, cr):
        """Fast validity check returning score or None."""
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_score = _fast_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    current_score = best_score
    if current_score is None:
        current_score = 0.0
    
    t = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb a random circle
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.5:
            # Move position
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.8:
            # Change radius
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            # Adjust two circles
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.008 * rng.randn()
            new_r[i] += delta_r
            new_r[j] -= delta_r
            new_r[i] = max(0.005, new_r[i])
            new_r[j] = max(0.005, new_r[j])
        
        new_score = _fast_score(new_x, new_y, new_r)
        
        if new_score is not None:
            if new_score > current_score:
                x, y, r = new_x, new_y, new_r
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_circles = np.column_stack([x, y, r]).copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / max(t, 1e-10))
                    if rng.rand() < prob:
                        x, y, r = new_x, new_y, new_r
                        current_score = new_score
        else:
            pass
        
        t *= cooling
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 120  # seconds
    safety_margin = 15  # seconds
    
    best_circles = None
    best_score = -1.0
    
    # Multi-start with different initializations
    initializations = []
    
    # Grid initialization
    initializations.append(_grid_initial(N))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789]:
        initializations.append(_random_initial(N, seed=seed))
    
    # Time per initialization phase
    n_inits = len(initializations)
    time_per_init = (time_budget - safety_margin) / (n_inits * 3)  # 3 phases per init
    
    for init_idx, init_circles in enumerate(initializations):
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining < 5:
            break
        
        # Phase 1: Local optimization
        t_alloc1 = min(time_per_init, remaining * 0.4)
        if t_alloc1 > 2:
            loc_circles, loc_score = _local_optimize(init_circles, time_limit=t_alloc1, seed=init_idx)
            if loc_score is not None and loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles
        
        # Phase 2: Simulated annealing
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining < 5:
            break
        t_alloc2 = min(time_per_init, remaining * 0.4)
        if t_alloc2 > 2:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else init_circles,
                max_iter=50000, t_init=0.02, t_min=0.0003,
                time_limit=t_alloc2, seed=init_idx + 100)
            if sa_score is not None and sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
        
        # Phase 3: Final local polish
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining < 3:
            break
        t_alloc3 = min(time_per_init * 0.5, remaining * 0.3)
        if t_alloc3 > 1:
            pol_circles, pol_score = _local_optimize(
                best_circles if best_circles is not None else init_circles,
                time_limit=t_alloc3, seed=init_idx + 200)
            if pol_score is not None and pol_score > best_score:
                best_score = pol_score
                best_circles = pol_circles
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
