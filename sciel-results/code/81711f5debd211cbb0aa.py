import numpy as np
import time
from scipy.spatial.distance import cdist

N = 26

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary_i, min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)

def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))

def _eval_packing(centers, radii, eps=1e-9):
    """Evaluate a packing: return (sum_of_radii, is_valid)."""
    sum_r = float(np.sum(radii))
    
    # Check boundaries
    for i in range(N):
        if centers[i][0] < -eps or centers[i][0] > 1 + eps:
            return sum_r, False
        if centers[i][1] < -eps or centers[i][1] > 1 + eps:
            return sum_r, False
        if radii[i] < -eps:
            return sum_r, False
        if radii[i] > centers[i][0] + eps:
            return sum_r, False
        if radii[i] > 1 - centers[i][0] + eps:
            return sum_r, False
        if radii[i] > centers[i][1] + eps:
            return sum_r, False
        if radii[i] > 1 - centers[i][1] + eps:
            return sum_r, False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - eps:
                return sum_r, False
    
    return sum_r, True

def _best_fit_local_search(pos, n_iter=2000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search: for each circle, search a fine grid
    around its current position to find the best new location.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    base_step = 0.015
    grid_res = 5

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)
        
        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            
            best_new_pos = None
            best_new_obj = current_obj

            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_pos = new_pos

            # Also try some random perturbations
            for _ in range(8):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_pos = new_pos

            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            # Try with smaller random steps
            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                x_i, y_i = current_pos[i]
                dx = rng.uniform(-step * 0.5, step * 0.5)
                dy = rng.uniform(-step * 0.5, step * 0.5)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(x_i + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(y_i + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-14:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                base_step *= 0.6
                if base_step < 0.0003:
                    break

    return current_pos, current_obj

def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """
    Simulated annealing to maximize sum of radii.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        t = t_start * (t_end / t_start) ** (it / n_iter)

        i = rng.randint(n)
        dx = rng.uniform(-0.05, 0.05)
        dy = rng.uniform(-0.05, 0.05)
        
        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
        
        new_obj = _objective(new_pos)
        delta = new_obj - current_obj
        
        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_pos = current_pos.copy()

    return best_pos, best_obj

def _grid_init():
    """Create initial grid packing."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            centers.append(((i + 0.5) / side, (j + 0.5) / side))
    return np.array(centers[:N])

def _random_init(seed=42):
    """Create random initial packing."""
    rng = np.random.RandomState(seed)
    centers = np.random.rand(N, 2)
    return centers

def _hex_init():
    """Create hexagonal-like initial packing."""
    centers = []
    r0 = 0.09
    dy = r0 * 2 * 0.866
    for row in range(6):
        x_offset = 0 if row % 2 == 0 else r0
        for col in range(5):
            x = r0 + col * 2 * r0 + x_offset
            y = r0 + row * dy
            if x < 1 and y < 1:
                centers.append((x, y))
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    rng = np.random.RandomState(123)
    while len(centers) < N:
        centers.append((rng.rand(), rng.rand()))
    
    return np.array(centers[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit)
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid init
    inits.append(_grid_init())
    
    # Random inits
    for seed in [42, 123, 456, 789, 1011, 2020, 3030, 4040]:
        inits.append(_random_init(seed))
    
    # Hex init
    inits.append(_hex_init())
    
    # Run optimization on each
    for idx, centers in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            # First, run local search
            c_opt, score_ls = _best_fit_local_search(centers, max_iters=1000, seed=idx, deadline=deadline)
            
            # Then, refine with simulated annealing
            c_opt, score_sa = _simulated_annealing(c_opt, n_iter=10000, seed=idx + 100, deadline=deadline)
            
            # Compute radii for final positions
            radii = _max_radius_for_positions(c_opt)
            
            # Validate
            valid_score, is_valid = _eval_packing(c_opt, radii)
            if is_valid and valid_score > best_score:
                best_score = valid_score
                best_centers = c_opt.copy()
                best_radii = radii.copy()
                
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return np.column_stack((best_centers, best_radii))
