"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute maximum possible radii for each circle given positions.
    Uses iterative relaxation to ensure strict non-overlap and boundary containment.
    """
    radii = np.minimum(
        np.minimum(pos[:, 0], pos[:, 1]),
        np.minimum(1.0 - pos[:, 0], 1.0 - pos[:, 1])
    )
    radii = np.maximum(radii, 1e-10)
    
    for _ in range(100):
        changed = False
        for i in range(n):
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = max(r_bound, 1e-10)
                changed = True
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    # Reduce the larger radius proportionally
                    total = radii[i] + radii[j]
                    if total > 1e-10:
                        frac_i = radii[i] / total
                        frac_j = radii[j] / total
                        radii[i] = max(1e-10, radii[i] - excess * frac_i)
                        radii[j] = max(1e-10, radii[j] - excess * frac_j)
                    else:
                        radii[i] = max(1e-10, radii[i] - excess / 2.0)
                        radii[j] = max(1e-10, radii[j] - excess / 2.0)
                    changed = True
        if not changed:
            break
    
    return radii

def _score(pos, n):
    """Compute the sum of radii for a given position configuration."""
    radii = _compute_radii(pos, n)
    return np.sum(radii)

def _initial_grid(n):
    """Create initial grid configuration."""
    pos = np.zeros((n, 2))
    rows = 5
    cols = 6
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_hexagonal(n):
    """Create hexagonal packing initial configuration."""
    pos = np.zeros((n, 2))
    spacing = 0.18
    row_spacing = spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            x = spacing / 2 + col * spacing + (row % 2) * spacing / 2
            y = spacing / 2 + row * row_spacing
            if x < 1.0 - spacing / 2 and y < 1.0 - spacing / 2:
                pos[idx] = [x, y]
                idx += 1
            col += 1
            if x >= 1.0 - spacing / 2:
                break
        row += 1
        if y >= 1.0 - spacing / 2:
            break
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_random(n, seed=42):
    """Create random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    min_sep = 0.06
    for i in range(n):
        attempts = 0
        while attempts < 200:
            p = rng.uniform(0.03, 0.97, size=2)
            ok = True
            for j in range(i):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[i] = p
                break
            attempts += 1
        if attempts >= 200:
            pos[i] = rng.uniform(0.03, 0.97, size=2)
    return pos

def _delaunay_init(n):
    """
    Use a greedy Delaunay-like initialization: place circles one by one
    at the point that maximizes the minimum distance to existing circles
    and to the boundary, then assign radii.
    """
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Start with a few anchor points
    anchors = [
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.5], [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5]
    ]
    
    for i in range(min(n, len(anchors))):
        pos[i] = anchors[i]
    
    # Greedily add remaining circles
    for i in range(len(anchors), n):
        best_pos = [0.5, 0.5]
        best_score = -1.0
        # Sample many candidate positions
        for _ in range(500):
            cx = np.random.uniform(0.05, 0.95)
            cy = np.random.uniform(0.05, 0.95)
            # Minimum distance to existing circles
            min_dist = min(
                min(cx, cy, 1.0 - cx, 1.0 - cy),
                min(np.linalg.norm([cx, cy] - pos[j]) for j in range(i))
            )
            if min_dist > best_score:
                best_score = min_dist
                best_pos = [cx, cy]
        pos[i] = best_pos
    
    # Compute radii
    radii = _compute_radii(pos, n)
    return pos, radii

def _local_search(pos, n, deadline, step_size=0.01, n_iter=500):
    """
    Simple local search: try small perturbations to each circle's position.
    """
    current_pos = pos.copy()
    current_score = _score(current_pos, n)
    best_pos = current_pos.copy()
    best_score = current_score
    
    rng = np.random.RandomState(int(time.time() * 1000) % 10000)
    
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        
        # Pick a random circle to move
        i = rng.randint(n)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(new_pos[i, 0] + dx, 0.01, 0.99)
        new_pos[i, 1] = np.clip(new_pos[i, 1] + dy, 0.01, 0.99)
        
        new_score = _score(new_pos, n)
        
        if new_score > current_score:
            current_pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_pos = current_pos.copy()
                best_score = current_score
        else:
            # With small probability, accept worse moves (simulated annealing-like)
            if rng.random() < 0.05:
                current_pos = new_pos
                current_score = new_score
    
    return best_pos, best_score

def _gradient_optimize(pos, n, deadline):
    """
    Gradient-based optimization using penalty method.
    """
    radii = _compute_radii(pos, n)
    
    params = np.zeros(3 * n)
    params[:n*2] = pos.flatten()
    params[n*2:] = radii
    
    def objective(params):
        x = params[:n*2].reshape(n, 2)
        r = params[n*2:]
        
        # Clip to valid range
        r = np.clip(r, 1e-10, 0.5)
        
        # Compute actual valid radii
        valid_r = _compute_radii(x, n)
        
        # Penalty for deviation from desired radii
        penalty = 0.0
        for i in range(n):
            if r[i] > valid_r[i]:
                penalty += (r[i] - valid_r[i]) ** 2
        
        # Boundary violations
        for i in range(n):
            if x[i, 0] < r[i] or x[i, 0] > 1 - r[i] or x[i, 1] < r[i] or x[i, 1] > 1 - r[i]:
                penalty += 10.0
        
        return -np.sum(valid_r) + penalty
    
    best_params = params.copy()
    best_score = _score(pos, n)
    
    # Try a few penalty weights
    for pw in [1.0, 5.0, 20.0]:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            x = result.x[:n*2].reshape(n, 2)
            score = _score(x, n)
            
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    x = best_params[:n*2].reshape(n, 2)
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_score = -np.inf
    
    # Generate initial configurations
    configs = []
    configs.append(_initial_grid(N))
    configs.append(_initial_hexagonal(N))
    
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        configs.append(_initial_random(N, seed=seed))
    
    # Add Delaunay-like init
    try:
        pos_d, _ = _delaunay_init(N)
        configs.append(pos_d)
    except Exception:
        pass
    
    # Process each configuration
    for idx, pos_init in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        try:
            # First, try local search
            pos_ls, score_ls = _local_search(pos_init, N, deadline - 5, step_size=0.008, n_iter=300)
            
            if score_ls > best_score:
                best_score = score_ls
                best_pos = pos_ls.copy()
            
            # Then, try gradient optimization from the local search result
            if time.time() < deadline - 5:
                pos_grad, score_grad = _gradient_optimize(pos_ls, N, deadline - 2)
                
                if score_grad > best_score:
                    best_score = score_grad
                    best_pos = pos_grad.copy()
        except Exception:
            continue
    
    # Final polish: run local search on the best configuration with smaller steps
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_final, score_final = _local_search(best_pos, N, deadline - 1, step_size=0.003, n_iter=200)
            if score_final > best_score:
                best_score = score_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    # Build result
    if best_pos is None:
        # Fallback
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
    
    radii = _compute_radii(best_pos, N)
    result = np.zeros((N, 3))
    result[:, :2] = best_pos
    result[:, 2] = radii
    
    return result

# EVOLVE-BLOCK-END
