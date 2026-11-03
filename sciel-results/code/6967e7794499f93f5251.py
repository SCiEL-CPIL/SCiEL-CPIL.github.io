"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time

N = 50
DEADLINE = 110.0  # safety margin before 120s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _soft_radius(pos, temperature=0.01):
    """
    Smooth surrogate: soft-min of pairwise distances/2 and wall distances.
    Returns (value, gradient) for gradient-based optimization.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    
    # Wall distances
    wall = 0.5 - np.abs(pos)  # shape (n, 2)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min over all constraints
    # Wall soft-min: for each circle, soft-min over its two walls
    # Pairwise soft-min: soft-min over all pairs
    
    # Combine all constraints into one vector
    # Wall constraints: n*2 values (0.5 - |x_i|, 0.5 - |y_i|)
    wall_vals = np.concatenate([0.5 - np.abs(pos[:, 0]), 0.5 - np.abs(pos[:, 1])])
    # Pair constraints: dist/2 for each pair
    pair_idx = np.triu_indices(n, k=1)
    pair_vals = dist[pair_idx] / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft-min using log-sum-exp trick
    t = temperature
    m = all_vals.min()
    exp_vals = np.exp((all_vals - m) / t)
    soft_min = m + t * np.log(exp_vals.sum())
    
    # Gradient
    grad = np.zeros_like(pos)
    # Wall gradient
    # d(0.5 - |x_i|)/dx_i = -sign(x_i), d(0.5 - |y_i|)/dy_i = -sign(y_i)
    weights = exp_vals / exp_vals.sum()
    
    # Wall part
    for i in range(n):
        w0 = weights[2*i]
        w1 = weights[2*i+1]
        if pos[i,0] != 0:
            grad[i,0] += -w0 * np.sign(pos[i,0])
        else:
            grad[i,0] += -w0  # subgradient, use 0 or -1; use 0 for symmetry
        if pos[i,1] != 0:
            grad[i,1] += -w1 * np.sign(pos[i,1])
        else:
            grad[i,1] += -w1
    
    # Pair part
    for k, (i, j) in enumerate(zip(pair_idx[0], pair_idx[1])):
        w = weights[2*n + k]
        d = diff[i, j]
        dist_ij = dist[i, j]
        if dist_ij > 1e-15:
            grad[i] += w * d / dist_ij  # d(dist_ij/2)/dx_i = d_i / (2*dist_ij) * 2 = d_i/dist_ij... 
            # Actually d(dist_ij)/dx_i = (x_i - x_j)/dist_ij = d_i/dist_ij
            # d(dist_ij/2)/dx_i = d_i/(2*dist_ij)
            # So we need to multiply by 1/2
            grad[i] *= 0.5
            grad[j] -= grad[i]  # by antisymmetry... wait let me redo
        # Let me redo pair gradient properly
        # f = dist_ij / 2
        # df/dx_i = (x_i - x_j) / (2 * dist_ij)
        # df/dx_j = (x_j - x_i) / (2 * dist_ij)
        if dist_ij > 1e-15:
            factor = 1.0 / (2.0 * dist_ij)
            grad[i] += w * factor * d
            grad[j] -= w * factor * d
    
    return soft_min, grad

def _optimize_from_start(pos0, time_budget):
    """Run gradient-based optimization from a starting position."""
    pos = pos0.copy()
    best_pos = pos.copy()
    best_score = _score(pos)
    
    t_start = time.time()
    temperature = 0.05
    lr = 0.005
    patience = 0
    max_patience = 200
    
    while time.time() - t_start < time_budget:
        # Anneal temperature
        temperature = max(0.001, temperature * 0.999)
        
        val, grad = _soft_radius(pos, temperature)
        
        # Gradient ascent (maximize soft radius)
        pos_new = pos + lr * grad
        
        # Clamp to square to keep valid
        pos_new = np.clip(pos_new, -0.5, 0.5)
        
        # Check actual score
        actual = _score(pos_new)
        if actual > best_score:
            best_score = actual
            best_pos = pos_new.copy()
            patience = 0
        else:
            patience += 1
        
        pos = pos_new
        
        if patience > max_patience:
            break
        
        # Reduce learning rate over time
        lr *= 0.999
    
    return best_pos, best_score

def _make_grid_start(m, n):
    """Create a grid starting position."""
    positions = []
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n) + 0.5) / n - 0.5
    for y in ys:
        for x in xs:
            positions.append([x, y])
    positions = np.array(positions[:N])
    return positions

def _make_random_start():
    """Create a random starting position with jitter."""
    m = int(np.ceil(np.sqrt(N)))
    pos = _make_grid_start(m, m)
    # Add small random jitter
    pos = pos + 0.02 * np.random.randn(N, 2)
    pos = np.clip(pos, -0.49, 0.49)
    return pos

def _make_hex_start():
    """Create a hexagonal lattice starting position."""
    positions = []
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use roughly 7x7 or 8x7 arrangement
    # Let's try a hex grid
    spacing = 0.14  # approximate
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    # Scale to fit in square
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def construct():
    t_start = time.time()
    
    # Multiple starting configurations
    starts = []
    
    # 1. Grid start
    m = int(np.ceil(np.sqrt(N)))
    starts.append(_make_grid_start(m, m))
    
    # 2. Random jittered grid
    np.random.seed(42)
    starts.append(_make_random_start())
    
    # 3. Hexagonal start
    starts.append(_make_hex_start())
    
    # 4. Another random
    np.random.seed(123)
    starts.append(_make_random_start())
    
    best_pos = None
    best_score = -1.0
    
    # Allocate time budget across starts
    n_starts = len(starts)
    time_per_start = (DEADLINE - (time.time() - t_start)) / n_starts * 0.9  # 90% of remaining
    
    for i, start in enumerate(starts):
        remaining = DEADLINE - (time.time() - t_start)
        if remaining < 5:
            break
        
        # Use remaining time divided by remaining starts, with safety margin
        budget = min(time_per_start, remaining * 0.85 / (n_starts - i))
        budget = max(budget, 2.0)  # minimum 2 seconds per start
        
        pos, score = _optimize_from_start(start, budget)
        
        if score > best_score:
            best_score = score
            best_pos = pos
        
        # Check time
        if time.time() - t_start > DEADLINE - 5:
            break
    
    # Final polish: run a few more iterations on the best
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 3 and best_pos is not None:
        pos, score = _optimize_from_start(best_pos, remaining * 0.8)
        if score > best_score:
            best_score = score
            best_pos = pos
    
    if best_pos is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
