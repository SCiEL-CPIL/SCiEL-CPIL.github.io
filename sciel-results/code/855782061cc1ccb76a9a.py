"""Circle packing in unit square n=32 — Force-directed gradient optimization."""
import time
import numpy as np

N = 32
_DEADLINE = 115.0


def _compute_forces(pos, rad, n, repulsion_strength=10.0, attraction_strength=0.1):
    """Compute forces on positions and radii.
    Returns: grad_pos (n, 2), grad_rad (n,)
    """
    grad_pos = np.zeros((n, 2))
    grad_rad = np.zeros(n)

    # Pairwise repulsion
    # d[i,j] is distance
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    
    # Avoid division by zero
    d_safe = np.where(d < 1e-6, 1e-6, d)
    
    # Force between i and j
    # If d < r_i + r_j, they overlap. Push apart.
    # Gradient of potential energy U = sum (max(0, r_i + r_j - d)^2)
    # dU/dd = -2 * (r_i + r_j - d) if d < r_i + r_j
    # Force on i from j: -dU/dpos_i = 2 * (r_i + r_j - d) * (pos_i - pos_j) / d
    
    # We want to maximize sum(rad).
    # Let's use a penalty method: Minimize Penalty = sum(max(0, r_i + r_j - d)^2) + sum(max(0, r_i - min(pos_i, 1-pos_i))^2)
    # Gradient descent on Penalty, while increasing radii?
    # Simpler: Simulated Annealing with force-directed relaxation.
    
    # For each pair, if overlapping, apply repulsive force to positions.
    for i in range(n):
        for j in range(i + 1, n):
            rij = rad[i] + rad[j]
            dij = d[i, j]
            if dij < rij:
                # Overlap
                # Force magnitude proportional to overlap
                overlap = rij - dij
                # Direction
                dir_vec = pos[i] - pos[j]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-9:
                    dir_vec = dir_vec / norm
                else:
                    # Random direction
                    dir_vec = np.array([1.0, 0.0])
                
                force_mag = repulsion_strength * overlap
                grad_pos[i] += force_mag * dir_vec
                grad_pos[j] -= force_mag * dir_vec
                
                # Also, we can shrink radii to resolve conflict
                # Gradient of penalty w.r.t rad_i is 2 * overlap
                grad_rad[i] -= 2.0 * overlap
                grad_rad[j] -= 2.0 * overlap

    # Wall constraints
    for i in range(n):
        # Left wall
        if pos[i, 0] < rad[i]:
            overlap = rad[i] - pos[i, 0]
            grad_pos[i, 0] += repulsion_strength * overlap
            grad_rad[i] -= 2.0 * overlap
        # Right wall
        if 1.0 - pos[i, 0] < rad[i]:
            overlap = rad[i] - (1.0 - pos[i, 0])
            grad_pos[i, 0] -= repulsion_strength * overlap
            grad_rad[i] -= 2.0 * overlap
        # Bottom wall
        if pos[i, 1] < rad[i]:
            overlap = rad[i] - pos[i, 1]
            grad_pos[i, 1] += repulsion_strength * overlap
            grad_rad[i] -= 2.0 * overlap
        # Top wall
        if 1.0 - pos[i, 1] < rad[i]:
            overlap = rad[i] - (1.0 - pos[i, 1])
            grad_pos[i, 1] -= repulsion_strength * overlap
            grad_rad[i] -= 2.0 * overlap

    return grad_pos, grad_rad


def _optimize(pos, rad, n, budget, rng):
    """Force-directed optimization with annealing."""
    pos = pos.copy()
    rad = rad.copy()
    t0 = time.time()
    temp = 0.05
    best_score = np.sum(rad)
    best_pos = pos.copy()
    best_rad = rad.copy()
    
    for it in range(5000):
        if time.time() - t0 > budget:
            break
            
        grad_pos, grad_rad = _compute_forces(pos, rad, n)
        
        # Add noise
        noise_pos = rng.normal(0, temp, size=(n, 2))
        noise_rad = rng.normal(0, temp, size=n)
        
        # Update
        new_pos = pos - 0.01 * grad_pos + noise_pos
        new_rad = rad - 0.01 * grad_rad + noise_rad
        
        # Clip radii
        new_rad = np.clip(new_rad, 1e-5, 0.45)
        
        # Clip positions to be inside square (roughly)
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, 1.0)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, 1.0)
        
        # Check feasibility
        d = np.linalg.norm(new_pos[:, None, :] - new_pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        c_nei = np.min(d - new_rad[:, None] - new_rad[None, :], axis=1)
        c_wall = np.minimum(np.minimum(new_pos[:, 0], 1.0 - new_pos[:, 0]),
                            np.minimum(new_pos[:, 1], 1.0 - new_pos[:, 1]))
        
        feasible = np.all(c_nei >= -1e-9) and np.all(c_wall >= -1e-9)
        
        if feasible:
            score = np.sum(new_rad)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_rad = new_rad.copy()
                pos = new_pos
                rad = new_rad
                temp *= 0.99
        else:
            # Reject
            temp *= 0.995
            
    return best_pos, best_rad, best_score


def _initial_grid(n, rng):
    side = int(np.ceil(np.sqrt(n)))
    pos = []
    rad = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            pos.append([x, y])
            rad.append(r)
            idx += 1
    pos = np.array(pos[:n])
    rad = np.array(rad[:n])
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    pos0, rad0 = _initial_grid(N, rng)
    
    best_pos, best_rad = pos0, rad0
    best_score = np.sum(rad0)
    
    n_restarts = 5
    remaining = _DEADLINE - (time.time() - t_start) - 5.0
    per_budget = remaining / n_restarts
    
    for r in range(n_restarts):
        if time.time() - t_start > _DEADLINE - 5.0:
            break
            
        p = best_pos.copy()
        rd = best_rad.copy()
        
        # Perturb
        for k in range(N):
            rd[k] *= (1.0 + (rng.random() - 0.5) * 0.2)
            p[k, 0] += (rng.random() - 0.5) * 0.1
            p[k, 1] += (rng.random() - 0.5) * 0.1
            p[k, 0] = np.clip(p[k, 0], 0.0, 1.0)
            p[k, 1] = np.clip(p[k, 1], 0.0, 1.0)
            
        p, rd, score = _optimize(p, rd, N, per_budget, rng)
        
        if score > best_score:
            best_score = score
            best_pos = p
            best_rad = rd
            
    # Final check
    d = np.linalg.norm(best_pos[:, None, :] - best_pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    c_nei = np.min(d - best_rad[:, None] - best_rad[None, :], axis=1)
    c_wall = np.minimum(np.minimum(best_pos[:, 0], 1.0 - best_pos[:, 0]),
                        np.minimum(best_pos[:, 1], 1.0 - best_pos[:, 1]))
    
    if np.any(c_nei < -1e-9) or np.any(c_wall < -1e-9):
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack([best_pos, best_rad])

