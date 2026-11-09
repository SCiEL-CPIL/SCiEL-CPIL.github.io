"""Rewritten program: physics-based soft-body simulation for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _repulsive_energy(pos, r, sigma=0.01):
    """Compute repulsive energy between all circle pairs using a soft potential."""
    n = len(r)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist + sigma:
                # Soft repulsive force when overlapping or close
                overlap = min_dist - dist
                if overlap > 0:
                    energy += 0.5 * overlap * overlap
    return energy


def _wall_energy(pos, r, W, H):
    """Compute energy from walls (circles should stay inside rectangle)."""
    energy = 0.0
    for i in range(len(r)):
        # Left wall
        if pos[i, 0] - r[i] < 0:
            energy += 0.5 * (r[i] - pos[i, 0]) ** 2
        # Right wall
        if pos[i, 0] + r[i] > W:
            energy += 0.5 * (pos[i, 0] + r[i] - W) ** 2
        # Bottom wall
        if pos[i, 1] - r[i] < 0:
            energy += 0.5 * (r[i] - pos[i, 1]) ** 2
        # Top wall
        if pos[i, 1] + r[i] > H:
            energy += 0.5 * (pos[i, 1] + r[i] - H) ** 2
    return energy


def _total_energy(pos, r, W, H, sigma=0.01):
    return _repulsive_energy(pos, r, sigma) + _wall_energy(pos, r, W, H)


def _is_valid(pos, r, W, H, tol=1e-7):
    """Check if configuration is valid (no overlaps, inside rectangle)."""
    n = len(r)
    # Check walls
    for i in range(n):
        if pos[i, 0] - r[i] < -tol or pos[i, 0] + r[i] > W + tol:
            return False
        if pos[i, 1] - r[i] < -tol or pos[i, 1] + r[i] > H + tol:
            return False
    # Check pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _simulate(pos, r, W, H, n_iter=5000, lr=0.01, cooling=0.999, sigma=0.01):
    """Run a simple gradient-based simulation to settle circles."""
    pos = pos.copy()
    r = r.copy()
    
    for it in range(n_iter):
        # Compute forces on positions
        force_pos = np.zeros_like(pos)
        for i in range(len(r)):
            # Repulsive forces from other circles
            for j in range(len(r)):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist + sigma and dist > 1e-10:
                    overlap = min_dist - dist
                    if overlap > 0:
                        force_pos[i, 0] += overlap * dx / dist
                        force_pos[i, 1] += overlap * dy / dist
            
            # Wall forces
            if pos[i, 0] - r[i] < 0:
                force_pos[i, 0] += (r[i] - pos[i, 0])
            if pos[i, 0] + r[i] > W:
                force_pos[i, 0] -= (pos[i, 0] + r[i] - W)
            if pos[i, 1] - r[i] < 0:
                force_pos[i, 1] += (r[i] - pos[i, 1])
            if pos[i, 1] + r[i] > H:
                force_pos[i, 1] -= (pos[i, 1] + r[i] - H)
        
        # Update positions
        pos += lr * force_pos
        
        # Keep inside rectangle (hard constraint)
        for i in range(len(r)):
            pos[i, 0] = np.clip(pos[i, 0], r[i], W - r[i])
            pos[i, 1] = np.clip(pos[i, 1], r[i], H - r[i])
        
        # Cool down learning rate
        lr *= cooling
    
    return pos, r


def _grow_radii(pos, r, W, H, n_steps=100, growth_rate=0.001):
    """Try to grow radii while maintaining validity."""
    r = r.copy()
    for step in range(n_steps):
        # Try growing all radii slightly
        r_new = r + growth_rate
        # Check if still valid
        if _is_valid(pos, r_new, W, H):
            r = r_new
        else:
            # Try growing one at a time
            for i in range(len(r)):
                r_trial = r.copy()
                r_trial[i] += growth_rate
                if _is_valid(pos, r_trial, W, H):
                    r = r_trial
                    break
            else:
                break
    return r


def _pack_single(W, H, seed=0):
    """Pack N circles in a rectangle of width W and height H."""
    rng = random.Random(seed)
    
    # Initial positions: random or grid
    # Use a jittered grid initialization
    cols = 5
    rows = 5
    cell_w = W / cols
    cell_h = H / rows
    
    pos = np.zeros((N, 2))
    for i in range(N):
        col = i % cols
        row = i // cols
        pos[i, 0] = (col + 0.5) * cell_w + rng.uniform(-0.1, 0.1) * cell_w
        pos[i, 1] = (row + 0.5) * cell_h + rng.uniform(-0.1, 0.1) * cell_h
    
    # Initial radii: small
    r = np.full(N, 0.05)
    
    # Run simulation
    pos, r = _simulate(pos, r, W, H, n_iter=3000, lr=0.02, cooling=0.9995, sigma=0.01)
    
    # Grow radii
    r = _grow_radii(pos, r, W, H, n_steps=200, growth_rate=0.002)
    
    # Final simulation with larger radii
    pos, r = _simulate(pos, r, W, H, n_iter=2000, lr=0.01, cooling=0.9995, sigma=0.01)
    
    # Grow radii again
    r = _grow_radii(pos, r, W, H, n_steps=100, growth_rate=0.001)
    
    return pos, r


def _evaluate(pos, r, W, H):
    """Return sum of radii if valid, else -1."""
    if not _is_valid(pos, r, W, H):
        return -1.0
    return np.sum(r)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, H = W * aspect
        # W + W*aspect = 2 => W = 2/(1+aspect)
        W = 2.0 / (1.0 + aspect)
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            continue
        
        # Try multiple seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            pos, r = _pack_single(W, H, seed=seed)
            score = _evaluate(pos, r, W, H)
            
            if score > best_score:
                best_score = score
                # Normalize to unit square for output (or keep as is)
                # The task says bounding rectangle has perimeter at most 4
                # Our rectangle has perimeter exactly 4, so we're good
                best_result = np.column_stack([pos, r])
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
