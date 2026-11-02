"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _validity(arr):
    """Check if circles are valid: inside [0,1]^2 and disjoint interiors."""
    n = len(arr)
    if n != 26:
        return False
    for i in range(n):
        x, y, r = arr[i]
        if r <= 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _score(arr):
    """Sum of radii, 0 if invalid."""
    if not _validity(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))

def _initialize_grid():
    """Initialize with a grid of equal circles."""
    side = 5  # 5x5 grid = 25, plus one more
    # Use 5x5 grid
    positions = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            r = 0.5 / 5.0 - 1e-4
            positions.append([x, y, r])
    # Add 26th circle in a corner
    positions.append([0.05, 0.05, 0.03])
    return np.array(positions)

def _initialize_random():
    """Random initialization with small circles."""
    positions = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.02, 0.08)
        positions.append([x, y, r])
    return np.array(positions)

def _gradient_ascent_step(pos, lr, max_iter=100):
    """Perform gradient ascent on sum of radii with constraints."""
    pos = pos.copy()
    n = len(pos)
    
    for it in range(max_iter):
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Boundary forces: push away from boundaries
        for i in range(n):
            x, y, r = pos[i]
            # Left boundary
            if x - r < 0:
                forces[i, 0] += (0 - (x - r)) * 10
            # Right boundary
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += (0 - (y - r)) * 10
            # Top boundary
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10
        
        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = pos[i]
                xj, yj, rj = pos[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = ri + rj
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
                elif dist < min_dist:
                    # Random push if coincident
                    forces[i, 0] += random.uniform(-0.01, 0.01)
                    forces[i, 1] += random.uniform(-0.01, 0.01)
                    forces[j, 0] -= random.uniform(-0.01, 0.01)
                    forces[j, 1] -= random.uniform(-0.01, 0.01)
        
        # Update positions
        for i in range(n):
            pos[i, 0] += lr * forces[i, 0]
            pos[i, 1] += lr * forces[i, 1]
            # Clamp to [0,1]
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
    
    return pos

def _expand_radii(pos, n_iter=50):
    """Try to expand radii while maintaining constraints."""
    pos = pos.copy()
    n = len(pos)
    
    for it in range(n_iter):
        improved = False
        for i in range(n):
            # Try to increase radius
            for _ in range(10):
                new_r = pos[i, 2] + 0.001
                pos[i, 2] = new_r
                if _validity(pos):
                    improved = True
                else:
                    pos[i, 2] = pos[i, 2] - 0.001
                    break
        if not improved:
            break
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = 0.0
    best_pos = None
    
    # Try multiple initializations
    initializations = [_initialize_grid(), _initialize_random()]
    
    # Add some varied random initializations
    for _ in range(3):
        initializations.append(_initialize_random())
    
    for init_pos in initializations:
        if time.time() > deadline:
            break
        
        # Gradient ascent to fix overlaps
        pos = _gradient_ascent_step(init_pos, lr=0.01, max_iter=50)
        
        # Expand radii
        pos = _expand_radii(pos, n_iter=30)
        
        score = _score(pos)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    
    # If we have time, try more iterations
    if best_pos is not None and time.time() < deadline - 10:
        # Local search with small perturbations
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbed = best_pos.copy()
            for i in range(N):
                perturbed[i, 0] += random.uniform(-0.01, 0.01)
                perturbed[i, 1] += random.uniform(-0.01, 0.01)
            perturbed = _gradient_ascent_step(perturbed, lr=0.01, max_iter=30)
            perturbed = _expand_radii(perturbed, n_iter=20)
            score = _score(perturbed)
            if score > best_score:
                best_score = score
                best_pos = perturbed.copy()
    
    if best_pos is None:
        # Fallback to grid
        best_pos = _initialize_grid()
    
    return best_pos
# EVOLVE-BLOCK-END
