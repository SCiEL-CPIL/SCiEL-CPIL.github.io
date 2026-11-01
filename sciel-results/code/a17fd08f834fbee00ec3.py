"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _pairwise_distances(pos):
    """Return (n, n) matrix of Euclidean distances."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _compute_score(pos, radii):
    """Compute the sum of radii, which is the objective value."""
    return np.sum(radii)

def _project_to_valid(pos, radii, tol=1e-10):
    """
    Given positions and radii, adjust to ensure validity.
    Returns (pos, radii, valid) where valid is a boolean.
    """
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions and radii to be within [0, 1]
    for i in range(n):
        # Ensure radius is positive
        if radii[i] < 1e-10:
            radii[i] = 1e-10
        # Clamp center so circle is inside [0,1]^2
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    
    # Iteratively resolve overlaps
    for iteration in range(100):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - tol:
                    # Overlap detected; push apart
                    if dist < 1e-12:
                        # Random perturbation
                        dx = np.random.rand() - 0.5
                        dy = np.random.rand() - 0.5
                        dist = np.sqrt(dx*dx + dy*dy)
                    overlap = min_dist - dist
                    # Push each circle half the overlap apart
                    push_x = (dx / dist) * overlap / 2
                    push_y = (dy / dist) * overlap / 2
                    pos[2*i] += push_x
                    pos[2*i+1] += push_y
                    pos[2*j] -= push_x
                    pos[2*j+1] -= push_y
                    # Re-clamp to boundaries
                    pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
                    pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
                    pos[2*j] = np.clip(pos[2*j], radii[j], 1.0 - radii[j])
                    pos[2*j+1] = np.clip(pos[2*j+1], radii[j], 1.0 - radii[j])
                    moved = True
        if not moved:
            break
    
    # Final check for validity
    valid = True
    for i in range(n):
        if pos[2*i] < radii[i] - tol or pos[2*i] > 1.0 - radii[i] + tol:
            valid = False
            break
        if pos[2*i+1] < radii[i] - tol or pos[2*i+1] > 1.0 - radii[i] + tol:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    valid = False
                    break
            if not valid:
                break
    
    return pos, radii, valid

def _grid_initialization():
    """Create a grid of equal circles as initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append((x, y, r))
    cells = cells[:N]
    pos = np.array([item[0] for item in cells] + [item[1] for item in cells])
    radii = np.array([item[2] for item in cells])
    return pos, radii

def _random_initialization():
    """Create random initial positions and radii."""
    # Start with small radii and random positions
    radii = np.full(N, 0.02)
    pos = np.random.rand(2*N)
    # Clamp to be within bounds
    for i in range(N):
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    return pos, radii

def _soft_repulsive_potential(pos, radii, beta=10.0):
    """
    Compute soft repulsive potential energy for overlapping circles.
    Uses a smooth approximation: E = sum_{i<j} max(0, r_i + r_j - d_ij)^2 * beta
    This is differentiable and encourages separation.
    """
    n = len(radii)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                energy += beta * overlap * overlap
    # Boundary penalties
    for i in range(n):
        for axis in range(2):
            val = pos[2*i + axis]
            r = radii[i]
            if val < r:
                energy += beta * (r - val) ** 2
            if val > 1.0 - r:
                energy += beta * (val - (1.0 - r)) ** 2
    return energy

def _soft_repulsive_gradient(pos, radii, beta=10.0):
    """
    Compute gradient of the soft repulsive potential.
    """
    n = len(radii)
    grad = np.zeros(2*n)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist and dist > 1e-12:
                overlap = min_dist - dist
                # dE/ddist = -2*beta*overlap
                # ddist/dpos_i = (pos_i - pos_j)/dist
                factor = 2 * beta * overlap / dist
                grad[2*i] += factor * dx
                grad[2*i+1] += factor * dy
                grad[2*j] -= factor * dx
                grad[2*j+1] -= factor * dy
    
    # Boundary gradients
    for i in range(n):
        for axis in range(2):
            val = pos[2*i + axis]
            r = radii[i]
            if val < r:
                grad[2*i + axis] += 2 * beta * (r - val)
            if val > 1.0 - r:
                grad[2*i + axis] -= 2 * beta * (val - (1.0 - r))
    
    return grad

def _physics_optimize(pos, radii, max_iters=200, lr=0.01, deadline=None):
    """
    Optimize positions using gradient descent on the soft repulsive potential.
    Radii are fixed; only positions are optimized.
    """
    n = len(radii)
    pos = pos.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _soft_repulsive_gradient(pos, radii)
        pos -= lr * grad
        
        # Clamp to valid region
        for i in range(n):
            pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
            pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-8:
            break
    
    return pos

def _increase_radii_and_optimize(pos, radii, scale=1.01, max_iters=100, deadline=None):
    """
    Try to increase all radii by a factor, then optimize positions.
    Returns (new_pos, new_radii, valid, score)
    """
    n = len(radii)
    new_radii = radii * scale
    new_pos = pos.copy()
    
    # Optimize positions with new radii
    new_pos = _physics_optimize(new_pos, new_radii, max_iters=max_iters, deadline=deadline)
    
    # Project to valid
    new_pos, new_radii, valid = _project_to_valid(new_pos, new_radii)
    
    if valid:
        score = _compute_score(new_pos, new_radii)
        return new_pos, new_radii, valid, score
    else:
        return None, None, False, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin
    
    # Start with grid initialization
    pos, radii = _grid_initialization()
    pos, radii, valid = _project_to_valid(pos, radii)
    best_score = _compute_score(pos, radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    if not valid:
        # Fall back to trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Physics-based optimization: iteratively increase radii and optimize positions
    # Use adaptive scaling
    current_scale = 0.01
    consecutive_failures = 0
    
    for attempt in range(100):
        if time.time() > deadline - 10.0:
            break
        
        new_pos, new_radii, new_valid, new_score = _increase_radii_and_optimize(
            pos, radii, scale=1.0 + current_scale, max_iters=50, deadline=deadline
        )
        
        if new_valid and new_score > best_score:
            best_score = new_score
            best_pos = new_pos.copy()
            best_radii = new_radii.copy()
            pos, radii = new_pos, new_radii
            consecutive_failures = 0
            # Try slightly larger increases
            current_scale = min(current_scale * 1.2, 0.05)
        else:
            consecutive_failures += 1
            # Decrease scale on failure
            current_scale = max(current_scale * 0.8, 0.001)
            if consecutive_failures > 10:
                break
    
    # Try different initializations with physics optimization
    for restart in range(8):
        if time.time() > deadline - 15.0:
            break
        
        pos, radii = _random_initialization()
        pos, radii, valid = _project_to_valid(pos, radii)
        
        if not valid:
            continue
        
        score = _compute_score(pos, radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
        
        # Physics-based improvement from this starting point
        current_scale = 0.02
        consecutive_failures = 0
        
        for attempt in range(60):
            if time.time() > deadline - 10.0:
                break
            
            new_pos, new_radii, new_valid, new_score = _increase_radii_and_optimize(
                pos, radii, scale=1.0 + current_scale, max_iters=40, deadline=deadline
            )
            
            if new_valid and new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos, radii = new_pos, new_radii
                consecutive_failures = 0
                current_scale = min(current_scale * 1.15, 0.04)
            else:
                consecutive_failures += 1
                current_scale = max(current_scale * 0.85, 0.001)
                if consecutive_failures > 8:
                    break
    
    # Final high-precision polish: use smaller steps and more iterations
    if time.time() < deadline - 20.0:
        current_scale = 0.001
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            new_pos, new_radii, new_valid, new_score = _increase_radii_and_optimize(
                best_pos, best_radii, scale=1.0 + current_scale, max_iters=80, deadline=deadline
            )
            
            if new_valid and new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            else:
                break
    
    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[2*i]
        result[i, 1] = best_pos[2*i+1]
        result[i, 2] = best_radii[i]
    
    return result

# EVOLVE-BLOCK-END
