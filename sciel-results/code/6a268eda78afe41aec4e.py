<<<<<<< SEARCH
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

def _constraints(pos, radii):
    """
    Returns list of constraint functions for scipy minimize.
    Constraints:
      1) x_i >= r_i
      2) y_i >= r_i
      3) x_i + r_i <= 1
      4) y_i + r_i <= 1
      5) dist(i,j) >= r_i + r_j for all i < j
    """
    n = len(radii)
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda i=i: pos[2*i] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: pos[2*i+1] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: 1.0 - pos[2*i] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: 1.0 - pos[2*i+1] - radii[i]})
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda i=i, j=j: 
                         np.sqrt((pos[2*i]-pos[2*j])**2 + (pos[2*i+1]-pos[2*j+1])**2) - radii[i] - radii[j]})
    return cons

def _make_objective(radii):
    """Objective: maximize sum of radii, i.e., minimize -sum(radii)."""
    def obj(pos):
        return -np.sum(radii)
    return obj

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
    for iteration in range(50):
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

def _compute_score(pos, radii):
    """Compute the sum of radii, which is the objective value."""
    return np.sum(radii)

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
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
    
    # Try to improve using local optimization
    # We'll use a simple gradient-free approach: try increasing radii and re-projecting
    
    # Strategy: iteratively increase radii slightly and project
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Try to increase radii
        new_radii = radii * (1.0 + 0.01)
        new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
        
        if new_valid:
            new_score = _compute_score(new_pos, new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos, radii = new_pos, new_radii
        else:
            # Try smaller increase
            new_radii = radii * (1.0 + 0.001)
            new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
            if new_valid:
                new_score = _compute_score(new_pos, new_radii)
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
                    best_radii = new_radii.copy()
                    pos, radii = new_pos, new_radii
    
    # Try different initializations
    for restart in range(5):
        if time.time() > deadline - 10.0:
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
        
        # Try to improve from this starting point
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            
            new_radii = radii * (1.0 + 0.02)
            new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
            
            if new_valid:
                new_score = _compute_score(new_pos, new_radii)
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
                    best_radii = new_radii.copy()
                    pos, radii = new_pos, new_radii
                elif new_score > score:
                    pos, radii = new_pos, new_radii
                    score = new_score
    
    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[2*i]
        result[i, 1] = best_pos[2*i+1]
        result[i, 2] = best_radii[i]
    
    return result

# EVOLVE-BLOCK-END
=======
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

def _smooth_penalty(pos, radii, beta=50.0):
    """
    Compute smooth penalty for constraint violations.
    Uses soft-min via log-sum-exp style smoothing.
    """
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        # x_i >= r_i
        viol = radii[i] - pos[2*i]
        if viol > 0:
            penalty += viol * viol
        # y_i >= r_i
        viol = radii[i] - pos[2*i+1]
        if viol > 0:
            penalty += viol * viol
        # x_i + r_i <= 1
        viol = pos[2*i] + radii[i] - 1.0
        if viol > 0:
            penalty += viol * viol
        # y_i + r_i <= 1
        viol = pos[2*i+1] + radii[i] - 1.0
        if viol > 0:
            penalty += viol * viol
    
    # Pairwise overlap violations (smoothed)
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            viol = min_dist - dist
            if viol > 0:
                # Smooth penalty: use quadratic for small violations
                penalty += viol * viol
    
    return penalty

def _objective_with_penalty(x, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * sum(violations^2)
    x = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = len(x) // 3
    pos = np.zeros(2*n)
    radii = np.zeros(n)
    for i in range(n):
        pos[2*i] = x[3*i]
        pos[2*i+1] = x[3*i+1]
        radii[i] = max(x[3*i+2], 1e-10)
    
    penalty = _smooth_penalty(pos, radii, beta)
    return -np.sum(radii) + beta * penalty

def _is_valid(pos, radii, tol=1e-9):
    """Check if packing is valid."""
    n = len(radii)
    for i in range(n):
        if pos[2*i] < radii[i] - tol or pos[2*i] > 1.0 - radii[i] + tol:
            return False
        if pos[2*i+1] < radii[i] - tol or pos[2*i+1] > 1.0 - radii[i] + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[2*i] - pos[2*j]
            dy = pos[2*i+1] - pos[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _project_to_valid(pos, radii, tol=1e-10, max_iter=100):
    """
    Given positions and radii, adjust to ensure validity.
    Returns (pos, radii, valid) where valid is a boolean.
    """
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions and radii to be within [0, 1]
    for i in range(n):
        if radii[i] < 1e-10:
            radii[i] = 1e-10
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    
    # Iteratively resolve overlaps
    for iteration in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        dx = np.random.rand() - 0.5
                        dy = np.random.rand() - 0.5
                        dist = np.sqrt(dx*dx + dy*dy)
                    overlap = min_dist - dist
                    push_x = (dx / dist) * overlap / 2
                    push_y = (dy / dist) * overlap / 2
                    pos[2*i] += push_x
                    pos[2*i+1] += push_y
                    pos[2*j] -= push_x
                    pos[2*j+1] -= push_y
                    pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
                    pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
                    pos[2*j] = np.clip(pos[2*j], radii[j], 1.0 - radii[j])
                    pos[2*j+1] = np.clip(pos[2*j+1], radii[j], 1.0 - radii[j])
                    moved = True
        if not moved:
            break
    
    valid = _is_valid(pos, radii, tol)
    return pos, radii, valid

def _hex_initialization(jitter=0.0):
    """Create hexagonal packing as initial guess with optional jitter."""
    # Hexagonal packing: rows offset by half spacing
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add hexagonal offset for odd rows
            if i % 2 == 1:
                x += 0.5 / side
            r = 0.48 / side
            if jitter > 0:
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
            cells.append((x, y, r))
    cells = cells[:N]
    pos = np.array([item[0] for item in cells] + [item[1] for item in cells])
    radii = np.array([item[2] for item in cells])
    return pos, radii

def _random_initialization():
    """Create random initial positions and radii."""
    radii = np.full(N, 0.02)
    pos = np.random.rand(2*N)
    for i in range(N):
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    return pos, radii

def _optimize_from_start(pos, radii, deadline):
    """
    Optimize from a starting configuration using scipy minimize.
    Returns improved (pos, radii, score).
    """
    n = len(radii)
    # Pack into single vector: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = pos[2*i]
        x0[3*i+1] = pos[2*i+1]
        x0[3*i+2] = radii[i]
    
    # Bounds: x, y in [0, 1], r in [1e-10, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((1e-10, 1.0))  # x
        bounds.append((1e-10, 1.0))  # y
        bounds.append((1e-10, 0.5))  # r
    
    # Try SLSQP
    best_pos, best_radii, best_score = pos.copy(), radii.copy(), np.sum(radii)
    
    try:
        result = minimize(
            _objective_with_penalty,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if result.success or result.fun < _objective_with_penalty(x0):
            x_opt = result.x
            new_pos = np.zeros(2*n)
            new_radii = np.zeros(n)
            for i in range(n):
                new_pos[2*i] = x_opt[3*i]
                new_pos[2*i+1] = x_opt[3*i+1]
                new_radii[i] = max(x_opt[3*i+2], 1e-10)
            
            # Project to valid
            new_pos, new_radii, valid = _project_to_valid(new_pos, new_radii)
            if valid:
                new_score = np.sum(new_radii)
                if new_score > best_score:
                    best_pos, best_radii, best_score = new_pos, new_radii, new_score
    except Exception:
        pass
    
    # Try COBYLA as alternative
    try:
        result2 = minimize(
            _objective_with_penalty,
            x0,
            method='COBYLA',
            bounds=bounds,
            options={'maxiter': 200, 'rhobeg': 1e-4}
        )
        
        if result2.fun < _objective_with_penalty(x0):
            x_opt = result2.x
            new_pos = np.zeros(2*n)
            new_radii = np.zeros(n)
            for i in range(n):
                new_pos[2*i] = x_opt[3*i]
                new_pos[2*i+1] = x_opt[3*i+1]
                new_radii[i] = max(x_opt[3*i+2], 1e-10)
            
            new_pos, new_radii, valid = _project_to_valid(new_pos, new_radii)
            if valid:
                new_score = np.sum(new_radii)
                if new_score > best_score:
                    best_pos, best_radii, best_score = new_pos, new_radii, new_score
    except Exception:
        pass
    
    # Local perturbation search
    for pert in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation
        pert_pos = best_pos + np.random.randn(2*n) * 0.01
        pert_radii = best_radii * (1 + np.random.randn(n) * 0.02)
        pert_radii = np.clip(pert_radii, 1e-10, 0.5)
        
        pert_pos, pert_radii, valid = _project_to_valid(pert_pos, pert_radii)
        if valid:
            pert_score = np.sum(pert_radii)
            if pert_score > best_score:
                best_pos, best_radii, best_score = pert_pos, pert_radii, pert_score
    
    return best_pos, best_radii, best_score

def _polish(pos, radii, deadline):
    """Final high-precision polish pass."""
    n = len(radii)
    best_pos, best_radii, best_score = pos.copy(), radii.copy(), np.sum(radii)
    
    # Iterative radius increase with projection
    for attempt in range(50):
        if time.time() > deadline - 2.0:
            break
        
        # Try small radius increase
        scale = 1.0 + 0.005 * (1 - attempt / 50.0)  # Decreasing step size
        new_radii = best_radii * scale
        new_pos, new_radii, valid = _project_to_valid(best_pos, new_radii, tol=1e-11, max_iter=200)
        
        if valid:
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_pos, best_radii, best_score = new_pos, new_radii, new_score
            else:
                # Try smaller step
                new_radii2 = best_radii * (1.0 + 0.001)
                new_pos2, new_radii2, valid2 = _project_to_valid(best_pos, new_radii2, tol=1e-11, max_iter=200)
                if valid2:
                    new_score2 = np.sum(new_radii2)
                    if new_score2 > best_score:
                        best_pos, best_radii, best_score = new_pos2, new_radii2, new_score2
        else:
            # Try much smaller step
            new_radii2 = best_radii * (1.0 + 0.0005)
            new_pos2, new_radii2, valid2 = _project_to_valid(best_pos, new_radii2, tol=1e-11, max_iter=200)
            if valid2:
                new_score2 = np.sum(new_radii2)
                if new_score2 > best_score:
                    best_pos, best_radii, best_score = new_pos2, new_radii2, new_score2
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Multiple restarts from different initializations
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Restart 1: Hexagonal grid
    pos, radii = _hex_initialization(jitter=0.0)
    pos, radii, valid = _project_to_valid(pos, radii)
    if valid:
        score = np.sum(radii)
        pos, radii, score = _optimize_from_start(pos, radii, deadline)
        if score > best_score:
            best_pos, best_radii, best_score = pos, radii, score
    
    # Restart 2: Jittered hexagonal
    if time.time() < deadline - 20.0:
        pos, radii = _hex_initialization(jitter=0.02)
        pos, radii, valid = _project_to_valid(pos, radii)
        if valid:
            score = np.sum(radii)
            pos, radii, score = _optimize_from_start(pos, radii, deadline)
            if score > best_score:
                best_pos, best_radii, best_score = pos, radii, score
    
    # Restart 3: Random
    if time.time() < deadline - 20.0:
        pos, radii = _random_initialization()
        pos, radii, valid = _project_to_valid(pos, radii)
        if valid:
            score = np.sum(radii)
            pos, radii, score = _optimize_from_start(pos, radii, deadline)
            if score > best_score:
                best_pos, best_radii, best_score = pos, radii, score
    
    # Restart 4: Another random
    if time.time() < deadline - 15.0:
        pos, radii = _random_initialization()
        pos, radii, valid = _project_to_valid(pos, radii)
        if valid:
            score = np.sum(radii)
            pos, radii, score = _optimize_from_start(pos, radii, deadline)
            if score > best_score:
                best_pos, best_radii, best_score = pos, radii, score
    
    # Final polish
    if best_pos is not None and time.time() < deadline - 10.0:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    # If we have no valid result, fall back to trivial grid
    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.48 / side
                cells.append((x, y, r))
        cells = cells[:N]
        return np.array(cells)
    
    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[2*i]
        result[i, 1] = best_pos[2*i+1]
        result[i, 2] = best_radii[i]
    
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
