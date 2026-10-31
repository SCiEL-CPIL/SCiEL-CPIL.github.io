"""Circle packing in perimeter-4 rectangle; maximize sum of radii for 21 circles."""
import time
import numpy as np

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _validate(arr):
    """Check disjoint interiors, non-negative radii, and perimeter <= 4."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] < 0):
        return False
    # Bounding box
    min_x = np.min(arr[:, 0] - arr[:, 2])
    max_x = np.max(arr[:, 0] + arr[:, 2])
    min_y = np.min(arr[:, 1] - arr[:, 2])
    max_y = np.max(arr[:, 1] + arr[:, 2])
    w = max_x - min_x
    h = max_y - min_y
    if 2.0 * (w + h) > 4.0 + 1e-9:
        return False
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < arr[i, 2] + arr[j, 2] - 1e-9:
                return False
    return True

def _objective(arr):
    """Return sum of radii if valid, else -inf."""
    if not _validate(arr):
        return -np.inf
    return np.sum(arr[:, 2])

def _grid_init():
    """Create a grid-like initial placement in a 1x1 square (perimeter=4)."""
    # Use a 5x5 grid, pick 21 cells
    side = 5
    # Center of each cell
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append((i, j))
    # Pick 21 out of 25 - use all except corners and center-ish for uniformity
    # Actually just take first 21 in row-major
    idx = list(range(side * side))[:N]
    cells = []
    r = 0.5 / side
    for k in idx:
        i = k // side
        j = k % side
        cells.append([j + 0.5, i + 0.5, r - 1e-4])
    return np.array(cells)

def _random_init():
    """Random placement in a 1x1 square."""
    # Place circles randomly with small radii
    radii = 0.03 + 0.02 * np.random.rand(N)
    xs = radii + (1.0 - 2*radii) * np.random.rand(N)
    ys = radii + (1.0 - 2*radii) * np.random.rand(N)
    return np.column_stack([xs, ys, radii])

def _gradient_step(arr, step_size=0.01, n_iter=50):
    """
    Gradient-based optimization: maximize sum of radii subject to constraints.
    We use a penalty method with subgradient of the constraint violation.
    """
    best = arr.copy()
    best_obj = _objective(best)
    
    for it in range(n_iter):
        # Compute forces: push circles apart, allow radii to grow
        # Penalty for overlap: for each pair, if dist < r_i + r_j, push apart
        # Also penalty for going outside [0,1]x[0,1]
        
        # Subgradient of penalty w.r.t. positions and radii
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        # Pairwise overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = arr[i, 0] - arr[j, 0]
                dy = arr[i, 1] - arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = arr[i, 2] + arr[j, 2] - dist
                if overlap > 0:
                    # Push apart
                    if dist > 1e-10:
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                    else:
                        angle = np.random.rand() * 2 * np.pi
                        fx = overlap * np.cos(angle)
                        fy = overlap * np.sin(angle)
                    grad_x[i] += fx
                    grad_y[i] += fy
                    grad_x[j] -= fx
                    grad_y[j] -= fy
                    # Reduce radii if overlapping
                    grad_r[i] -= overlap * 0.5
                    grad_r[j] -= overlap * 0.5
        
        # Boundary constraints: keep circles inside [0,1]x[0,1]
        for i in range(N):
            # Left
            viol = arr[i, 2] - arr[i, 0]
            if viol > 0:
                grad_x[i] += viol
                grad_r[i] -= viol
            # Right
            viol = arr[i, 0] + arr[i, 2] - 1.0
            if viol > 0:
                grad_x[i] -= viol
                grad_r[i] -= viol
            # Bottom
            viol = arr[i, 2] - arr[i, 1]
            if viol > 0:
                grad_y[i] += viol
                grad_r[i] -= viol
            # Top
            viol = arr[i, 1] + arr[i, 2] - 1.0
            if viol > 0:
                grad_y[i] -= viol
                grad_r[i] -= viol
        
        # We want to maximize sum of radii, so add positive gradient for radii
        # but penalize constraint violations
        # Update: move in direction of (grad_r + alpha * ones) for radii
        # and in direction of grad for positions (to reduce violations)
        
        # Scale step
        s = step_size
        arr[:, 0] += s * grad_x
        arr[:, 1] += s * grad_y
        arr[:, 2] += s * (grad_r + 0.5)  # encourage radius growth
        
        # Clamp radii to be positive
        arr[:, 2] = np.maximum(arr[:, 2], 1e-6)
        
        # Evaluate
        obj = _objective(arr)
        if obj > best_obj:
            best_obj = obj
            best = arr.copy()
        
        # Decay step size
        step_size *= 0.99
    
    return best

def _simulated_annealing(arr, n_steps=2000):
    """Simple SA to improve placement."""
    best = arr.copy()
    best_obj = _objective(best)
    curr = arr.copy()
    curr_obj = _objective(curr)
    T = 0.05
    T_min = 0.001
    
    for step in range(n_steps):
        # Random perturbation
        idx = np.random.randint(N)
        dx = (np.random.rand() - 0.5) * 0.05
        dy = (np.random.rand() - 0.5) * 0.05
        dr = (np.random.rand() - 0.5) * 0.02
        new = curr.copy()
        new[idx, 0] += dx
        new[idx, 1] += dy
        new[idx, 2] = max(new[idx, 2] + dr, 1e-6)
        
        new_obj = _objective(new)
        
        if new_obj > curr_obj or (np.random.rand() < np.exp((new_obj - curr_obj) / max(T, 1e-10))):
            curr = new
            curr_obj = new_obj
            if curr_obj > best_obj:
                best_obj = curr_obj
                best = curr.copy()
        
        T *= 0.999
        
        if time.time() > DEADLINE:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with grid initialization
    arr = _grid_init()
    best = arr.copy()
    best_obj = _objective(best)
    
    # Try multiple random restarts with gradient + SA
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        if r == 0:
            curr = _grid_init()
        else:
            curr = _random_init()
        
        # Gradient refinement
        curr = _gradient_step(curr, step_size=0.02, n_iter=100)
        
        # Simulated annealing
        curr = _simulated_annealing(curr, n_steps=1000)
        
        obj = _objective(curr)
        if obj > best_obj:
            best_obj = obj
            best = curr.copy()
        
        if time.time() > DEADLINE - 10:
            break
    
    # Final validation
    if not _validate(best):
        # Fallback to simple grid
        best = _grid_init()
    
    return best
