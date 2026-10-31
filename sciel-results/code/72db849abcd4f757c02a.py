"""Improved circle packing: gradient-based optimization with multiple restarts."""
import numpy as np
import time

N = 26


def _valid(arr):
    """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
    n = arr.shape[0]
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Inside unit square
    if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # Non-overlapping
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(arr):
    """Sum of radii, 0 if invalid."""
    if not _valid(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))


def _project_to_valid(arr, iters=2000, lr=0.01):
    """Project circles to be valid using iterative repulsion."""
    arr = arr.copy()
    n = arr.shape[0]
    for _ in range(iters):
        # Repel overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = arr[i, 0] - arr[j, 0]
                dy = arr[i, 1] - arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = arr[i, 2] + arr[j, 2]
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    arr[i, 0] += nx * overlap * 0.5
                    arr[i, 1] += ny * overlap * 0.5
                    arr[j, 0] -= nx * overlap * 0.5
                    arr[j, 1] -= ny * overlap * 0.5
        # Keep inside unit square
        for i in range(n):
            if arr[i, 0] < arr[i, 2]:
                arr[i, 0] = arr[i, 2]
            if arr[i, 0] > 1 - arr[i, 2]:
                arr[i, 0] = 1 - arr[i, 2]
            if arr[i, 1] < arr[i, 2]:
                arr[i, 1] = arr[i, 2]
            if arr[i, 1] > 1 - arr[i, 2]:
                arr[i, 1] = 1 - arr[i, 2]
    return arr


def _optimize(arr, max_iters=5000, lr=0.005):
    """Optimize sum of radii using gradient ascent with repulsion constraints."""
    arr = arr.copy()
    n = arr.shape[0]
    best_arr = arr.copy()
    best_score = _score(arr)
    
    for it in range(max_iters):
        x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        
        # Objective: maximize sum(r_i)
        # Constraint: dist(i,j) >= r_i + r_j, and circles inside [0,1]^2
        
        # For each pair, compute gradient of constraint violation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist > 1e-10:
                    if dist < min_dist:
                        # Violation: push apart and shrink
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        grad_x[i] += nx * overlap
                        grad_y[i] += ny * overlap
                        grad_x[j] -= nx * overlap
                        grad_y[j] -= ny * overlap
                        grad_r[i] -= overlap
                        grad_r[j] -= overlap
                    else:
                        # No violation, but we can still grow radii
                        # Gradient of r_i is 1, but constrained by neighbors
                        pass
        
        # Boundary constraints
        for i in range(n):
            if x[i] < r[i]:
                grad_x[i] += (r[i] - x[i])
                grad_r[i] -= (r[i] - x[i])
            if x[i] > 1 - r[i]:
                grad_x[i] -= (x[i] - (1 - r[i]))
                grad_r[i] -= (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                grad_y[i] += (r[i] - y[i])
                grad_r[i] -= (r[i] - y[i])
            if y[i] > 1 - r[i]:
                grad_y[i] -= (y[i] - (1 - r[i]))
                grad_r[i] -= (y[i] - (1 - r[i]))
        
        # Update
        arr[:, 0] += lr * grad_x
        arr[:, 1] += lr * grad_y
        arr[:, 2] += lr * grad_r
        
        # Project to valid
        arr = _project_to_valid(arr, iters=50, lr=lr)
        
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
        
        # Check convergence
        if it % 100 == 0 and s <= best_score * (1 - 1e-6):
            # Stagnation check
            pass
    
    return best_arr


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_arr = None
    best_score = 0.0
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    init_arr = np.array(cells[:N])
    
    # Try multiple random restarts
    np.random.seed(42)
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            arr = init_arr.copy()
        else:
            # Perturb the initial arrangement
            arr = init_arr.copy()
            noise = np.random.randn(N, 2) * 0.05
            arr[:, 0] += noise[:, 0]
            arr[:, 1] += noise[:, 1]
            # Slightly randomize radii
            arr[:, 2] *= (1 + np.random.randn(N) * 0.1)
            arr = _project_to_valid(arr, iters=500)
        
        # Optimize
        arr_opt = _optimize(arr, max_iters=3000, lr=0.003)
        s = _score(arr_opt)
        
        if s > best_score:
            best_score = s
            best_arr = arr_opt.copy()
        
        # Time check
        if time.time() > deadline - 10:
            break
    
    if best_arr is None:
        best_arr = init_arr
    
    # Final polish
    if time.time() < deadline - 5:
        best_arr = _project_to_valid(best_arr, iters=1000)
    
    return best_arr
