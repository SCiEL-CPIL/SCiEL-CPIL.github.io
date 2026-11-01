"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_valid(arr, eps=1e-9):
    """Check if circles are disjoint and inside unit square."""
    n = arr.shape[0]
    for i in range(n):
        x, y, r = arr[i]
        if r < -eps:
            return False
        if x - r < -eps or x + r > 1 + eps:
            return False
        if y - r < -eps or y + r > 1 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True

def _penalty(arr):
    """Compute violation penalty for constraints."""
    n = arr.shape[0]
    penalty = 0.0
    # Boundary constraints
    for i in range(n):
        x, y, r = arr[i]
        if r < 0:
            penalty += (-r) ** 2
        if x - r < 0:
            penalty += (-(x - r)) ** 2
        if x + r > 1:
            penalty += (x + r - 1) ** 2
        if y - r < 0:
            penalty += (-(y - r)) ** 2
        if y + r > 1:
            penalty += (y + r - 1) ** 2
    # Disjointness constraints
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj:
                penalty += (ri + rj - dist) ** 2
    return penalty

def _objective(params, n):
    """Objective: minimize negative sum of radii + penalty."""
    arr = params.reshape(n, 3)
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Negative sum of radii (we want to maximize)
    obj = -np.sum(r)
    
    # Add penalty
    obj += 10.0 * _penalty(arr)
    
    return obj

def _objective_grad(params, n):
    """Gradient of objective."""
    arr = params.reshape(n, 3)
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Gradient of penalty
    pen_factor = 20.0  # 2 * 10
    
    # Boundary penalties
    for i in range(n):
        xi, yi, ri = x[i], y[i], r[i]
        
        # r < 0
        if ri < 0:
            grad[3*i + 2] += pen_factor * (-ri)
        
        # x - r < 0
        if xi - ri < 0:
            val = -(xi - ri)
            grad[3*i + 0] += pen_factor * val  # d/dx of (-(x-r))^2 = -2(x-r), but we have (-(x-r))^2, so d/dx = -2(x-r) * (-1) = 2(x-r) ... wait
            # Let's be careful: penalty term is (-(x-r))^2 = (x-r)^2 when x-r<0
            # d/dx (x-r)^2 = 2(x-r)
            # d/dr (x-r)^2 = -2(x-r)
            grad[3*i + 0] += pen_factor * (xi - ri)
            grad[3*i + 2] += pen_factor * (-(xi - ri))
        
        # x + r > 1
        if xi + ri > 1:
            val = xi + ri - 1
            # d/dx (x+r-1)^2 = 2(x+r-1)
            # d/dr (x+r-1)^2 = 2(x+r-1)
            grad[3*i + 0] += pen_factor * val
            grad[3*i + 2] += pen_factor * val
        
        # y - r < 0
        if yi - ri < 0:
            val = yi - ri
            grad[3*i + 1] += pen_factor * val
            grad[3*i + 2] += pen_factor * (-(yi - ri))
        
        # y + r > 1
        if yi + ri > 1:
            val = yi + ri - 1
            grad[3*i + 1] += pen_factor * val
            grad[3*i + 2] += pen_factor * val
    
    # Disjointness penalties
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = x[i], y[i], r[i]
            xj, yj, rj = x[j], y[j], r[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
            if dist < ri + rj:
                val = ri + rj - dist
                # penalty term: (ri + rj - dist)^2
                # d/dri = 2(ri+rj-dist) * (1 - 0) ... wait, d(dist)/dri = 0
                # d/deri of (ri+rj-dist)^2 = 2(ri+rj-dist)
                # d/dxi of (ri+rj-dist)^2 = 2(ri+rj-dist) * (-d(dist)/dxi) = 2(ri+rj-dist) * (-dx/dist)
                grad[3*i + 2] += pen_factor * val
                grad[3*j + 2] += pen_factor * val
                grad[3*i + 0] += pen_factor * val * (-dx / dist)
                grad[3*j + 0] += pen_factor * val * (dx / dist)
                grad[3*i + 1] += pen_factor * val * (-dy / dist)
                grad[3*j + 1] += pen_factor * val * (dy / dist)
    
    return grad

def _push_apart(arr, n, iters=50, strength=1.0):
    """Local repair: push overlapping circles apart."""
    arr = arr.copy()
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = arr[i]
                xj, yj, rj = arr[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Push apart along the line connecting centers
                    nx = dx / dist
                    ny = dy / dist
                    push = overlap * 0.5 * strength
                    arr[i, 0] += nx * push
                    arr[i, 1] += ny * push
                    arr[j, 0] -= nx * push
                    arr[j, 1] -= ny * push
                    moved = True
        # Clamp to boundaries
        for i in range(n):
            ri = arr[i, 2]
            arr[i, 0] = np.clip(arr[i, 0], ri, 1 - ri)
            arr[i, 1] = np.clip(arr[i, 1], ri, 1 - ri)
        if not moved:
            break
    return arr

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Initial grid-based guess
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    initial = np.array(cells[:N])
    
    best_arr = initial.copy()
    best_score = np.sum(initial[:, 2])
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate random initial configuration
        if restart == 0:
            arr = initial.copy()
        else:
            # Random positions with small radii
            arr = np.zeros((N, 3))
            arr[:, 0] = np.random.uniform(0.1, 0.9, N)
            arr[:, 1] = np.random.uniform(0.1, 0.9, N)
            arr[:, 2] = np.random.uniform(0.02, 0.15, N)
        
        # Push apart to get feasible starting point
        arr = _push_apart(arr, N, iters=30, strength=0.8)
        
        # Clamp radii to be non-negative
        arr[:, 2] = np.maximum(arr[:, 2], 0.0)
        
        # Optimize using L-BFGS-B
        params = arr.flatten()
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
        
        try:
            result = minimize(
                _objective,
                params,
                args=(N,),
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < _objective(params, N):
                new_arr = result.x.reshape(N, 3)
                new_arr[:, 2] = np.maximum(new_arr[:, 2], 0.0)
                
                # Check validity
                if _check_valid(new_arr, eps=1e-8):
                    new_score = np.sum(new_arr[:, 2])
                    if new_score > best_score:
                        best_score = new_score
                        best_arr = new_arr.copy()
        except Exception:
            pass
    
    # Final polish: try to increase radii slightly
    if time.time() - start_time < deadline - 5:
        for _ in range(50):
            if time.time() - start_time > deadline - 2:
                break
            # Try increasing all radii by a small amount
            test_arr = best_arr.copy()
            test_arr[:, 2] += 1e-5
            if _check_valid(test_arr, eps=1e-10):
                best_arr = test_arr
            else:
                break
    
    return best_arr
# EVOLVE-BLOCK-END
