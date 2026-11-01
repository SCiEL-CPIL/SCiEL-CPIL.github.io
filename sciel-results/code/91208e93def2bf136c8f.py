"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import random

N = 21
DEADLINE = 120.0  # seconds wall-clock limit


def _check_validity(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return False
    
    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: 2*(w + h) <= 4 => w + h <= 2
    if w + h > 2.0 + tol:
        return False
    
    # Circles must be inside their bounding box (trivially true by construction of bbox)
    # but we need to ensure they don't extend outside... actually the bounding box IS the
    # minimal rectangle containing all circles, so by definition each circle is inside.
    # However, the "bounding rectangle" in the problem likely means the axis-aligned
    # rectangle that contains all circles. Let me re-read.
    
    # "disjoint circles whose bounding rectangle has perimeter at most 4"
    # The bounding rectangle is the minimal axis-aligned rectangle containing all circles.
    # So: for each circle, x_i - r_i >= x_min and x_i + r_i <= x_max, etc.
    # This is automatically satisfied if we define x_min = min(x_i - r_i) and x_max = max(x_i + r_i).
    # But the perimeter is 2*((x_max - x_min) + (y_max - y_min)) where these are the
    # extents including radii.
    
    # Let me reconsider: the bounding rectangle of the set of circles is:
    # [min(x_i - r_i), max(x_i + r_i)] x [min(y_i - r_i), max(y_i + r_i)]
    # Perimeter = 2 * ((max(x_i+r_i) - min(x_i-r_i)) + (max(y_i+r_i) - min(y_i-r_i)))
    
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    # Disjoint interiors: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _objective(arr):
    """Negative sum of radii (to minimize)."""
    return -np.sum(arr[:, 2])


def _initial_grid():
    """Create a reasonable initial grid of 21 circles."""
    # Use a 5x5 grid, pick 21 positions
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append((i, j))
    # Shuffle and take first 21
    random.seed(42)
    random.shuffle(positions)
    positions = positions[:N]
    
    # Place in a unit square initially, will be scaled
    arr = np.zeros((N, 3))
    for idx, (i, j) in enumerate(positions):
        arr[idx, 0] = (i + 0.5) / side
        arr[idx, 1] = (j + 0.5) / side
        arr[idx, 2] = 0.15  # initial radius
    
    return arr


def _pack_with_optimization(arr_init, max_iter=200, tol=1e-8):
    """Optimize circle positions and radii to maximize sum of radii subject to constraints."""
    n = arr_init.shape[0]
    
    # Variables: x_1..x_n, y_1..y_n, r_1..r_n, W, H
    # where W = x_max - x_min + 2*margin... actually let's use:
    # Let's define the bounding box explicitly.
    
    # Alternative approach: use a fixed rectangle of perimeter 4, say width w and height h
    # with 2*(w+h) = 4, so w+h=2. Try several aspect ratios.
    
    best_arr = arr_init.copy()
    best_obj = _objective(arr_init)
    
    # Try different aspect ratios for the rectangle
    for aspect in [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75]:
        # w + h = 2, w = aspect * h => h*(aspect+1) = 2 => h = 2/(aspect+1), w = 2*aspect/(aspect+1)
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Now pack n circles in rectangle [0,w] x [0,h]
        # Variables: x_i, y_i, r_i for i in 1..n
        # Constraints:
        # 1. x_i - r_i >= 0, x_i + r_i <= w
        # 2. y_i - r_i >= 0, y_i + r_i <= h
        # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i != j
        # 4. r_i >= 0
        
        def obj(vars_):
            rs = vars_[2*n:]
            return -np.sum(rs)
        
        # Initial guess: spread circles in the rectangle
        x0 = np.zeros(3 * n)
        # Grid initial positions
        side = int(np.ceil(np.sqrt(n)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x0[count * 3] = (i + 0.5) * w / side
                x0[count * 3 + 1] = (j + 0.5) * h / side
                x0[count * 3 + 2] = min(w, h) / (2 * side) * 0.9
                count += 1
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, w))       # x_i
            bounds.append((0, h))       # y_i
            bounds.append((1e-6, min(w, h)/2))  # r_i
        
        # Use SLSQP with nonlinear constraints
        constraints = []
        
        # Boundary constraints (linear)
        # x_i - r_i >= 0 => -x_i + r_i <= 0
        # x_i + r_i <= w => x_i + r_i - w <= 0
        # y_i - r_i >= 0 => -y_i + r_i <= 0
        # y_i + r_i <= h => y_i + r_i - h <= 0
        
        # We'll use inequality constraints in the form: func >= 0
        # SLSQP expects: 'ineq' means func >= 0
        
        for i in range(n):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[i*3] - v[i*3+2]
            })
            # w - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, w=w: w - v[i*3] - v[i*3+2]
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]
            })
            # h - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, h=h: h - v[i*3+1] - v[i*3+2]
            })
        
        # Pairwise distance constraints: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i*3]-v[j*3])**2 + (v[i*3+1]-v[j*3+1])**2 - (v[i*3+2]+v[j*3+2])**2
                })
        
        # This creates too many constraints for SLSQP with 63 variables.
        # Let's use a different approach: penalty method or sequential optimization.
        
        # Alternative: use a simpler iterative approach
        # Start with grid, then iteratively adjust radii and positions
        
        # Actually, let's try a different strategy: 
        # 1. Fix the rectangle
        # 2. Use a simple local search / gradient-based approach
        
        # Let's use a simpler method: start with equal radii and optimize
        r0 = min(w, h) / (2 * side) * 0.85
        x0 = np.zeros(3 * n)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x0[count * 3] = (i + 0.5) * w / side
                x0[count * 3 + 1] = (j + 0.5) * h / side
                x0[count * 3 + 2] = r0
                count += 1
        
        # Simple local optimization using scipy
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_obj:
                arr_candidate = result.x.reshape(n, 3)
                if _check_validity(arr_candidate):
                    best_obj = result.fun
                    best_arr = arr_candidate
        except Exception:
            pass
    
    return best_arr, -best_obj


def _local_search(arr, max_iters=500, step_size=0.01):
    """Simple local search to improve the packing."""
    n = arr.shape[0]
    best = arr.copy()
    best_sum = np.sum(best[:, 2])
    
    for it in range(max_iters):
        # Try moving a random circle
        i = random.randint(0, n - 1)
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        dr = random.uniform(-step_size * 0.5, step_size * 0.5)
        
        arr_new = best.copy()
        arr_new[i, 0] += dx
        arr_new[i, 1] += dy
        arr_new[i, 2] = max(1e-6, arr_new[i, 2] + dr)
        
        if _check_validity(arr_new):
            new_sum = np.sum(arr_new[:, 2])
            if new_sum > best_sum:
                best = arr_new
                best_sum = new_sum
    
    return best, best_sum


def _grid_packing():
    """Try several grid-based packings."""
    results = []
    
    # 5x5 grid, remove 4 corners or edges
    for remove_pattern in [
        [(0,0), (0,4), (4,0), (4,4)],  # corners
        [(0,0), (0,4), (4,0), (4,4)],
        [(2,0), (2,4), (0,2), (4,2)],  # mid-edges
        [(0,0), (4,4), (0,4), (4,0)],
    ]:
        side = 5
        positions = []
        for i in range(side):
            for j in range(side):
                if (i, j) not in remove_pattern:
                    positions.append((i, j))
        
        if len(positions) < N:
            continue
        positions = positions[:N]
        
        # Scale to fit in a square of perimeter 4 (side 1)
        # Actually perimeter 4 means 2*(w+h)=4, for square w=h=1
        w = h = 1.0
        r = 0.5 / side - 0.005  # small gap
        
        arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            arr[idx, 0] = (i + 0.5) / side
            arr[idx, 1] = (j + 0.5) / side
            arr[idx, 2] = r
        
        if _check_validity(arr):
            results.append(arr)
    
    return results


def construct():
    start_time = time.time()
    
    # Strategy 1: Grid-based packing
    best_arr = None
    best_sum = 0.0
    
    # Try grid packings
    for arr in _grid_packing():
        s = np.sum(arr[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: simple grid
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.005
        best_sum = np.sum(best_arr[:, 2])
    
    # Strategy 2: Try optimization from multiple starting points
    deadline = start_time + DEADLINE - 5.0  # safety margin
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Different initial configurations
        if restart == 0:
            arr_init = best_arr.copy()
        elif restart == 1:
            # Hexagonal-like packing
            arr_init = np.zeros((N, 3))
            row_height = 0.18
            for idx in range(N):
                row = idx // 5
                col = idx % 5
                x_off = 0.09 if row % 2 == 1 else 0.0
                arr_init[idx, 0] = (col + 0.5) * 0.2 + x_off
                arr_init[idx, 1] = (row + 0.5) * row_height
                arr_init[idx, 2] = 0.08
        elif restart == 2:
            # Random initial
            arr_init = np.random.rand(N, 3)
            arr_init[:, 0] *= 0.8
            arr_init[:, 1] *= 0.8
            arr_init[:, 2] = 0.05 + np.random.rand(N) * 0.05
        elif restart == 3:
            # Concentric-ish
            arr_init = np.zeros((N, 3))
            arr_init[0] = [0.5, 0.5, 0.15]
            for idx in range(1, N):
                angle = 2 * np.pi * idx / (N - 1)
                radius = 0.3
                arr_init[idx, 0] = 0.5 + radius * np.cos(angle)
                arr_init[idx, 1] = 0.5 + radius * np.sin(angle)
                arr_init[idx, 2] = 0.08
        else:
            arr_init = best_arr.copy()
            arr_init[:, 0] += np.random.randn(N) * 0.02
            arr_init[:, 1] += np.random.randn(N) * 0.02
        
        # Local search
        if time.time() > deadline:
            break
        
        arr_improved, sum_improved = _local_search(arr_init, max_iters=200, step_size=0.02)
        if sum_improved > best_sum:
            best_sum = sum_improved
            best_arr = arr_improved
    
    # Final check and return
    if not _check_validity(best_arr):
        # Return a known-valid fallback
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
    
    return best_arr
