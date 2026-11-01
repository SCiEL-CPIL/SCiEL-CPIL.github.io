"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(arr):
    """Validate circles and compute sum of radii. Returns (valid, value)."""
    if arr is None:
        return False, 0.0
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (N, 3):
        return False, 0.0
    if np.any(~np.isfinite(arr)):
        return False, 0.0
    
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False, 0.0
    
    # Bounding box
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    W = xmax - xmin
    H = ymax - ymin
    if W <= 0 or H <= 0:
        return False, 0.0
    
    # Perimeter constraint
    if 2 * (W + H) > 4 + 1e-9:
        return False, 0.0
    
    # Circle containment in bounding box
    if np.any(x - r < xmin - 1e-9) or np.any(x + r > xmax + 1e-9):
        return False, 0.0
    if np.any(y - r < ymin - 1e-9) or np.any(y + r > ymax + 1e-9):
        return False, 0.0
    
    # Pairwise disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False, 0.0
    
    return True, float(np.sum(r))


def _grid_init():
    """Create initial grid packing."""
    # Try different grid arrangements
    best_val = 0
    best_arr = None
    
    for cols in range(3, 8):
        rows = int(np.ceil(N / cols))
        # Create grid
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for idx in range(N):
            ci = idx % cols
            ri = idx // cols
            x[idx] = (ci + 0.5) / cols
            y[idx] = (ri + 0.5) / rows
            r[idx] = 0.5 / max(cols, rows) - 1e-4
        
        arr = np.column_stack([x, y, r])
        valid, val = _validate_and_compute(arr)
        if valid and val > best_val:
            best_val = val
            best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: single row
        x = np.linspace(0.5/N + 0.05, 1 - 0.5/N - 0.05, N)
        y = np.full(N, 0.5)
        r = np.full(N, 0.04)
        best_arr = np.column_stack([x, y, r])
    
    return best_arr


def _optimize_slsqp(arr, max_iter=500, tol=1e-8):
    """Optimize circle packing using SLSQP."""
    n = arr.shape[0]
    
    def objective(v):
        # Maximize sum of radii = minimize negative sum
        return -v[2*n:]
    
    def grad_objective(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g
    
    constraints = []
    
    # Perimeter constraint: 2*(W+H) <= 4
    # W = max(x) - min(x), H = max(y) - min(y)
    # This is non-smooth, so we use a smooth approximation or handle via bounds
    
    # For SLSQP, we need smooth constraints. Let's use the fact that in optimal
    # packing, circles touch walls. We'll use auxiliary variables for bbox.
    # Actually, let's reformulate: introduce W, H as variables with constraints
    
    # Simpler approach: use the fact that at optimum, the bounding box is tight.
    # We'll add constraints that ensure circles are within [xmin, xmin+W] x [ymin, ymin+H]
    # with 2*(W+H) <= 4.
    
    # Variables: x[n], y[n], r[n], xmin, ymin, W, H
    # Total: 3n + 4 variables
    
    def obj2(v):
        r = v[2*n:3*n]
        return -np.sum(r)
    
    def grad_obj2(v):
        g = np.zeros(3*n + 4)
        g[2*n:3*n] = -1.0
        return g
    
    # Constraints:
    # 1. 2*(W+H) <= 4  =>  W + H <= 2
    # 2. For each circle i: xmin + r[i] <= x[i] <= xmin + W - r[i]
    #    ymin + r[i] <= y[i] <= ymin + H - r[i]
    # 3. For each pair (i,j): (x[i]-x[j])^2 + (y[i]-y[j])^2 >= (r[i]+r[j])^2
    #    We use a smooth approximation: (x[i]-x[j])^2 + (y[i]-y[j])^2 - (r[i]+r[j])^2 >= 0
    
    # For SLSQP, inequality constraints are of form c(v) >= 0
    
    cons = []
    
    # Perimeter: W + H <= 2  =>  2 - W - H >= 0
    def c_perim(v):
        return 2.0 - v[3*n] - v[3*n+1]
    def gc_perim(v):
        g = np.zeros(3*n+4)
        g[3*n] = -1.0
        g[3*n+1] = -1.0
        return g
    cons.append({'type': 'ineq', 'fun': c_perim, 'jac': gc_perim})
    
    # Circle-wall constraints
    for i in range(n):
        # x[i] - xmin >= r[i]
        def c_xmin(v, i=i):
            return v[i] - v[3*n] - v[2*n+i]
        def gc_xmin(v, i=i):
            g = np.zeros(3*n+4)
            g[i] = 1.0
            g[3*n] = -1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_xmin, 'jac': gc_xmin})
        
        # xmin + W - x[i] >= r[i]
        def c_xmax(v, i=i):
            return v[3*n] + v[3*n] - v[i] - v[2*n+i]  # Wait, this is wrong
            # Should be: xmin + W - x[i] >= r[i]
            # => xmin + W - x[i] - r[i] >= 0
            return v[3*n] + v[3*n+1] - v[i] - v[2*n+i]
        def gc_xmax(v, i=i):
            g = np.zeros(3*n+4)
            g[i] = -1.0
            g[3*n] = 1.0
            g[3*n+1] = 1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_xmax, 'jac': gc_xmax})
        
        # y[i] - ymin >= r[i]
        def c_ymin(v, i=i):
            return v[n+i] - v[3*n+1] - v[2*n+i]
        def gc_ymin(v, i=i):
            g = np.zeros(3*n+4)
            g[n+i] = 1.0
            g[3*n+1] = -1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_ymin, 'jac': gc_ymin})
        
        # ymin + H - y[i] >= r[i]
        def c_ymax(v, i=i):
            return v[3*n+1] + v[3*n+1] - v[n+i] - v[2*n+i]
            # Should be: ymin + H - y[i] >= r[i]
            # => ymin + H - y[i] - r[i] >= 0
            return v[3*n+1] + v[3*n+1] - v[n+i] - v[2*n+i]
            # Wait, H is v[3*n+1], so ymin + H = v[3*n+1] + v[3*n+1]? No.
            # ymin is v[3*n+1], H is v[3*n+2]? Let me redefine.
            # Let's use: v[3*n] = xmin, v[3*n+1] = ymin, v[3*n+2] = W, v[3*n+3] = H
            return v[3*n+1] + v[3*n+3] - v[n+i] - v[2*n+i]
        def gc_ymax(v, i=i):
            g = np.zeros(3*n+4)
            g[n+i] = -1.0
            g[3*n+1] = 1.0
            g[3*n+3] = 1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_ymax, 'jac': gc_ymax})
    
    # I made an error in variable indexing. Let me restart with clearer indexing.
    # Variables: [x(0..n-1), y(n..2n-1), r(2n..3n-1), xmin(3n), ymin(3n+1), W(3n+2), H(3n+3)]
    
    return None  # This is getting complex; let's use a simpler approach


def _optimize_simple(arr, max_iter=300):
    """Simple SLSQP optimization without bbox variables."""
    n = arr.shape[0]
    x0 = arr.flatten()
    
    def objective(v):
        return -np.sum(v[2*n:])
    
    def grad(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g
    
    # We need to handle the perimeter constraint. Since the bounding box depends
    # on the positions, this is tricky. Let's use a penalty approach or fix
    # the bounding box.
    
    # Alternative: fix the rectangle to be 1x1 (perimeter 4) and optimize
    # circles within it. Then the perimeter is automatically 4.
    
    # Let's try: fix W=1, H=1, xmin=0, ymin=0. Then perimeter = 4.
    # Constraints:
    # 0 <= x[i] - r[i], x[i] + r[i] <= 1
    # 0 <= y[i] - r[i], y[i] + r[i] <= 1
    # Pairwise: (x[i]-x[j])^2 + (y[i]-y[j])^2 >= (r[i]+r[j])^2
    
    cons = []
    
    # Wall constraints
    for i in range(n):
        # x[i] - r[i] >= 0
        def c1(v, i=i):
            return v[i] - v[2*n+i]
        def gc1(v, i=i):
            g = np.zeros(3*n)
            g[i] = 1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c1, 'jac': gc1})
        
        # 1 - x[i] - r[i] >= 0
        def c2(v, i=i):
            return 1.0 - v[i] - v[2*n+i]
        def gc2(v, i=i):
            g = np.zeros(3*n)
            g[i] = -1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c2, 'jac': gc2})
        
        # y[i] - r[i] >= 0
        def c3(v, i=i):
            return v[n+i] - v[2*n+i]
        def gc3(v, i=i):
            g = np.zeros(3*n)
            g[n+i] = 1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c3, 'jac': gc3})
        
        # 1 - y[i] - r[i] >= 0
        def c4(v, i=i):
            return 1.0 - v[n+i] - v[2*n+i]
        def gc4(v, i=i):
            g = np.zeros(3*n)
            g[n+i] = -1.0
            g[2*n+i] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c4, 'jac': gc4})
    
    # Pairwise constraints: use smooth approximation
    # (x[i]-x[j])^2 + (y[i]-y[j])^2 - (r[i]+r[j])^2 >= 0
    # This is not convex, but SLSQP can handle it locally.
    
    for i in range(n):
        for j in range(i+1, n):
            def c_pair(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                rsum = v[2*n+i] + v[2*n+j]
                return dx*dx + dy*dy - rsum*rsum
            def gc_pair(v, i=i, j=j):
                g = np.zeros(3*n)
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                rsum = v[2*n+i] + v[2*n+j]
                g[i] = 2*dx
                g[j] = -2*dx
                g[n+i] = 2*dy
                g[n+j] = -2*dy
                g[2*n+i] = -2*rsum
                g[2*n+j] = -2*rsum
                return g
            cons.append({'type': 'ineq', 'fun': c_pair, 'jac': gc_pair})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.49))  # r
    
    res = minimize(objective, x0, jac=grad, bounds=bounds, constraints=cons,
                   method='SLSQP', options={'maxiter': max_iter, 'ftol': 1e-8})
    
    if res.success:
        arr_opt = res.x.reshape(n, 3)
        # Scale to ensure perimeter <= 4
        xmin, xmax = arr_opt[:, 0].min(), arr_opt[:, 0].max()
        ymin, ymax = arr_opt[:, 1].min(), arr_opt[:, 1].max()
        W = xmax - xmin
        H = ymax - ymin
        if W > 0 and H > 0:
            scale = 2.0 / (W + H)
            if scale < 1.0:
                # Scale down
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                arr_opt[:, 0] = cx + (arr_opt[:, 0] - cx) * scale
                arr_opt[:, 1] = cy + (arr_opt[:, 1] - cy) * scale
                arr_opt[:, 2] = arr_opt[:, 2] * scale
    
    return arr_opt


def _polish(arr, max_iter=200):
    """Polish the packing by trying to increase radii."""
    n = arr.shape[0]
    
    # Try to increase all radii uniformly
    for _ in range(max_iter):
        # Try increasing all radii by a small amount
        delta = 1e-4
        arr_new = arr.copy()
        arr_new[:, 2] += delta
        
        # Check if still valid
        valid, _ = _validate_and_compute(arr_new)
        if not valid:
            # Try smaller delta
            delta /= 10
            if delta < 1e-7:
                break
            arr_new[:, 2] += delta
            valid, _ = _validate_and_compute(arr_new)
            if not valid:
                break
        else:
            arr = arr_new
    
    return arr


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial grid packing
    arr = _grid_init()
    
    # Optimize
    if time.time() < deadline:
        arr = _optimize_simple(arr, max_iter=200)
    
    # Polish
    if time.time() < deadline:
        arr = _polish(arr)
    
    # Final validation
    valid, val = _validate_and_compute(arr)
    if not valid:
        # Return a valid fallback
        arr = _grid_init()
    
    return arr

# EVOLVE-BLOCK-END
