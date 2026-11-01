"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (i.e., 2*(w+h) <= 4 => w+h <= 2).
    Maximize sum of radii.
    Returns array of shape (N, 3) with [x, y, r] for each circle.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Variables: x1..xN, y1..yN, r1..rN, w, h
    # Total variables: 3*N + 2 = 65
    # Constraints:
    # 1. w + h <= 2 (perimeter constraint)
    # 2. 0 <= xi <= w - ri  (for all i)
    # 3. 0 <= yi <= h - ri  (for all i)
    # 4. (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 (for all i<j)
    # 5. ri >= 0
    
    # Strategy: Use a heuristic initial guess based on grid packing,
    # then refine with nonlinear optimization using penalty or SLSQP.
    
    # Initial guess: grid-like arrangement
    # Try different grid configurations and pick the best initial sum of radii
    
    def grid_packing(cols, rows):
        """Generate initial packing in a cols x rows grid within unit square [0,1]^2."""
        if cols * rows < N:
            return None
        xs = []
        ys = []
        rs = []
        # We'll place circles in a grid, with some extra circles added
        # First fill the grid, then add extras
        positions = []
        for i in range(cols):
            for j in range(rows):
                positions.append((i, j))
        
        # Shuffle to distribute extras better? No, keep order
        # Actually, let's just use first N positions
        for idx in range(N):
            i, j = positions[idx]
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            # Radius limited by distance to neighbors
            # In a unit square, max radius is min(1/(2*cols), 1/(2*rows)) / 2
            # But we want to maximize sum, so start with a reasonable radius
            r = 0.5 / max(cols, rows) * 0.8
            xs.append(x)
            ys.append(y)
            rs.append(r)
        
        return np.array(xs), np.array(ys), np.array(rs)
    
    # Try several grid configurations
    best_init = None
    best_init_sum = -1
    
    for cols in range(3, 8):
        for rows in range(3, 8):
            if cols * rows >= N:
                result = grid_packing(cols, rows)
                if result is not None:
                    xs, ys, rs = result
                    # Scale to fit in rectangle with w+h=2
                    # Start with w=1, h=1
                    # The grid is in [0,1]x[0,1], so w=1, h=1 works if we scale radii
                    # But radii are also in [0,1] scale
                    # Sum of radii in this scale
                    s = np.sum(rs)
                    if s > best_init_sum:
                        best_init_sum = s
                        best_init = (xs.copy(), ys.copy(), rs.copy(), 1.0, 1.0)
    
    if best_init is None:
        # Fallback
        xs = np.linspace(0.1, 0.9, N)
        ys = np.linspace(0.1, 0.9, N)
        rs = np.full(N, 0.04)
        best_init = (xs, ys, rs, 1.0, 1.0)
    
    xs0, ys0, rs0, w0, h0 = best_init
    
    # Define the optimization problem
    # Variables: [x1..xN, y1..yN, r1..rN, w, h]
    def unpack(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = v[3*N+1]
        return x, y, r, w, h
    
    def objective(v):
        x, y, r, w, h = unpack(v)
        # Maximize sum of radii => minimize negative sum
        # Also add penalty for violating constraints to help convergence
        # Perimeter penalty
        perim_viol = max(0, w + h - 2)
        # Boundary penalties
        bound_pen = 0
        for i in range(N):
            bound_pen += max(0, -x[i]) + max(0, x[i] - (w - r[i]))
            bound_pen += max(0, -y[i]) + max(0, y[i] - (h - r[i]))
        # Overlap penalties
        overlap_pen = 0
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                min_dist = r[i] + r[j]
                if dist2 < min_dist**2:
                    overlap_pen += (min_dist - np.sqrt(dist2))**2
        # Radius non-negativity
        r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = 100 * (perim_viol**2 + bound_pen**2 + overlap_pen**2 + r_pen**2)
        return -np.sum(r) + penalty
    
    def gradient(v):
        # Numerical gradient for simplicity
        eps = 1e-7
        grad = np.zeros_like(v)
        f0 = objective(v)
        for i in range(len(v)):
            v_pert = v.copy()
            v_pert[i] += eps
            grad[i] = (objective(v_pert) - f0) / eps
        return grad
    
    # Constraints for SLSQP
    # Inequality constraints: g(v) >= 0
    def constraints(v):
        x, y, r, w, h = unpack(v)
        cons = []
        # Perimeter: w + h <= 2 => 2 - w - h >= 0
        cons.append(2 - w - h)
        # w, h positive
        cons.append(w - 0.1)
        cons.append(h - 0.1)
        # Boundary constraints
        for i in range(N):
            cons.append(x[i])  # x[i] >= 0
            cons.append(y[i])  # y[i] >= 0
            cons.append(w - r[i] - x[i])  # x[i] <= w - r[i]
            cons.append(h - r[i] - y[i])  # y[i] <= h - r[i]
            cons.append(r[i])  # r[i] >= 0
        # Overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append((x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2)
        return cons
    
    # Initial point
    v0 = np.concatenate([xs0, ys0, rs0, [w0, h0]])
    
    # Try SLSQP
    best_v = v0.copy()
    best_obj = -np.sum(rs0)
    
    # First, try to improve with SLSQP
    try:
        res = minimize(objective, v0, method='SLSQP', 
                      constraints=[{'type': 'ineq', 'fun': lambda v: np.array(constraints(v))}],
                      options={'maxiter': 200, 'ftol': 1e-8})
        if res.success or res.fun < objective(v0):
            v_new = res.x
            obj_new = -np.sum(v_new[2*N:3*N])
            # Check constraints
            cons_vals = np.array(constraints(v_new))
            if np.min(cons_vals) > -1e-4:
                best_v = v_new
                best_obj = obj_new
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline:
        x, y, r, w, h = unpack(best_v)
        # Ensure validity
        # Clip to valid region
        for i in range(N):
            r[i] = max(0.001, r[i])
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
        # Scale if perimeter violated
        if w + h > 2:
            scale = 2 / (w + h)
            x *= scale
            y *= scale
            r *= scale
            w *= scale
            h *= scale
        return np.column_stack([x, y, r])
    
    # Now try a local refinement with a different approach
    # Use the current best as starting point for a more careful optimization
    # with a smooth surrogate for the overlap constraints
    
    def smooth_objective(v):
        x, y, r, w, h = unpack(v)
        # Maximize sum of radii
        # Soft penalty for overlaps using log-sum-exp
        overlap_pen = 0
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                min_dist = r[i] + r[j]
                violation = min_dist - np.sqrt(dist2 + 1e-12)
                if violation > 0:
                    overlap_pen += violation**2
        # Boundary
        bound_pen = 0
        for i in range(N):
            bound_pen += max(0, -x[i])**2 + max(0, x[i] - (w - r[i]))**2
            bound_pen += max(0, -y[i])**2 + max(0, y[i] - (h - r[i]))**2
        # Perimeter
        perim_pen = max(0, w + h - 2)**2
        # Radius
        r_pen = np.sum(np.maximum(0, -r)**2)
        
        penalty = 50 * (overlap_pen + bound_pen + perim_pen + r_pen)
        return -np.sum(r) + penalty
    
    try:
        res2 = minimize(smooth_objective, best_v, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if res2.fun < smooth_objective(best_v):
            v_new = res2.x
            obj_new = -np.sum(v_new[2*N:3*N])
            cons_vals = np.array(constraints(v_new))
            if np.min(cons_vals) > -1e-3:
                best_v = v_new
                best_obj = obj_new
    except Exception as e:
        pass
    
    # Final check and cleanup
    x, y, r, w, h = unpack(best_v)
    
    # Ensure all constraints are satisfied
    # Scale if needed
    if w + h > 2:
        scale = 2 / (w + h)
        x *= scale
        y *= scale
        r *= scale
        w *= scale
        h *= scale
    
    # Clip positions and radii
    for i in range(N):
        r[i] = max(0.0001, r[i])
        x[i] = np.clip(x[i], r[i], w - r[i])
        y[i] = np.clip(y[i], r[i], h - r[i])
    
    # Verify no overlaps after clipping (if overlaps exist, reduce radii)
    for _ in range(10):
        overlap = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j]:
                    overlap = True
                    # Reduce both radii slightly
                    factor = dist / (r[i] + r[j]) * 0.99
                    r[i] *= factor
                    r[j] *= factor
                    # Re-clip
                    x[i] = np.clip(x[i], r[i], w - r[i])
                    y[i] = np.clip(y[i], r[i], h - r[i])
                    x[j] = np.clip(x[j], r[j], w - r[j])
                    y[j] = np.clip(y[j], r[j], h - r[j])
        if not overlap:
            break
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
