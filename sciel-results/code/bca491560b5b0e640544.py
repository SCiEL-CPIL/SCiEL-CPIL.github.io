"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin

    # We want to pack 21 circles in a rectangle of perimeter 4.
    # Let rectangle be w x h with 2(w+h) = 4 => w+h = 2.
    # We'll parameterize by aspect ratio r = w/h, so h = 2/(1+r), w = 2r/(1+r).
    #
    # Strategy: Use a force-directed / gradient-based optimization.
    # Variables: positions (x_i, y_i) and radii (r_i) for i=1..N.
    # Objective: maximize sum(r_i)
    # Constraints: 
    #   - Circles don't overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #   - Circles inside rectangle: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    #
    # We'll use a penalty method with a smooth surrogate for the overlap constraint.
    # For each pair, the constraint violation is max(0, (r_i+r_j)^2 - dist^2).
    # Smooth approximation: log-sum-exp or softplus.
    #
    # Actually, let's use a simpler approach: 
    # 1. Start with a grid layout.
    # 2. Use scipy.optimize.minimize with L-BFGS-B to maximize sum of radii
    #    subject to penalty terms for overlaps and boundary violations.
    #
    # But a cleaner approach: fix the rectangle aspect ratio, then optimize.
    # Try several aspect ratios and pick the best.

    def solve_for_aspect(ratio):
        """Solve for given aspect ratio. Returns best sum of radii found."""
        w = 2.0 * ratio / (1.0 + ratio)
        h = 2.0 / (1.0 + ratio)

        # Initialize: grid layout
        side = int(np.ceil(np.sqrt(N)))
        # Place circles in a grid
        xs = np.linspace(0.5 / side, w - 0.5 / side, side)
        ys = np.linspace(0.5 / side, h - 0.5 / side, side)
        # Create grid positions
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    positions.append([xs[i], ys[j]])
        positions = np.array(positions[:N])
        
        # Initial radii: small
        radii = np.full(N, 0.05)

        # Combined variable vector: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]

        def objective(v):
            """Negative sum of radii + penalty for violations."""
            if time.time() > deadline - 2.0:
                return 1e10  # Force termination
            
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            
            # Boundary penalties
            penalty = 0.0
            # Left boundary: x_i >= r_i
            viol = np.maximum(0, rs - xs)
            penalty += 100.0 * np.sum(viol**2)
            # Right boundary: x_i + r_i <= w
            viol = np.maximum(0, xs + rs - w)
            penalty += 100.0 * np.sum(viol**2)
            # Bottom boundary: y_i >= r_i
            viol = np.maximum(0, rs - ys)
            penalty += 100.0 * np.sum(viol**2)
            # Top boundary: y_i + r_i <= h
            viol = np.maximum(0, ys + rs - h)
            penalty += 100.0 * np.sum(viol**2)
            
            # Overlap penalties
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    if min_dist > 0:
                        viol = max(0, min_dist**2 - dist_sq)
                        penalty += 50.0 * viol**2
            
            return -np.sum(rs) + penalty

        def gradient(v):
            """Gradient of objective."""
            if time.time() > deadline - 2.0:
                return np.zeros(3*N)
            
            xs = v[0::3].copy()
            ys = v[1::3].copy()
            rs = v[2::3].copy()
            
            grad = np.zeros(3*N)
            
            # Objective gradient: -1 for each radius
            grad[2::3] -= 1.0
            
            # Boundary gradient contributions
            # Left: x_i >= r_i
            viol = np.maximum(0, rs - xs)
            grad[0::3] += 200.0 * viol * (-1)  # d/dx_i of (r_i - x_i)^2 when active
            grad[2::3] += 200.0 * viol * (1)   # d/dr_i of (r_i - x_i)^2 when active
            
            # Right: x_i + r_i <= w
            viol = np.maximum(0, xs + rs - w)
            grad[0::3] += 200.0 * viol * (1)
            grad[2::3] += 200.0 * viol * (1)
            
            # Bottom: y_i >= r_i
            viol = np.maximum(0, rs - ys)
            grad[1::3] += 200.0 * viol * (-1)
            grad[2::3] += 200.0 * viol * (1)
            
            # Top: y_i + r_i <= h
            viol = np.maximum(0, ys + rs - h)
            grad[1::3] += 200.0 * viol * (1)
            grad[2::3] += 200.0 * viol * (1)
            
            # Overlap gradient
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    if min_dist > 0:
                        viol = max(0.0, min_dist**2 - dist_sq)
                        if viol > 0:
                            # d/dx_i of (min_dist^2 - dist^2)^2 * 50
                            # = 50 * 2 * viol * (-2 * dx)
                            factor = 200.0 * viol
                            grad[3*i] += factor * (-dx)
                            grad[3*i+1] += factor * (-dy)
                            grad[3*i+2] += factor * (2 * min_dist)
                            grad[3*j] += factor * (dx)
                            grad[3*j+1] += factor * (dy)
                            grad[3*j+2] += factor * (2 * min_dist)
            
            return grad

        # Use L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return 0.0, x0
        
        if time.time() > deadline - 2.0:
            return 0.0, x0
        
        # Extract best solution
        xs = result.x[0::3]
        ys = result.x[1::3]
        rs = result.x[2::3]
        
        # Clamp radii to be positive and within bounds
        rs = np.maximum(0.01, rs)
        for i in range(N):
            rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
            if rs[i] < 0.01:
                rs[i] = 0.01
        
        # Check feasibility
        feasible = True
        for i in range(N):
            if xs[i] < rs[i] - 1e-6 or xs[i] + rs[i] > w + 1e-6:
                feasible = False
            if ys[i] < rs[i] - 1e-6 or ys[i] + rs[i] > h + 1e-6:
                feasible = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx*dx + dy*dy < (rs[i] + rs[j])**2 - 1e-6:
                    feasible = False
        
        if not feasible:
            return 0.0, result.x
        
        return np.sum(rs), result.x

    # Try several aspect ratios
    best_sum = 0.0
    best_x = None
    
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        if time.time() > deadline - 10.0:
            break
        
        s, x = solve_for_aspect(ratio)
        if s > best_sum:
            best_sum = s
            best_x = x
    
    # If no feasible solution found, fall back to grid
    if best_x is None or best_sum <= 0:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = min(x, w - x, y, h - y) - 1e-3
                    cells.append([x, y, max(r, 0.01)])
        return np.array(cells[:N])

    # Extract final solution
    xs = best_x[0::3]
    ys = best_x[1::3]
    rs = best_x[2::3]
    
    # Determine rectangle from the solution
    # We need to find w, h such that 2(w+h)=4 and all circles fit
    # Use the aspect ratio that gave the best solution
    # Re-derive w, h from the best solution
    max_x = np.max(xs + rs)
    min_x = np.min(xs - rs)
    max_y = np.max(ys + rs)
    min_y = np.min(ys - rs)
    
    # The rectangle should contain all circles
    w = max_x - min_x
    h = max_y - min_y
    
    # Scale to perimeter 4: 2(w+h) = 4 => w+h = 2
    current_perim = 2 * (w + h)
    scale = 2.0 / (w + h)
    
    # Actually, we need to be more careful. The rectangle is fixed by the aspect ratio.
    # Let's just return the solution with the rectangle that was used.
    # We'll reconstruct from the best aspect ratio.
    
    # Find which ratio gave the best
    best_ratio = 1.0
    best_s = -1
    for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        if time.time() > deadline - 5.0:
            break
        s, x = solve_for_aspect(ratio)
        if s > best_s:
            best_s = s
            best_ratio = ratio
    
    if best_s <= 0:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = min(x, w - x, y, h - y) - 1e-3
                    cells.append([x, y, max(r, 0.01)])
        return np.array(cells[:N])

    w = 2.0 * best_ratio / (1.0 + best_ratio)
    h = 2.0 / (1.0 + best_ratio)
    
    # Re-solve with best ratio to get clean solution
    s, x = solve_for_aspect(best_ratio)
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Final clamping
    for i in range(N):
        rs[i] = max(0.01, min(rs[i], xs[i] - 1e-6, w - xs[i] - 1e-6, ys[i] - 1e-6, h - ys[i] - 1e-6))
        if rs[i] < 0.01:
            rs[i] = 0.01
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [xs[i], ys[i], rs[i]]
    
    return result

# EVOLVE-BLOCK-END
