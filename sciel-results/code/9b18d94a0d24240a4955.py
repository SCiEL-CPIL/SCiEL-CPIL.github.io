"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 110.0  # safety margin before 120s limit


def _hexagonal_init(n, w, h, rng):
    """Generate a hexagonal lattice initialization inside rectangle [0,w]x[0,h]."""
    # Try different spacings to fill the rectangle
    # Hexagonal packing: row spacing = r*sqrt(3), col spacing = 2*r
    # We want n circles in w x h
    
    # Estimate radius from area
    # For hexagonal packing, density ~ pi/(2*sqrt(3)) ≈ 0.9069
    # n * pi * r^2 ≈ 0.9069 * w * h
    r_est = np.sqrt(0.9069 * w * h / (n * np.pi))
    
    # Try a grid of possible spacings
    best_positions = None
    best_score = -1
    
    for scale in np.linspace(0.7, 1.5, 15):
        r_try = r_est * scale
        # Hexagonal lattice
        dx = 2.0 * r_try
        dy = np.sqrt(3) * r_try
        
        positions = []
        row = 0
        y = r_try + rng.uniform(0, dy * 0.1)
        while y < h - r_try * 0.5:
            x_offset = 0.0 if row % 2 == 0 else r_try
            x = r_try + x_offset + rng.uniform(0, dx * 0.1)
            col = 0
            while x < w - r_try * 0.5:
                positions.append([x, y])
                x += dx
                col += 1
            y += dy
            row += 1
            if len(positions) >= n:
                break
            if len(positions) >= n * 2:
                break
        
        if len(positions) < n:
            # Add random points
            while len(positions) < n:
                positions.append([rng.uniform(r_try, w - r_try), 
                                  rng.uniform(r_try, h - r_try)])
        
        positions = np.array(positions[:n])
        
        # Score: sum of radii (use r_try as initial radius)
        # Check feasibility
        valid = True
        for i in range(n):
            if positions[i, 0] < 0 or positions[i, 0] > w:
                valid = False
            if positions[i, 1] < 0 or positions[i, 1] > h:
                valid = False
        
        if not valid:
            continue
        
        # Compute actual min radius for this configuration
        min_r = min(w, h) * 0.1
        for i in range(n):
            min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((positions[i] - positions[j])**2))
                min_r = min(min_r, d / 2.0)
        
        score = min_r * n
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
            best_r = min_r
    
    if best_positions is None:
        # Fallback: uniform grid
        side = int(np.ceil(np.sqrt(n)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < n:
                    positions.append([(i+0.5)*w/side, (j+0.5)*h/side])
        best_positions = np.array(positions[:n])
        best_r = min(w, h) / (2 * side)
    
    return best_positions, best_r


def _solve_packing(w, h, n, rng, time_limit):
    """Solve circle packing in rectangle w x h using SQP."""
    # State: [x1,y1,r1, x2,y2,r2, ..., xn,yn,rn]
    # Objective: maximize sum(ri)
    # Constraints:
    #   ri >= 0
    #   xi - ri >= 0, xi + ri <= w
    #   yi - ri >= 0, yi + ri <= h
    #   (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)
    
    def objective_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g
    
    constraints = []
    
    # Wall constraints
    for i in range(n):
        xi, yi, ri = 3*i, 3*i+1, 3*i+2
        # xi - ri >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2],
            'jac': lambda x, i=i: np.array([1.0, 0.0, -1.0] if False else 
                np.concatenate([np.zeros(3*i), [1.0, 0.0, -1.0], np.zeros(3*(n-i-1))]))
        })
        # xi + ri <= w  =>  w - xi - ri >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: w - x[3*i] - x[3*i+2],
            'jac': lambda x, i=i: np.concatenate([np.zeros(3*i), [-1.0, 0.0, -1.0], np.zeros(3*(n-i-1))])
        })
        # yi - ri >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2],
            'jac': lambda x, i=i: np.concatenate([np.zeros(3*i), [0.0, 1.0, -1.0], np.zeros(3*(n-i-1))])
        })
        # yi + ri <= h  =>  h - yi - ri >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: h - x[3*i+1] - x[3*i+2],
            'jac': lambda x, i=i: np.concatenate([np.zeros(3*i), [0.0, -1.0, -1.0], np.zeros(3*(n-i-1))])
        })
    
    # Pairwise constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            
            def pair_fun(x, i=i, j=j):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                return dx*dx + dy*dy - (x[ri] + x[rj])**2
            
            def pair_jac(x, i=i, j=j):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                grad = np.zeros(3*n)
                grad[xi] = 2*dx
                grad[xj] = -2*dx
                grad[yi] = 2*dy
                grad[yj] = -2*dy
                grad[ri] = -2*(x[ri] + x[rj])
                grad[rj] = -2*(x[ri] + x[rj])
                return grad
            
            constraints.append({
                'type': 'ineq',
                'fun': pair_fun,
                'jac': pair_jac
            })
    
    # Initial guess
    pos, r0 = _hexagonal_init(n, w, h, rng)
    x0 = np.zeros(3*n)
    x0[0::3] = pos[:, 0]
    x0[1::3] = pos[:, 1]
    x0[2::3] = r0 * np.ones(n)
    
    bounds = [(0, w)] * n + [(0, h)] * n + [(0, min(w,h))] * n
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-8})
        if res.success or res.fun < objective(x0):
            return res.x, -res.fun
        else:
            return x0, -objective(x0)
    except Exception:
        return x0, -objective(x0)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_sum = 0
    best_result = None
    
    # Try several aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # w in (0, 2), h = 2 - w
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.8, 0.56, 2.0, 0.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE:
            break
        
        # w + h = 2, w/h = ar => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() - start_time > DEADLINE:
                break
            
            x, total_r = _solve_packing(w, h, N, rng, DEADLINE - (time.time() - start_time))
            
            if total_r > best_sum:
                best_sum = total_r
                # Extract circles: (cx, cy, r)
                circles = np.zeros((N, 3))
                circles[:, 0] = x[0::3]
                circles[:, 1] = x[1::3]
                circles[:, 2] = x[2::3]
                best_result = circles
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    r = 0.5 / side - 1e-4
                    cells.append([(i+0.5)*w/side, (j+0.5)*h/side, r])
        best_result = np.array(cells[:N])
    
    # Final validity check
    w_rect = max(best_result[:, 0] + best_result[:, 2])
    h_rect = max(best_result[:, 1] + best_result[:, 2])
    # Actually need to check all circles fit
    for i in range(len(best_result)):
        cx, cy, r = best_result[i]
        if cx - r < -1e-9 or cx + r > w_rect + 1e-9 or cy - r < -1e-9 or cy + r > h_rect + 1e-9:
            # Clip
            best_result[i, 2] = min(cx, w_rect - cx, cy, h_rect - cy)
    
    # Check pairwise
    for i in range(len(best_result)):
        for j in range(i+1, len(best_result)):
            d = np.sqrt(np.sum((best_result[i,:2] - best_result[j,:2])**2))
            min_r = d / 2.0 - 1e-6
            if best_result[i, 2] > min_r:
                best_result[i, 2] = min_r
            if best_result[j, 2] > min_r:
                best_result[j, 2] = min_r
    
    return best_result
