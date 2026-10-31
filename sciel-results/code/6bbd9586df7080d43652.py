"""
Circle packing in perimeter-4 rectangle: Variable aspect ratio optimization with analytic gradients.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # Safety margin before 150s kill

def _objective(x, mu=1e4):
    """
    Objective function to minimize: -sum(r) + mu * penalties.
    x = [W, x1, y1, r1, ..., xN, yN, rN]
    W is the width. H = 2 - W.
    """
    W = x[0]
    H = 2.0 - W
    
    # If W or H are non-positive, return a large penalty
    if W <= 0 or H <= 0:
        return 1e10, np.zeros_like(x)
        
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[1::3] = 0.0 # dx
    grad[2::3] = 0.0 # dy
    grad[3::3] = -1.0 # dr
    
    # Penalty for W, H bounds
    # We want W in (0, 2). Let's enforce W > 0.1 and H > 0.1 to avoid degenerate rectangles
    if W < 0.1:
        val += mu * (0.1 - W)**2
        grad[0] += 2 * mu * (0.1 - W) * (-1)
    if H < 0.1:
        val += mu * (0.1 - H)**2
        grad[0] += 2 * mu * (0.1 - H) * (-1) # dH/dW = -1
    
    # Boundary constraints: circles must stay inside [0, W] x [0, H]
    # r_i <= min(x_i, y_i, W-x_i, H-y_i)
    # We use soft penalties for violations
    for i in range(N):
        xi, yi, ri = cx[i], cy[i], r[i]
        
        # Distances to boundaries
        d_left = xi
        d_right = W - xi
        d_bottom = yi
        d_top = H - yi
        
        min_dist = min(d_left, d_right, d_bottom, d_top)
        
        # Violation if ri > min_dist
        viol = ri - min_dist
        if viol > 0:
            val += mu * viol**2
            dviol_dri = 1.0
            dviol_dxi = 0.0
            dviol_dyi = 0.0
            
            # Gradient of min_dist
            # If multiple boundaries are equidistant, average the gradients
            active = []
            if d_left == min_dist: active.append(('left', 1.0, 0.0, 1.0)) # d(d_left)/dx = 1
            if d_right == min_dist: active.append(('right', -1.0, 0.0, -1.0)) # d(d_right)/dx = -1, d(d_right)/dW = -1
            if d_bottom == min_dist: active.append(('bottom', 0.0, 1.0, 1.0)) # d(d_bottom)/dy = 1
            if d_top == min_dist: active.append(('top', 0.0, -1.0, -1.0)) # d(d_top)/dy = -1, d(d_top)/dW = -1
            
            if active:
                # Average the gradients of the active boundaries
                avg_dxi = sum(a[1] for a in active) / len(active)
                avg_dyi = sum(a[2] for a in active) / len(active)
                avg_dW = sum(a[3] for a in active) / len(active)
                
                dviol_dxi = -avg_dxi
                dviol_dyi = -avg_dyi
                dviol_dW = -avg_dW
            else:
                # Should not happen if min_dist is calculated correctly
                pass
            
            # Update gradients
            grad[3 + 3*i] += 2 * mu * viol * dviol_dri
            grad[1 + 3*i] += 2 * mu * viol * dviol_dxi
            grad[2 + 3*i] += 2 * mu * viol * dviol_dyi
            grad[0] += 2 * mu * viol * dviol_dW

    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx*dx + dy*dy)
            req = r[i] + r[j]
            
            viol = req - d
            if viol > 0:
                val += mu * viol**2
                if d < 1e-6:
                    # Perturb slightly to avoid division by zero
                    dx = 1e-6
                    d = 1e-6
                
                dviol_dxi = -dx/d
                dviol_dyi = -dy/d
                dviol_dri = 1.0
                
                dviol_dxj = dx/d
                dviol_dyj = dy/d
                dviol_drj = 1.0
                
                grad[1 + 3*i] += 2 * mu * viol * dviol_dxi
                grad[2 + 3*i] += 2 * mu * viol * dviol_dyi
                grad[3 + 3*i] += 2 * mu * viol * dviol_dri
                
                grad[1 + 3*j] += 2 * mu * viol * dviol_dxj
                grad[2 + 3*j] += 2 * mu * viol * dviol_dyj
                grad[3 + 3*j] += 2 * mu * viol * dviol_drj
                
    return val, grad

def _evaluate(x):
    """
    Calculate the valid sum of radii for a given solution x.
    Returns 0 if invalid.
    """
    W = x[0]
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0
        
    cx = x[1::3]
    cy = x[2::3]
    r = x[3::3]
    
    # Clamp positions and radii to be valid
    valid = True
    for i in range(N):
        if cx[i] < 0 or cx[i] > W or cy[i] < 0 or cy[i] > H:
            # Adjust position
            cx[i] = np.clip(cx[i], 0, W)
            cy[i] = np.clip(cy[i], 0, H)
            
        # Adjust radius to fit boundary
        max_r = min(cx[i], cy[i], W - cx[i], H - cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
        if r[i] < 1e-6:
            return 0.0
            
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if d < r[i] + r[j] - 1e-8:
                return 0.0
                
    return np.sum(r)

def _grid_init(W, H):
    """Initialize with a square grid."""
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.1*W, 0.9*W, side)
    ys = np.linspace(0.1*H, 0.9*H, side)
    
    cx = []
    cy = []
    r = []
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N: break
            cx.append(xs[i])
            cy.append(ys[j])
            # Estimate radius based on spacing
            dx = W / side
            dy = H / side
            ri = 0.4 * min(dx, dy)
            # Ensure it fits in boundary
            ri = min(ri, cx[-1], cy[-1], W - cx[-1], H - cy[-1])
            r.append(ri)
            count += 1
        if count >= N: break
            
    return np.array([W] + cx + cy + r).flatten()

def _hex_init(W, H):
    """Initialize with a hexagonal grid."""
    # Approximate radius for hex packing
    area = W * H
    r_guess = np.sqrt(area * np.sqrt(3) / (N * 2 * np.pi)) * 0.9
    
    dx = 2 * r_guess
    dy = dx * np.sqrt(3) / 2
    
    cx = []
    cy = []
    r = []
    
    row = 0
    while len(cx) < N:
        offset = dx/2 if row % 2 else 0
        col = 0
        while len(cx) < N:
            x = offset + col * dx
            y = row * dy
            if 0 <= x <= W and 0 <= y <= H:
                cx.append(x)
                cy.append(y)
                ri = min(x, y, W-x, H-y, r_guess)
                r.append(ri)
            col += 1
        row += 1
        
    return np.array([W] + cx[:N] + cy[:N] + r[:N]).flatten()

def _random_init():
    """Initialize with random positions and small radii."""
    W = random.uniform(0.8, 1.2)
    H = 2.0 - W
    
    cx = []
    cy = []
    r = []
    
    for _ in range(N):
        x = random.uniform(0.1*W, 0.9*W)
        y = random.uniform(0.1*H, 0.9*H)
        # Small initial radius
        ri = random.uniform(0.05, 0.1)
        ri = min(ri, x, y, W-x, H-y)
        cx.append(x)
        cy.append(y)
        r.append(ri)
        
    return np.array([W] + cx + cy + r).flatten()

def _optimize_from(x0, max_iter=200):
    """Run SLSQP optimization from x0."""
    if time.time() > DEADLINE - 10:
        return x0, _evaluate(x0)
        
    best_x = x0.copy()
    best_sum = _evaluate(x0)
    
    def fun(x):
        val, _ = _objective(x)
        return val
        
    def grad(x):
        _, g = _objective(x)
        return g
        
    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP', 
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x)
            if s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass
        
    # Second pass with tighter tolerance if time permits
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 100, 'ftol': 1e-10, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x)
                if s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass
            
    return best_x, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    
    # Multi-start
    starts = []
    
    # 1. Grid inits with different aspect ratios
    for W in [0.8, 0.9, 1.0, 1.1, 1.2]:
        H = 2.0 - W
        x0 = _grid_init(W, H)
        starts.append(x0)
        
    # 2. Hex inits
    for W in [0.9, 1.0, 1.1]:
        H = 2.0 - W
        x0 = _hex_init(W, H)
        starts.append(x0)
        
    # 3. Random inits
    for _ in range(5):
        x0 = _random_init()
        starts.append(x0)
        
    # Add some perturbed versions of the best initial guesses
    # We'll just run them all for now
    
    for x0 in starts:
        if time.time() > DEADLINE - 15:
            break
            
        # Add small noise to break symmetry
        noise = np.random.normal(0, 0.01, x0.shape)
        noise[0] = 0 # Don't perturb W too much initially
        x0_pert = x0 + noise
        
        res_x, res_sum = _optimize_from(x0_pert)
        
        if res_sum > best_sum:
            best_sum = res_sum
            best_arr = res_x.copy()
            
    # If we found a good solution, try to refine the best one further
    if best_arr is not None and time.time() < DEADLINE - 20:
        res_x, res_sum = _optimize_from(best_arr, max_iter=300)
        if res_sum > best_sum:
            best_sum = res_sum
            best_arr = res_x.copy()
            
    if best_arr is None:
        # Fallback: simple grid in 1x1 square
        W = 1.0
        H = 1.0
        side = 5
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N: break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                pts.append([x, y, r])
        return np.array(pts)
        
    # Format output
    W = best_arr[0]
    H = 2.0 - W
    cx = best_arr[1::3]
    cy = best_arr[2::3]
    r = best_arr[3::3]
    
    # Final validation and clamping
    for i in range(N):
        cx[i] = np.clip(cx[i], 0, W)
        cy[i] = np.clip(cy[i], 0, H)
        max_r = min(cx[i], cy[i], W-cx[i], H-cy[i])
        r[i] = min(r[i], max_r)
        
    result = np.column_stack([cx, cy, r])
    return result

