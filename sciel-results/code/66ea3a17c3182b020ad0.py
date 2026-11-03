import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Initial guess: Grid layout
    # Aspect ratio 1:1 (Square) is a safe start, but we optimize w
    # Perimeter 4 -> w + h = 2 -> h = 2 - w
    # Let's start with w=1, h=1
    
    # Create a grid of N points
    # For N=21, a 5x5 grid has 25 spots. We take 21.
    # Or a 6x4 grid (24 spots).
    # Let's use a 5x5 grid and drop the last 4 corners to make it more compact? 
    # Or just center of mass.
    
    # Better initialization: Hexagonal packing approximation
    # But grid is easier to code robustly.
    side = int(np.ceil(np.sqrt(N)))
    # 5x5 grid
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            # Center of cell
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            xs.append(x)
            ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Initial radii
    # Max radius in a unit square grid cell is 0.5/side
    r0 = 0.5 / side
    rs = np.full(N, r0)
    
    # Variable vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # w is the width, h = 2 - w
    w0 = 1.0
    
    x0 = np.concatenate([xs, ys, rs, [w0]])
    
    def objective(v):
        xs_v = v[0:N]
        ys_v = v[N:2*N]
        rs_v = v[2*N:3*N]
        w_v = v[3*N]
        
        # Ensure w is valid (0 < w < 2)
        # Clip w to [0.1, 1.9] to avoid degenerate rectangles
        w_v = np.clip(w_v, 0.1, 1.9)
        h_v = 2.0 - w_v
        
        # Calculate overlaps
        # Vectorized distance calculation
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # We need sum of max(0, ri+rj - dij)^2
        
        # For N=21, O(N^2) is small.
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs_v[i] - xs_v[j]
                dy = ys_v[i] - ys_v[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs_v[i] + rs_v[j]
                if dist < min_dist:
                    diff = min_dist - dist
                    overlap_penalty += diff * diff
        
        # Boundary penalty
        bound_penalty = 0.0
        for i in range(N):
            # x bounds: [ri, w-ri]
            if xs_v[i] < rs_v[i]:
                bound_penalty += (rs_v[i] - xs_v[i])**2
            if xs_v[i] > w_v - rs_v[i]:
                bound_penalty += (xs_v[i] - (w_v - rs_v[i]))**2
                
            # y bounds: [ri, h-ri]
            if ys_v[i] < rs_v[i]:
                bound_penalty += (rs_v[i] - ys_v[i])**2
            if ys_v[i] > h_v - rs_v[i]:
                bound_penalty += (ys_v[i] - (h_v - rs_v[i]))**2
        
        # Total penalty
        penalty = overlap_penalty + bound_penalty
        
        # We want to maximize sum(r). So minimize -sum(r) + lambda * penalty
        # Lambda needs to be large enough to enforce constraints.
        # Since radii are ~0.1-0.2, penalties can be small. 
        # Let's use a large lambda.
        lam = 100.0
        
        return -np.sum(rs_v) + lam * penalty

    # Optimization
    # L-BFGS-B is good for box constraints, but we have non-linear constraints.
    # Nelder-Mead is derivative-free and robust.
    # Let's try Nelder-Mead.
    
    # Initial simplex size
    # We can just let scipy handle it, or provide options.
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    # Extract solution
    xs_res = result.x[0:N]
    ys_res = result.x[N:2*N]
    rs_res = result.x[2*N:3*N]
    w_res = np.clip(result.x[3*N], 0.1, 1.9)
    h_res = 2.0 - w_res
    
    # Post-processing: Ensure no overlaps and within bounds
    # If there are still tiny overlaps, we can shrink radii slightly or push circles.
    # Given the penalty, they should be very close to valid.
    # To be safe, we can run a few iterations of simple relaxation if invalid.
    
    # Check validity
    def is_valid(xs, ys, rs, w, h):
        for i in range(N):
            if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                return False
            if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - 1e-6:
                    return False
        return True
    
    if not is_valid(xs_res, ys_res, rs_res, w_res, h_res):
        # Fallback to seed-like solution if optimization failed badly
        # But usually Nelder-Mead works well.
        # We can try to repair: shrink all radii by 1%
        rs_res *= 0.99
        if not is_valid(xs_res, ys_res, rs_res, w_res, h_res):
             # Return a safe grid
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    # Format output
    # Array of shape (N, 3): [x, y, r]
    out = np.zeros((N, 3))
    out[:, 0] = xs_res
    out[:, 1] = ys_res
    out[:, 2] = rs_res
    
    return out
