import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial guess for aspect ratio w (h = 2 - w)
    # Square is w=1, h=1.
    # Let's try a few initial aspect ratios or optimize w as well.
    
    def objective(params):
        # params: x1, y1, r1, ..., x21, y21, r21, w
        # Extract variables
        coords = params[:3*N]
        w = params[3*N]
        
        # Bounds for w: must be positive and h=2-w must be positive.
        # Also w and h must be large enough to fit the largest circle.
        # Min radius is 0. Max radius is min(w,h)/2.
        # Let's enforce w in [0.5, 1.5] to avoid degenerate rectangles.
        if w < 0.2 or w > 1.8:
            return 1e6
        h = 2.0 - w
        
        # Reshape coords
        x = coords[0::3]
        y = coords[1::3]
        r = coords[2::3]
        
        # Ensure radii are positive
        r = np.maximum(r, 0.0)
        
        # Penalty for boundary violations
        # x_i must be in [r_i, w - r_i]
        # y_i must be in [r_i, h - r_i]
        
        # If r_i > w/2 or r_i > h/2, it's impossible.
        # Penalty for radius being too large for the box
        pen_radius_box = np.sum(np.maximum(0, 2*r - w)**2 + np.maximum(0, 2*r - h)**2)
        
        # Boundary penalties
        pen_x_low = np.sum(np.maximum(0, r - x)**2)
        pen_x_high = np.sum(np.maximum(0, r - (w - x))**2)
        pen_y_low = np.sum(np.maximum(0, r - y)**2)
        pen_y_high = np.sum(np.maximum(0, r - (h - y))**2)
        
        pen_boundary = pen_x_low + pen_x_high + pen_y_low + pen_y_high + pen_radius_box
        
        # Penalty for overlaps
        # dist(i,j) >= r_i + r_j
        # We can vectorize this using broadcasting
        # x, y, r are arrays of size N
        # dx = x[:, None] - x[None, :]
        # dy = y[:, None] - y[None, :]
        # dist = sqrt(dx^2 + dy^2)
        # min_dist = r[:, None] + r[None, :]
        # overlap = max(0, min_dist - dist)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Zero out diagonal to avoid self-overlap penalty
        np.fill_diagonal(dist, np.inf)
        
        overlap = np.maximum(0, min_dist - dist)
        pen_overlap = np.sum(overlap**2)
        
        # Total penalty
        penalty = 100.0 * pen_overlap + 100.0 * pen_boundary
        
        # Objective: maximize sum of radii -> minimize -sum(r) + penalty
        return -np.sum(r) + penalty

    # Bounds for parameters
    # x, y, r are unbounded in the optimizer, but we use penalties.
    # However, providing bounds helps.
    # Let's assume initial w=1, h=1.
    # x, y in [0, 2], r in [0, 1]
    bounds = []
    for i in range(N):
        bounds.append((0, 2.0)) # x
        bounds.append((0, 2.0)) # y
        bounds.append((0, 1.0)) # r
    bounds.append((0.5, 1.5)) # w
    
    # Multiple restarts
    np.random.seed(42)
    max_restarts = 50
    restarts_done = 0
    
    while restarts_done < max_restarts:
        if time.time() - start_time > time_limit:
            break
            
        # Random initial guess
        # Try to start with a reasonable grid-like distribution
        # Or just random within the box
        w_init = np.random.uniform(0.8, 1.2)
        h_init = 2.0 - w_init
        
        x0 = np.random.uniform(0.1, w_init - 0.1, N)
        y0 = np.random.uniform(0.1, h_init - 0.1, N)
        r0 = np.random.uniform(0.05, 0.2, N)
        
        params0 = np.concatenate([
            np.column_stack([x0, y0, r0]).flatten(),
            [w_init]
        ])
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check validity strictly
                params = res.x
                w = params[3*N]
                h = 2.0 - w
                coords = params[:3*N]
                x = coords[0::3]
                y = coords[1::3]
                r = coords[2::3]
                
                # Strict check
                valid = True
                if w <= 0 or h <= 0:
                    valid = False
                else:
                    # Check boundary
                    for i in range(N):
                        if r[i] < 1e-6:
                            continue
                        if x[i] < r[i] - 1e-5 or x[i] > w - r[i] + 1e-5:
                            valid = False
                            break
                        if y[i] < r[i] - 1e-5 or y[i] > h - r[i] + 1e-5:
                            valid = False
                            break
                    
                    if valid:
                        # Check overlap
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                                if dist < r[i] + r[j] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        # Format result: (x, y, r) for each circle
                        # Note: The evaluator expects an array of shape (n, 3).
                        # The rectangle dimensions are not returned, only the circles.
                        # The evaluator will infer the rectangle from the bounding box?
                        # "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
                        # This implies the evaluator calculates the bounding box of the returned circles and checks its perimeter.
                        # OR does it expect the rectangle dimensions to be part of the output?
                        # "array of shape (n, 3)" -> x, y, r.
                        # "bounding rectangle has perimeter at most 4"
                        # If I return circles that fit in a 1x1 square, perimeter is 4.
                        # If I return circles that fit in a 0.5x1.5 rectangle, perimeter is 4.
                        # The evaluator likely computes min_x, max_x, min_y, max_y from the circles.
                        # w_eval = max_x - min_x + 2*max_r? No.
                        # The "bounding rectangle" of the circles is the smallest rectangle containing them.
                        # Its width is max(x_i + r_i) - min(x_i - r_i).
                        # Its height is max(y_i + r_i) - min(y_i - r_i).
                        # Perimeter = 2 * (W + H).
                        # We need this to be <= 4.
                        
                        # My optimization assumed a fixed container w, h with w+h=2.
                        # The circles are inside [0, w] x [0, h].
                        # The actual bounding box of the circles might be smaller than w, h.
                        # If the bounding box is smaller, the perimeter is smaller, which is valid.
                        # However, if I pack them tightly in a w x h box, the bounding box will be close to w x h.
                        # So minimizing the objective for a fixed w+h=2 should yield a valid bounding box with perimeter <= 4 (actually <= 4).
                        
                        # Wait, if the circles don't touch the boundaries, the bounding box is smaller.
                        # That's fine.
                        
                        # Let's double check the "bounding rectangle" definition.
                        # Usually, it means the tightest axis-aligned bounding box.
                        # So W = max(x+r) - min(x-r).
                        # H = max(y+r) - min(y-r).
                        # Perimeter = 2(W+H).
                        # We need 2(W+H) <= 4 => W+H <= 2.
                        
                        # My optimization forces circles to be inside [0,w]x[0,h] with w+h=2.
                        # Thus W <= w and H <= h.
                        # So W+H <= w+h = 2.
                        # So the perimeter constraint is satisfied.
                        
                        best_result = np.column_stack([x, y, r])
                        
        except Exception as e:
            pass
            
        restarts_done += 1
        
    if best_result is None:
        # Fallback to seed program logic
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
