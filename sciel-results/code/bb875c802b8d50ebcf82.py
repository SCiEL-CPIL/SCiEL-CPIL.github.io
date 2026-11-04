"""Combined program: Multi-start penalty optimization with smooth gradients."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds [0, w] x [0, h].
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
        
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _validate_global(circles):
    """
    Validate circles in global coordinates. 
    The evaluator checks: 
    1. n == 21
    2. rs > 0
    3. Perimeter of bounding box <= 4
    4. No overlaps
    """
    n = len(circles)
    if n != N:
        return False
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
        
    # Perimeter check
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    bottom = np.min(ys - rs)
    top = np.max(ys + rs)
    width = right - left
    height = top - bottom
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + 1e-9:
        return False
        
    # Disjoint check
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True

def _score(circles):
    if not _validate_global(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _resolve_collisions(circles, w, h, max_iters=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random perturbation if exactly overlapping
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _generate_hex_init(w, h, n_circles):
    """
    Generate a hexagonal cluster initialization scaled to fit w x h.
    Structure: 5 rows with counts [4, 5, 4, 5, 3] (Total 21)
    """
    rows = [4, 5, 4, 5, 3]
    centers = []
    h_unit = np.sqrt(3) / 2
    y_coords = [0, h_unit, 2*h_unit, 3*h_unit, 4*h_unit]
    
    for i, count in enumerate(rows):
        y = y_coords[i]
        center_x = 1.5  # Center of the widest row (width 4)
        start_x = center_x - (count - 1) / 2.0
        for k in range(count):
            x = start_x + k
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Normalize to [0, 1] x [0, 1]
    min_x, max_x = np.min(centers[:, 0]), np.max(centers[:, 0])
    min_y, max_y = np.min(centers[:, 1]), np.max(centers[:, 1])
    w_c = max_x - min_x
    h_c = max_y - min_y
    
    # Scale to fit inside [r, w-r] x [r, h-r] approximately
    # We don't know r yet, so assume a small r initially or just fit in [0, w]x[0,h] with padding
    pad = 0.1 # 10% padding
    scale_x = (1 - 2*pad) / w_c
    scale_y = (1 - 2*pad) / h_c
    
    # Maintain aspect ratio of the cluster if possible, or stretch
    # Stretching might distort contacts, but relaxation will fix positions.
    # Let's scale uniformly to fit, then center.
    scale = min(scale_x, scale_y)
    
    scaled_centers = (centers - [min_x, min_y]) * scale
    # Center in box
    offset_x = (w - scale * w_c) / 2
    offset_y = (h - scale * h_c) / 2
    scaled_centers += [offset_x, offset_y]
    
    # Estimate initial radius
    # Min distance in scaled space
    min_dist = np.inf
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            d = np.linalg.norm(scaled_centers[i] - scaled_centers[j])
            if d < min_dist:
                min_dist = d
                
    r_init = min_dist * 0.49 # Slightly less than touching
    
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = scaled_centers[:, 0]
    circles[:, 1] = scaled_centers[:, 1]
    circles[:, 2] = r_init
    
    return circles

def _smooth_penalty_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using L-BFGS-B with smooth penalty method.
    Uses smooth log-sum-exp approximations for max(0, violation) terms.
    """
    if time.time() > deadline - 2.0:
        return None

    n = len(circles)
    x0 = circles.flatten()
    
    # Pre-compute indices for vectorization
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Smooth parameters
    # beta is the penalty weight
    # eps controls the smoothness of the max(0, x) approximation
    # We will ramp up beta and decrease eps
    
    def objective(x, beta, eps):
        if time.time() > deadline:
            return 1e10, None
        
        pos = x.reshape(-1, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0 # d(-sum(r))/dr = -1
        
        # 1. Boundary Penalty
        # Constraints: x >= r, w - x >= r, y >= r, h - y >= r
        # Violation v = max(0, r - x) etc.
        # Smooth approx: beta * eps * log(1 + exp(-v/eps))
        
        # Helper for smooth max(0, -val) where val is the slack (constraint >= 0)
        # We want to penalize when slack < 0.
        # Let slack = x - r. We want penalty for slack < 0.
        # Smooth max(0, -slack) = eps * log(1 + exp(-slack/eps))
        
        # Boundary X
        slacks_x_left = xs - rs
        slacks_x_right = w - xs - rs
        slacks_y_bottom = ys - rs
        slacks_y_top = h - ys - rs
        
        # Vectorized smooth penalty calculation
        # P = beta * eps * log(1 + exp(-slack/eps))
        # dP/dslack = -beta * sigmoid(-slack/eps)
        
        def smooth_penalty_and_grad(slacks, beta, eps):
            # slacks: array
            # Returns penalty sum and gradient wrt slacks
            # For numerical stability, use logaddexp(0, -t) where t = slack/eps
            t = slacks / eps
            # log(1 + exp(-t)) = logaddexp(0, -t)
            log_term = np.logaddexp(0, -t)
            pen = beta * eps * log_term
            # d(log(1+exp(-t)))/dt = -sigmoid(-t) = -1/(1+exp(t))
            # d(t)/dslack = 1/eps
            # dP/dslack = beta * eps * (-1/(1+exp(t))) * (1/eps) = -beta / (1+exp(t))
            # Note: sigmoid(-t) = 1/(1+exp(t))
            grad_slack = -beta / (1.0 + np.exp(t))
            return np.sum(pen), grad_slack
        
        # Apply to boundaries
        # slack = x - r => dslack/dx = 1, dslack/dr = -1
        # slack = w - x - r => dslack/dx = -1, dslack/dr = -1
        
        pen_xl, g_xl = smooth_penalty_and_grad(slacks_x_left, beta, eps)
        pen_xr, g_xr = smooth_penalty_and_grad(slacks_x_right, beta, eps)
        pen_yb, g_yb = smooth_penalty_and_grad(slacks_y_bottom, beta, eps)
        pen_yt, g_yt = smooth_penalty_and_grad(slacks_y_top, beta, eps)
        
        obj += pen_xl + pen_xr + pen_yb + pen_yt
        
        # Gradients
        # g_xl: dP/d(x-r) => dP/dx += g_xl, dP/dr += -g_xl
        # g_xr: dP/d(w-x-r) => dP/dx += -g_xr, dP/dr += -g_xr
        grad[:n] += g_xl - g_xr
        grad[n:2*n] += g_yb - g_yt
        grad[2*n:] += -g_xl - g_xr - g_yb - g_yt
        
        # 2. Overlap Penalty
        # Constraint: dist >= r_i + r_j
        # Slack = dist - r_i - r_j
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dists = np.sqrt(dx**2 + dy**2 + 1e-12) # Avoid division by zero
        min_dists = rs[i_idx] + rs[j_idx]
        slacks_overlap = dists - min_dists
        
        pen_ov, g_ov = smooth_penalty_and_grad(slacks_overlap, beta, eps)
        obj += pen_ov
        
        # Gradients for overlap
        # slack = dist - r_i - r_j
        # dslack/dx_i = (x_i - x_j)/dist = -nx
        # dslack/dx_j = (x_j - x_i)/dist = nx
        # dslack/dr_i = -1
        # dslack/dr_j = -1
        
        # g_ov is dP/dslack for each pair
        # We need to accumulate gradients into the main vector
        
        # nx, ny are unit vectors from j to i?
        # dx = x_i - x_j. dist = |dx|.
        # d(dist)/d(x_i) = (x_i - x_j)/dist = dx/dist
        # d(dist)/d(x_j) = (x_j - x_i)/dist = -dx/dist
        
        nx = dx / dists
        ny = dy / dists
        
        # dP/dx_i = g_ov * (dx/dist) = g_ov * nx
        # dP/dx_j = g_ov * (-dx/dist) = -g_ov * nx
        
        # Use np.add.at for accumulation
        # Indices for i: i_idx
        # Indices for j: j_idx
        
        # x components
        np.add.at(grad, i_idx, g_ov * nx)
        np.add.at(grad, j_idx, -g_ov * nx)
        
        # y components
        np.add.at(grad, n + i_idx, g_ov * ny)
        np.add.at(grad, n + j_idx, -g_ov * ny)
        
        # r components
        # dP/dr_i = g_ov * (-1) = -g_ov
        # dP/dr_j = g_ov * (-1) = -g_ov
        np.add.at(grad, 2*n + i_idx, -g_ov)
        np.add.at(grad, 2*n + j_idx, -g_ov)
        
        return obj, grad

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, w - 1e-4)) # x
        bounds.append((1e-4, h - 1e-4)) # y
        bounds.append((1e-4, min(w, h)/2)) # r

    # Annealing schedule
    # Start with small beta, large eps (smooth)
    # End with large beta, small eps (sharp)
    n_steps = 5
    betas = np.geomspace(10.0, 1000.0, n_steps)
    epsilons = np.geomspace(0.1, 0.001, n_steps)
    
    x = x0.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        beta = betas[step]
        eps = epsilons[step]
        
        try:
            res = minimize(
                objective,
                x,
                args=(beta, eps),
                method='L-BFGS-B',
                jac=True, # We return (f, g)
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-9, 'maxfun': 500, 'disp': False}
            )
            x = res.x
        except Exception:
            continue
            
    new_circles = x.reshape(-1, 3)
    
    # Post-processing: Ensure validity
    # The penalty method might leave tiny overlaps or boundary violations.
    # We run a few iterations of relaxation to fix any residual issues.
    
    # First, clamp to bounds strictly
    new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], w - new_circles[:, 2])
    new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], h - new_circles[:, 2])
    
    # Check validity
    if _validate_solution(new_circles, w, h, N, tol=1e-7):
        return new_circles
        
    # Try relaxation
    fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
    if _validate_solution(fixed, w, h, N, tol=1e-7):
        return fixed
        
    # If still invalid, shrink radii
    scale = 0.95
    shrunk = new_circles.copy()
    shrunk[:, 2] *= scale
    if _validate_solution(shrunk, w, h, N, tol=1e-7):
        return shrunk
        
    return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio (box size w x h).
    """
    # 1. Generate hex init
    init_circles = _generate_hex_init(w, h, N)
    
    # 2. Relax to ensure validity
    relaxed = _resolve_collisions(init_circles, w, h, max_iters=500)
    
    if not _validate_solution(relaxed, w, h, N):
        # Fallback grid
        cols, rows = 5, 5
        grid = np.zeros((N, 3))
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N: break
                grid[count, 0] = (i + 0.5) * w / cols
                grid[count, 1] = (j + 0.5) * h / rows
                grid[count, 2] = 0.45 * min(w/cols, h/rows)
                count += 1
        relaxed = _resolve_collisions(grid, w, h, max_iters=500)
        if not _validate_solution(relaxed, w, h, N):
            return None

    # 3. Smooth Penalty Optimize
    optimized = _smooth_penalty_optimize(relaxed, w, h, deadline)
    
    if optimized is not None:
        return optimized
    else:
        if _validate_solution(relaxed, w, h, N, tol=1e-7):
            return relaxed
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin for 120s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try different aspect ratios
    # Perimeter 2(W+H)=4 => W+H=2
    # W = 2 * ar / (1 + ar), H = 2 / (1 + ar)
    aspect_ratios = [0.9, 1.0, 1.1]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        try:
            result = _solve_aspect(W, H, deadline)
            
            if result is not None:
                if _validate_global(result):
                    score = _score(result)
                    if score > best_score:
                        best_score = score
                        best_circles = result.copy()
                    
        except Exception:
            continue
            
    # Fallback
    if best_circles is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-5
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    if not _validate_global(best_circles):
        r = 0.01
        best_circles = np.zeros((N, 3))
        for i in range(N):
            best_circles[i, 0] = r + i * (2 * r + 1e-6)
            best_circles[i, 1] = r
            best_circles[i, 2] = r
            
    return best_circles

# EVOLVE-BLOCK-END
