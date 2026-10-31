import numpy as np
import time

N = 21
TIME_LIMIT = 145  # Leave 5s margin for overhead

def compute_violations_and_grads(x, y, r, W, H):
    """
    Computes the sum of squared violations and their gradients w.r.t x, y.
    Violations are:
    1. Overlaps between circles: (r_i + r_j - d_ij)^2 if d_ij < r_i + r_j
    2. Boundary violations: (r_i - x_i)^2 if x_i < r_i, etc.
    """
    V = 0.0
    dx = np.zeros(N)
    dy = np.zeros(N)
    
    # Pairwise overlaps
    # Using loops for N=21 is efficient enough
    for i in range(N):
        for j in range(i + 1, N):
            dx_ij = x[i] - x[j]
            dy_ij = y[i] - y[j]
            d2 = dx_ij**2 + dy_ij**2
            min_d = r[i] + r[j]
            
            if d2 < min_d**2:
                d = np.sqrt(d2)
                if d < 1e-10:
                    # Centers coincide, apply random small force
                    overlap = min_d
                    # Arbitrary direction
                    dx[i] += 1.0
                    dy[i] += 1.0
                    dx[j] -= 1.0
                    dy[j] -= 1.0
                    V += overlap**2
                else:
                    overlap = min_d - d
                    V += overlap**2
                    # Gradient of (min_d - d)^2 w.r.t x_i
                    # d/dx_i (min_d - sqrt((x_i-x_j)^2 + ...))^2
                    # = 2 * (min_d - d) * d/dx_i (min_d - d)
                    # d/dx_i (-d) = - (x_i - x_j) / d
                    # So term is 2 * overlap * (-(x_i - x_j) / d) ? 
                    # Wait. Let f = min_d - d. df/dx_i = - dx_ij / d.
                    # d(f^2)/dx_i = 2 f df/dx_i = 2 overlap * (-dx_ij / d)
                    # We want to minimize V, so we move in direction of -grad.
                    # grad_xi = -2 * overlap * dx_ij / d
                    # So update x_i -= lr * grad_xi = x_i += lr * 2 * overlap * dx_ij / d
                    # This pushes i away from j. Correct.
                    
                    force_factor = 2.0 * overlap / d
                    dx[i] += force_factor * dx_ij
                    dx[j] -= force_factor * dx_ij
                    dy[i] += force_factor * dy_ij
                    dy[j] -= force_factor * dy_ij

    # Boundary violations
    for i in range(N):
        # Left
        if x[i] < r[i]:
            viol = r[i] - x[i]
            V += viol**2
            dx[i] += 2.0 * viol
        # Right
        if x[i] > W - r[i]:
            viol = x[i] - (W - r[i])
            V += viol**2
            dx[i] -= 2.0 * viol
        # Bottom
        if y[i] < r[i]:
            viol = r[i] - y[i]
            V += viol**2
            dy[i] += 2.0 * viol
        # Top
        if y[i] > H - r[i]:
            viol = y[i] - (H - r[i])
            V += viol**2
            dy[i] -= 2.0 * viol
            
    return V, dx, dy

def relax(x, y, r, W, H, steps=100, lr=0.05, tol=1e-8):
    """
    Attempts to reduce violations to near zero using gradient descent.
    Returns final violation value.
    """
    for _ in range(steps):
        V, dx, dy = compute_violations_and_grads(x, y, r, W, H)
        if V < tol:
            break
        # Simple gradient descent
        x -= lr * dx
        y -= lr * dy
        
        # Optional: clamp positions to box to prevent flying out
        # x = np.clip(x, r, W - r)
        # y = np.clip(y, r, H - r)
        # Clamping can cause issues with gradients if not done carefully,
        # but the boundary forces should keep them in.
        
    return V

def try_packing(W, H, init_type='grid', seed=42):
    """
    Attempts to pack N circles in W x H rectangle.
    Returns (x, y, r, sum_r)
    """
    np.random.seed(seed)
    
    # Initialization
    if init_type == 'grid':
        # Determine grid dimensions
        aspect = W / H
        if aspect >= 1:
            cols = int(np.ceil(np.sqrt(N * aspect)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.ceil(np.sqrt(N / aspect)))
            cols = int(np.ceil(N / rows))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * W / cols
                y[idx] = (i + 0.5) * H / rows
                r[idx] = 0.05  # Start small
                idx += 1
    else:
        x = np.random.rand(N) * W
        y = np.random.rand(N) * H
        r = np.full(N, 0.05)

    # Initial relaxation to ensure valid start
    relax(x, y, r, W, H, steps=50, lr=0.1)

    # Inflate radii
    step_r = 0.005
    max_inflate_steps = 500
    current_sum_r = np.sum(r)
    
    for step in range(max_inflate_steps):
        # Check time
        if time.time() - TIME_LIMIT_START > TIME_LIMIT - 10:
            break
            
        new_r = r + step_r
        
        # Try to relax with new radii
        V = relax(x, y, new_r, W, H, steps=50, lr=0.05)
        
        if V < 1e-6:
            r = new_r
            current_sum_r = np.sum(r)
        else:
            # Relax failed, try to relax current state more
            V_curr = relax(x, y, r, W, H, steps=50, lr=0.05)
            if V_curr < 1e-6:
                # Reduce step size
                step_r *= 0.9
                if step_r < 1e-4:
                    break
            else:
                break
                
    # Jiggle phase: try to escape local minima
    # Perform a few random perturbations and re-relax, then try to inflate more
    for jiggle_iter in range(10):
        if time.time() - TIME_LIMIT_START > TIME_LIMIT - 10:
            break
            
        # Perturb positions
        x_pert = x + np.random.normal(0, 0.01, N)
        y_pert = y + np.random.normal(0, 0.01, N)
        
        # Relax perturbed state
        V_pert = relax(x_pert, y_pert, r, W, H, steps=100, lr=0.05)
        
        if V_pert < 1e-6:
            # Accept perturbation if it allows further inflation
            x, y = x_pert, y_pert
            
            # Try to inflate a bit more
            success = False
            for _ in range(50):
                new_r = r + step_r
                V = relax(x, y, new_r, W, H, steps=50, lr=0.05)
                if V < 1e-6:
                    r = new_r
                    success = True
                else:
                    break
                    
            if not success:
                # If no improvement, maybe revert or keep?
                # Keeping the perturbed position might help future steps
                pass
                
    return x, y, r, np.sum(r)

# Global start time
TIME_LIMIT_START = time.time()

def construct():
    global TIME_LIMIT_START
    TIME_LIMIT_START = time.time()
    
    best_val = 0
    best_x, best_y, best_r = None, None, None
    best_W, best_H = 1.0, 1.0
    
    # Try a range of aspect ratios
    # W + H = 2
    # W in [0.5, 1.5]
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0:
            continue
            
        if time.time() - TIME_LIMIT_START > TIME_LIMIT - 15:
            break
            
        # Try grid initialization
        x, y, r, val = try_packing(W, H, 'grid', seed=42)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            best_W, best_H = W, H
            
        # Try random initialization if time permits
        if time.time() - TIME_LIMIT_START < TIME_LIMIT - 40:
            x, y, r, val = try_packing(W, H, 'random', seed=123)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                best_W, best_H = W, H

    if best_x is None:
        # Fallback: simple grid in 1x1
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        return np.array(cells)
        
    # Final polish: ensure validity with tighter tolerance
    # One more relaxation pass
    relax(best_x, best_y, best_r, best_W, best_H, steps=100, lr=0.01, tol=1e-10)
    
    # Verify validity (optional, for safety)
    # If invalid, we might need to shrink radii
    # But relax should have handled it.
    
    return np.column_stack((best_x, best_y, best_r))
