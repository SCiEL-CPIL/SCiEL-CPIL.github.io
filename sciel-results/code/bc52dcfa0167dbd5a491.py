import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _smooth_max(values, beta):
    """
    Compute smooth maximum of values using LogSumExp.
    values: array of shape (M,)
    beta: inverse temperature
    """
    if len(values) == 0:
        return 0.0
    v_max = np.max(values)
    # Handle case where v_max is very large to prevent overflow in exp
    # exp(beta * (values - v_max)) is safe if beta * (values - v_max) < 700
    shifted = beta * (values - v_max)
    # Clip shifted to avoid overflow in exp, though theoretically values <= v_max
    # If values can be > v_max due to numerical issues, clip.
    # Actually, v_max is the max, so values - v_max <= 0. exp(<=0) <= 1. Safe.
    sum_exp = np.sum(np.exp(shifted))
    return (v_max + np.log(sum_exp)) / beta

def _compute_objective(x, beta_overlap, beta_perim, w_overlap=1.0, w_perim=1.0):
    """
    x: array of shape (3*N,) -> [x_1...x_N, y_1...y_N, r_1...r_N]
    """
    xs = x[:N]
    ys = x[N:2*N]
    rs = x[2*N:]
    
    # 1. Overlap Penalty
    # Calculate all pairwise distances
    # Using broadcasting for NxN matrix
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    # Avoid sqrt(0) issues, though dist=0 is fine for penalty
    dist = np.sqrt(dist_sq)
    
    # Calculate all pairwise sum of radii
    rs_i = rs[:, np.newaxis]
    rs_j = rs[np.newaxis, :]
    sum_r = rs_i + rs_j
    
    # Overlap values g_ij = sum_r - dist
    # We only care about i < j to avoid double counting and self-overlap (which is 0)
    # But smooth max over the whole matrix (excluding diagonal) is fine.
    # Set diagonal to -inf so it doesn't affect max
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(sum_r, 0) # Doesn't matter as dist is inf
    
    g = sum_r - dist
    
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    g_pairs = g[iu]
    
    # Smooth max of overlaps
    s_overlap = _smooth_max(g_pairs, beta_overlap)
    
    # 2. Perimeter Penalty
    # w = max(x+r) - min(x-r)
    # h = max(y+r) - min(y-r)
    
    x_max_vals = xs + rs
    x_min_vals = xs - rs
    y_max_vals = ys + rs
    y_min_vals = ys - rs
    
    # Smooth max/min
    max_x = _smooth_max(x_max_vals, beta_perim)
    min_x = -_smooth_max(-x_min_vals, beta_perim)
    max_y = _smooth_max(y_max_vals, beta_perim)
    min_y = -_smooth_max(-y_min_vals, beta_perim)
    
    w = max_x - min_x
    h = max_y - min_y
    
    perim_val = 2 * (w + h) - 4.0
    
    # We want perim_val <= 0. If > 0, penalize.
    # Smooth max of (perim_val, 0)
    s_perim = _smooth_max(np.array([perim_val, 0.0]), beta_perim)
    
    val = -np.sum(rs) + w_overlap * s_overlap + w_perim * s_perim
    
    return val

def _gradient(x, beta_overlap, beta_perim, w_overlap=1.0, w_perim=1.0):
    """
    Numerical gradient is safer for complex smooth max, but L-BFGS-B can use approx.
    However, analytic gradient is faster.
    Given time constraints, let's use approx_grad first, or implement analytic if simple.
    Analytic gradient of LogSumExp:
    d/dx (1/beta * log(sum(exp(beta * g_i)))) = sum(exp(beta * g_i) * beta * dg_i/dx) / sum(exp(beta * g_i))
    = sum(prob_i * dg_i/dx) where prob_i = exp(beta * g_i) / sum(exp(beta * g_j))
    
    This is doable.
    """
    xs = x[:N]
    ys = x[N:2*N]
    rs = x[2*N:]
    
    grad = np.zeros_like(x)
    
    # 1. Overlap Gradient
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    rs_i = rs[:, np.newaxis]
    rs_j = rs[np.newaxis, :]
    sum_r = rs_i + rs_j
    np.fill_diagonal(sum_r, 0)
    
    g = sum_r - dist
    iu = np.triu_indices(N, k=1)
    g_pairs = g[iu]
    
    # Weights for smooth max
    v_max = np.max(g_pairs)
    shifted = beta_overlap * (g_pairs - v_max)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    probs = exp_shifted / sum_exp # shape (M,)
    
    # dS_overlap/dx_k = sum_m probs_m * d(g_m)/dx_k
    # g_m = r_i + r_j - dist_ij
    # d(g_m)/dx_i = - (x_i - x_j) / dist_ij
    # d(g_m)/dx_j = (x_i - x_j) / dist_ij
    # d(g_m)/dr_i = 1
    # d(g_m)/dr_j = 1
    
    # We need to accumulate gradients.
    # Loop over pairs is 210 iterations, which is fast.
    idx_i, idx_j = iu
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        p = probs[k]
        d = dist[i, j]
        if d < 1e-10:
            continue
        # dx component
        dx_val = (xs[i] - xs[j]) / d
        dy_val = (ys[i] - ys[j]) / d
        
        # d g / d x_i = -dx_val
        grad[i] += -p * dx_val
        # d g / d x_j = dx_val
        grad[j] += p * dx_val
        
        # d g / d y_i = -dy_val
        grad[N+i] += -p * dy_val
        # d g / d y_j = dy_val
        grad[N+j] += p * dy_val
        
        # d g / d r_i = 1
        grad[2*N+i] += p
        # d g / d r_j = 1
        grad[2*N+j] += p
        
    # Multiply by w_overlap
    grad[:2*N] *= w_overlap
    grad[2*N:] *= w_overlap
    
    # 2. Perimeter Gradient
    x_max_vals = xs + rs
    x_min_vals = xs - rs
    y_max_vals = ys + rs
    y_min_vals = ys - rs
    
    # Helper to get probs for smooth max
    def get_probs(vals, beta):
        v_max = np.max(vals)
        shifted = beta * (vals - v_max)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        return exp_shifted / sum_exp
    
    # max_x = smooth_max(x_max_vals)
    # min_x = -smooth_max(-x_min_vals)
    # w = max_x - min_x
    # h = max_y - min_y
    # perim_val = 2*(w+h) - 4
    
    # We need d(perim_val)/dx, dy, dr
    
    # d(max_x)/dx_i = probs_max_x[i]
    # d(max_x)/dr_i = probs_max_x[i]
    
    # d(min_x)/dx_i = -probs_min_x[i]  (where probs_min_x is for -x_min_vals)
    # d(min_x)/dr_i = probs_min_x[i]
    
    probs_max_x = get_probs(x_max_vals, beta_perim)
    probs_min_x = get_probs(-x_min_vals, beta_perim) # This is probs for max of -x_min
    probs_max_y = get_probs(y_max_vals, beta_perim)
    probs_min_y = get_probs(-y_min_vals, beta_perim)
    
    # d w / d x_i = probs_max_x[i] - (-probs_min_x[i]) = probs_max_x[i] + probs_min_x[i] ?
    # Wait. min_x = - smooth_max(-x_min_vals).
    # d(min_x)/dx_i = - (d/dx_i smooth_max(-x_min_vals)) = - (probs_min_x[i] * d(-x_min_vals[i])/dx_i)
    # -x_min_vals[i] = -(x_i - r_i) = -x_i + r_i.
    # d/dx_i (-x_i + r_i) = -1.
    # So d(min_x)/dx_i = - (probs_min_x[i] * -1) = probs_min_x[i].
    # d(min_x)/dr_i = - (probs_min_x[i] * 1) = -probs_min_x[i].
    
    # So:
    # d w / d x_i = probs_max_x[i] + probs_min_x[i]
    # d w / d r_i = probs_max_x[i] - probs_min_x[i]
    
    # Similarly for h:
    # d h / d y_i = probs_max_y[i] + probs_min_y[i]
    # d h / d r_i = probs_max_y[i] - probs_min_y[i]
    
    d_w_dx = probs_max_x + probs_min_x
    d_w_dr = probs_max_x - probs_min_x
    d_h_dy = probs_max_y + probs_min_y
    d_h_dr = probs_max_y - probs_min_y
    
    # d perim_val / d x_i = 2 * d_w_dx[i]
    # d perim_val / d y_i = 2 * d_h_dy[i]
    # d perim_val / d r_i = 2 * (d_w_dr[i] + d_h_dr[i])
    
    # We want to penalize if perim_val > 0.
    # s_perim = smooth_max(perim_val, 0)
    # If perim_val > 0, d s_perim / d perim_val = 1 (approx, if beta large).
    # More precisely, it's the probability weight on perim_val in the smooth max of [perim_val, 0].
    
    v_perim = np.array([2*(w + h) - 4.0, 0.0])
    probs_perim = get_probs(v_perim, beta_perim)
    # If perim_val > 0, probs_perim[0] is close to 1. If < 0, close to 0.
    weight_perim = probs_perim[0]
    
    grad[:N] += w_perim * weight_perim * 2 * d_w_dx
    grad[N:2*N] += w_perim * weight_perim * 2 * d_h_dy
    grad[2*N:] += w_perim * weight_perim * 2 * (d_w_dr + d_h_dr)
    
    # Base gradient for -sum(rs)
    grad[2*N:] -= 1.0
    
    return grad

def _separate_circles(circles, iterations=20):
    """Simple separation to fix initial overlaps."""
    circles = circles.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    if dist < 1e-10:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    ux, uy = dx/dist, dy/dist
                    push = overlap / 2 + 1e-6
                    circles[i, 0] += ux * push
                    circles[i, 1] += uy * push
                    circles[j, 0] -= ux * push
                    circles[j, 1] -= uy * push
                    moved = True
        if not moved: break
    return circles

def _validate(circles):
    if circles.shape[0] != N: return False
    xs, ys, rs = circles[:,0], circles[:,1], circles[:,2]
    if np.any(rs <= 0): return False
    min_x, max_x = np.min(xs-rs), np.max(xs+rs)
    min_y, max_y = np.min(ys-rs), np.max(ys+rs)
    w, h = max_x-min_x, max_y-min_y
    if 2*(w+h) > 4.0 + 1e-9: return False
    for i in range(N):
        for j in range(i+1, N):
            dx, dy = xs[i]-xs[j], ys[i]-ys[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < rs[i]+rs[j] - 1e-9: return False
    return True

def _get_score(circles):
    return np.sum(circles[:,2])

def construct():
    start_time = time.time()
    deadline = start_time + 145 # 5s margin
    
    best_score = 0
    best_circles = None
    
    # Initializations
    inits = []
    
    # 1. 5x5 Grid (minus corners)
    pos = []
    for i in range(5):
        for j in range(5):
            if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                continue
            pos.append([0.5 + 0.5*i, 0.5 + 0.5*j])
    pos = np.array(pos)
    r = 0.49
    # Scale to perimeter 4
    min_x, max_x = np.min(pos[:,0]-r), np.max(pos[:,0]+r)
    min_y, max_y = np.min(pos[:,1]-r), np.max(pos[:,1]+r)
    w, h = max_x-min_x, max_y-min_y
    scale = 4.0 / (2*(w+h))
    pos[:,0] *= scale
    pos[:,1] *= scale
    r_scaled = r * scale
    x0 = np.concatenate([pos[:,0], pos[:,1], np.full(N, r_scaled)])
    inits.append(x0)
    
    # 2. 3x7 Grid
    pos = []
    for i in range(3):
        for j in range(7):
            pos.append([0.5 + 0.5*i, 0.5 + 0.5*j])
    pos = np.array(pos)
    r = 0.49
    min_x, max_x = np.min(pos[:,0]-r), np.max(pos[:,0]+r)
    min_y, max_y = np.min(pos[:,1]-r), np.max(pos[:,1]+r)
    w, h = max_x-min_x, max_y-min_y
    scale = 4.0 / (2*(w+h))
    pos[:,0] *= scale
    pos[:,1] *= scale
    r_scaled = r * scale
    x0 = np.concatenate([pos[:,0], pos[:,1], np.full(N, r_scaled)])
    inits.append(x0)
    
    # 3. Hexagonal
    pos = []
    # 4 rows: 5, 6, 5, 5? No, 21 circles.
    # 5, 6, 5, 5 is 21.
    rows = [5, 6, 5, 5]
    y = 0
    for count in rows:
        x_start = - (count - 1) * 0.866 # approx hex spacing
        for i in range(count):
            pos.append([x_start + i * 1.732, y])
        y += 1.5
    pos = np.array(pos)
    r = 0.9
    min_x, max_x = np.min(pos[:,0]-r), np.max(pos[:,0]+r)
    min_y, max_y = np.min(pos[:,1]-r), np.max(pos[:,1]+r)
    w, h = max_x-min_x, max_y-min_y
    scale = 4.0 / (2*(w+h))
    pos[:,0] *= scale
    pos[:,1] *= scale
    r_scaled = r * scale
    x0 = np.concatenate([pos[:,0], pos[:,1], np.full(N, r_scaled)])
    inits.append(x0)
    
    # 4. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        xs = rng.uniform(0, 1, N)
        ys = rng.uniform(0, 1, N)
        rs = np.full(N, 0.3)
        x0 = np.concatenate([xs, ys, rs])
        # Scale
        min_x, max_x = np.min(xs-rs), np.max(xs+rs)
        min_y, max_y = np.min(ys-rs), np.max(ys+rs)
        w, h = max_x-min_x, max_y-min_y
        scale = 4.0 / (2*(w+h))
        x0[:N] *= scale
        x0[N:2*N] *= scale
        x0[2*N:] *= scale
        inits.append(x0)
        
    # Optimization
    beta_schedule = [5, 20, 50, 100, 200, 500, 1000]
    
    for x_init in inits:
        if time.time() > deadline: break
        
        x = x_init.copy()
        # Initial separation
        circles = np.column_stack([x[:N], x[N:2*N], x[2*N:]])
        circles = _separate_circles(circles)
        x = np.concatenate([circles[:,0], circles[:,1], circles[:,2]])
        
        for beta in beta_schedule:
            if time.time() > deadline: break
            
            try:
                res = minimize(
                    _compute_objective, 
                    x, 
                    args=(beta, beta, 1.0, 1.0),
                    method='L-BFGS-B',
                    jac=_gradient,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < _compute_objective(x, beta, beta, 1.0, 1.0):
                    x = res.x
            except Exception:
                pass
                
        # Final high precision polish
        if time.time() < deadline - 2:
             try:
                res = minimize(
                    _compute_objective, 
                    x, 
                    args=(2000, 2000, 1.0, 1.0),
                    method='L-BFGS-B',
                    jac=_gradient,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < _compute_objective(x, 2000, 2000, 1.0, 1.0):
                    x = res.x
             except Exception:
                 pass

        circles = np.column_stack([x[:N], x[N:2*N], x[2*N:]])
        
        # Validate
        if _validate(circles):
            score = _get_score(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback
        pos = []
        for i in range(5):
            for j in range(5):
                if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                    continue
                pos.append([0.5 + 0.5*i, 0.5 + 0.5*j])
        pos = np.array(pos)
        r = 0.45
        min_x, max_x = np.min(pos[:,0]-r), np.max(pos[:,0]+r)
        min_y, max_y = np.min(pos[:,1]-r), np.max(pos[:,1]+r)
        w, h = max_x-min_x, max_y-min_y
        scale = 4.0 / (2*(w+h))
        pos[:,0] *= scale
        pos[:,1] *= scale
        r_scaled = r * scale
        best_circles = np.column_stack([pos[:,0], pos[:,1], np.full(N, r_scaled)])
        
    return best_circles
