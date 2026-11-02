"""Improved circle packing: SA with vectorized overlap + gradient polish."""
import numpy as np
import time

N = 21
DEADLINE = 135.0  # safety margin before 150s kill


def _overlap_penalty(x, y, r, w, h):
    """Compute violation amount: sum of max(0, r_i+r_j - dist_ij) for all pairs,
    plus boundary violations. Returns (penalty, grad_x, grad_y, grad_r)."""
    n = len(r)
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-30)
    min_dist = r[:, None] + r[None, :]
    # Overlap amount (positive means overlap)
    overlap = np.maximum(0.0, min_dist - d)
    # Set diagonal to 0
    np.fill_diagonal(overlap, 0.0)
    # Boundary violations
    bnd = np.maximum(0.0, r - x) + np.maximum(0.0, r - (w - x)) + \
          np.maximum(0.0, r - y) + np.maximum(0.0, r - (h - y))
    
    penalty = np.sum(overlap) + np.sum(bnd)
    
    # Gradients
    # d(overlap)/d(x_i) for pair (i,j): if overlap>0, sign = (x_i - x_j)/d_ij
    # Sum over all j
    inv_d = np.where(d > 1e-15, 1.0 / d, 0.0)
    # grad_x[i] = sum_j overlap_ij * (x_i - x_j) / d_ij  (for overlapping pairs)
    # But overlap = max(0, r_i+r_j - d), so d(overlap)/d(x_i) = (x_i-x_j)/d_ij when overlapping
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    # Only compute for overlapping pairs (and boundary)
    mask = overlap > 0
    # For each overlapping pair (i,j), contribution to grad:
    # d(overlap_ij)/d(x_i) = (x_i - x_j)/d_ij
    # d(overlap_ij)/d(x_j) = (x_j - x_i)/d_ij
    # d(overlap_ij)/d(r_i) = 1
    # d(overlap_ij)/d(r_j) = 1
    
    # Vectorized: for each i, sum over j where mask[i,j]
    # grad_x[i] += sum_j mask[i,j] * (x_i - x_j) * inv_d[i,j]
    # This is: grad_x = (mask * (x[:,None]-x[None,:]) * inv_d).sum(axis=1)
    # But we need to be careful: the gradient of max(0, r_i+r_j-d) wrt x_i is (x_i-x_j)/d when r_i+r_j>d
    
    # Actually let's use a smoother approach for SA: just use the penalty value
    # and do gradient steps on a smoothed version
    
    # For the gradient polish, we'll use a different approach
    return penalty


def _gradient_step(x, y, r, w, h, lr, iters=50):
    """Do gradient descent on a smoothed objective to reduce overlaps and maximize radii."""
    n = len(r)
    # Objective: maximize sum(r) - lambda * smooth_overlap
    # Use soft-min: -log(sum(exp(-(r_i+r_j-d)/tau))) approximates min
    # Simpler: use sum of (r_i+r_j-d)^2 for overlapping pairs + boundary terms
    # Plus a term to grow radii: -mu * sum(r)
    
    for it in range(iters):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-30)
        min_d = r[:, None] + r[None, :]
        
        # Smooth overlap: (r_i+r_j - d) when positive, squared for smoothness
        diff = min_d - d
        # Gradient of sum(diff^2) for diff>0
        # d/dx_i sum_j diff_ij^2 = 2 * sum_j diff_ij * (x_i - x_j)/d_ij
        inv_d = np.where(d > 1e-15, 1.0/d, 0.0)
        
        # Only where diff > 0
        mask = diff > 0
        grad_x = 2.0 * np.sum(np.where(mask, diff * (x[:, None] - x[None, :]) * inv_d, 0.0), axis=1)
        grad_y = 2.0 * np.sum(np.where(mask, diff * (y[:, None] - y[None, :]) * inv_d, 0.0), axis=1)
        grad_r = 2.0 * np.sum(np.where(mask, diff, 0.0), axis=1)
        
        # Boundary gradients
        # d/dx_i max(0, r_i-x_i)^2 = -2*max(0, r_i-x_i)
        # d/dx_i max(0, r_i-(w-x_i))^2 = 2*max(0, r_i-(w-x_i))
        gb_x = -2.0 * np.maximum(0.0, r - x) + 2.0 * np.maximum(0.0, r - (w - x))
        gb_y = -2.0 * np.maximum(0.0, r - y) + 2.0 * np.maximum(0.0, r - (h - y))
        gb_r = 2.0 * (np.maximum(0.0, r - x) + np.maximum(0.0, r - (w - x)) +
                      np.maximum(0.0, r - y) + np.maximum(0.0, r - (h - y)))
        
        # We want to maximize sum(r), so gradient of -sum(r) is -1
        # Total gradient for minimization: grad - mu*(1) for r
        mu = 1.0  # weight for growing radii
        
        gx = grad_x + gb_x
        gy = grad_y + gb_y
        gr = grad_r + gb_r - mu * np.ones(n)
        
        x -= lr * gx
        y -= lr * gy
        r -= lr * gr
        r = np.maximum(r, 1e-6)
        
        # Clamp to boundaries (soft)
        x = np.clip(x, 0.0, w)
        y = np.clip(y, 0.0, h)
    
    return x, y, r


def _evaluate(x, y, r, w, h):
    """Check validity and return sum of radii if valid."""
    n = len(r)
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) * (r[i] + r[j]) - 1e-9:
                return -1.0
    # Check boundaries
    for i in range(n):
        if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
            return -1.0
        if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
            return -1.0
    return np.sum(r)


def _sa_optimize(w, h, n_restarts=6, time_budget=18.0):
    """Simulated annealing to find good packing in rectangle w x h."""
    n = N
    best_val = -1.0
    best_x = best_y = best_r = None
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        # Initialize: random positions, small radii
        x = np.random.uniform(0.1 * w, 0.9 * w, n)
        y = np.random.uniform(0.1 * h, 0.9 * h, n)
        r = np.full(n, 0.02 * min(w, h))
        
        T = 0.1
        T_min = 1e-6
        alpha = 0.9995
        iters = 20000
        
        cur_val = -1.0
        cur_x, cur_y, cur_r = x.copy(), y.copy(), r.copy()
        
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            # Perturb a random circle
            idx = np.random.randint(n)
            scale = T * 0.1
            dx = np.random.normal(0, scale)
            dy = np.random.normal(0, scale)
            dr = np.random.normal(0, scale * 0.5)
            
            new_x = cur_x[idx] + dx
            new_y = cur_y[idx] + dy
            new_r = max(1e-6, cur_r[idx] + dr)
            
            # Quick check: just compute the objective (penalty-based)
            # Use a fast penalty: sum of overlaps + boundary violations
            # This is O(n) for the moved circle
            pen = 0.0
            for j in range(n):
                if j == idx:
                    continue
                d = np.sqrt((new_x - cur_x[j])**2 + (new_y - cur_y[j])**2)
                ov = new_r + cur_r[j] - d
                if ov > 0:
                    pen += ov
            # Boundary
            pen += max(0.0, new_r - new_x) + max(0.0, new_r - (w - new_x))
            pen += max(0.0, new_r - new_y) + max(0.0, new_r - (h - new_y))
            
            # Objective: we want to minimize penalty and maximize sum(r)
            # Use: -sum(r) + lambda * penalty
            lam = 10.0
            obj = -np.sum(cur_r) + lam * pen
            cur_obj = -np.sum(cur_r) + lam * 0.0  # current has no penalty if valid
            
            # Actually let's track current penalty too
            # For simplicity, accept if new penalty < current penalty or with prob exp(-delta/T)
            # Let's just use the penalty as the cost
            # Cost = penalty (lower is better), but we also want larger radii
            # Combined cost = penalty - 0.1 * sum(r)
            
            # Recompute current penalty (expensive but needed)
            # Actually, let's use a simpler approach: track penalty incrementally
            
            # For speed, let's just do: if new_pen < cur_pen or accept with probability
            # We need cur_pen. Let me restructure.
            
            # This is getting complex. Let me use a simpler SA that just minimizes
            # a combined cost function computed from scratch each time (vectorized).
            pass
        
        # After SA, do gradient polish
        if time.time() < DEADLINE - 5:
            for _ in range(3):
                cur_x, cur_y, cur_r = _gradient_step(cur_x, cur_y, cur_r, w, h, lr=0.01, iters=200)
            
            val = _evaluate(cur_x, cur_y, cur_r, w, h)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = cur_x.copy(), cur_y.copy(), cur_r.copy()
    
    return best_val, best_x, best_y, best_r


def _sa_optimize_v2(w, h, time_budget=18.0):
    """Better SA: vectorized penalty evaluation, multiple perturbations."""
    n = N
    best_val = -1.0
    best_x = best_y = best_r = None
    start_time = time.time()
    
    n_restarts = 4
    per_restart = time_budget / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize on a grid with jitter
        side = int(np.ceil(np.sqrt(n)))
        x = np.array([(i + 0.5) / side * w + np.random.uniform(-0.1, 0.1) * w / side for i in range(n)])
        y = np.array([(j + 0.5) / side * h + np.random.uniform(-0.1, 0.1) * h / side for j in range(n)])
        r = np.full(n, 0.4 / side * min(w, h))
        
        T = 0.05
        T_min = 1e-7
        alpha = 0.999
        max_iters = int(per_restart * 8000)  # target ~8k iters/sec
        
        cur_pen = _fast_penalty(x, y, r, w, h)
        cur_sum_r = np.sum(r)
        cur_cost = cur_pen - 0.5 * cur_sum_r
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            if it % 100 == 0 and time.time() - start_time > time_budget:
                break
            
            # Perturb 1-2 random circles
            n_perturb = 1 if np.random.random() < 0.8 else 2
            indices = np.random.choice(n, n_perturb, replace=False)
            
            scale = T * 0.5
            new_x = x.copy()
            new_y = y.copy()
            new_r = r.copy()
            
            for idx in indices:
                new_x[idx] += np.random.normal(0, scale)
                new_y[idx] += np.random.normal(0, scale)
                new_r[idx] = max(1e-6, new_r[idx] + np.random.normal(0, scale * 0.3))
            
            new_pen = _fast_penalty(new_x, new_y, new_r, w, h)
            new_sum_r = np.sum(new_r)
            new_cost = new_pen - 0.5 * new_sum_r
            
            delta = new_cost - cur_cost
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                x, y, r = new_x, new_y, new_r
                cur_pen, cur_sum_r, cur_cost = new_pen, new_sum_r, new_cost
            
            T = max(T * alpha, T_min)
        
        # Gradient polish
        if time.time() < DEADLINE - 5:
            for _ in range(5):
                x, y, r = _gradient_step(x, y, r, w, h, lr=0.005, iters=100)
            
            val = _evaluate(x, y, r, w, h)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    return best_val, best_x, best_y, best_r


def _fast_penalty(x, y, r, w, h):
    """Vectorized penalty: sum of overlaps + boundary violations."""
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-30)
    min_d = r[:, None] + r[None, :]
    overlap = np.maximum(0.0, min_d - d)
    np.fill_diagonal(overlap, 0.0)
    pen = np.sum(overlap)
    bnd = np.maximum(0.0, r - x) + np.maximum(0.0, r - (w - x)) + \
          np.maximum(0.0, r - y) + np.maximum(0.0, r - (h - y))
    pen += np.sum(bnd)
    return pen


def construct():
    start_time = time.time()
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5]
    
    best_val = -1.0
    best_result = None
    
    for ar in aspect_ratios:
        if time.time() > DEADLINE - 10:
            break
        
        # Perimeter = 2*(w+h) = 4, so w+h = 2
        # w/h = ar => w = ar*h, so (ar+1)*h = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        time_left = DEADLINE - time.time()
        budget = min(15.0, time_left * 0.8)
        
        val, bx, by, br = _sa_optimize_v2(w, h, time_budget=budget)
        
        if val > best_val:
            best_val = val
            best_result = np.column_stack([bx, by, br])
    
    # If no valid solution found, fall back to grid
    if best_result is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
