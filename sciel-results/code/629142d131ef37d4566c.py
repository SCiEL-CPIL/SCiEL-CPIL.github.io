import numpy as np
import time
from scipy.optimize import linprog

N = 26

def compute_energy(x, y, r, alpha, beta):
    """
    x, y: (N,)
    r: (N,)
    alpha: weight for overlap penalty
    beta: weight for boundary penalty
    """
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    overlap_penalty = np.sum(overlap**2)
    
    bx1 = np.maximum(0, r - x)
    bx2 = np.maximum(0, r - (1 - x))
    by1 = np.maximum(0, r - y)
    by2 = np.maximum(0, r - (1 - y))
    bound_penalty = np.sum(bx1**2) + np.sum(bx2**2) + np.sum(by1**2) + np.sum(by2**2)
    
    return obj + alpha * overlap_penalty + beta * bound_penalty

def compute_grad(x, y, r, alpha, beta):
    """
    Returns gradients for x, y, r
    """
    N = len(r)
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    gr -= 1.0
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    gx -= 2.0 * alpha * np.sum(overlap * ux, axis=1)
    gy -= 2.0 * alpha * np.sum(overlap * uy, axis=1)
    
    gr += 2.0 * alpha * np.sum(overlap, axis=1)
    
    bx1 = np.maximum(0, r - x)
    bx2 = np.maximum(0, r - (1 - x))
    by1 = np.maximum(0, r - y)
    by2 = np.maximum(0, r - (1 - y))
    
    gx += 2.0 * beta * (bx2 - bx1)
    gy += 2.0 * beta * (by2 - by1)
    gr += 2.0 * beta * (bx1 + bx2 + by1 + by2)
    
    return gx, gy, gr

def is_valid(x, y, r, tol=1e-5):
    if np.any(r < 0): return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol): return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol): return False
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def solve_radii_lp(x, y, epsilon=1e-7):
    """
    Solves the Linear Programming problem to maximize sum(r)
    given fixed centers x, y.
    """
    n = len(x)
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    A_rows = []
    b_vals = []
    
    # Pairwise constraints: r_i + r_j <= d_ij - epsilon
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < epsilon:
                return None
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_vals.append(d - epsilon)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i) - epsilon
    b_i = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_vals.append(b_i[i] - epsilon)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = res.x
            # Verify validity
            if is_valid(x, y, r):
                return r
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = 0.0
    best_config = None
    
    n_restarts = 100
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialization
        if restart % 2 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
        else:
            # Random init
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            
        # Relaxation Phase: Optimize centers using fixed radius penalty
        R_relax = 0.08
        r_fixed = np.full(N, R_relax)
        
        alpha = 10.0
        beta = 10.0
        lr = 0.02
        
        for step in range(300):
            if time.time() > deadline - 5:
                break
                
            gx, gy, gr = compute_grad(x, y, r_fixed, alpha, beta)
            
            x -= lr * gx
            y -= lr * gy
            
            # Project to bounds
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Decay learning rate
            lr *= 0.995
            
            # Increase penalty weights to enforce separation
            if step % 50 == 0:
                alpha *= 1.2
                beta *= 1.2
                
        # Radius Maximization Phase: Solve LP
        r_opt = solve_radii_lp(x, y)
        
        if r_opt is not None:
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_config = (x.copy(), y.copy(), r_opt.copy())
                
    # Fallback
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
