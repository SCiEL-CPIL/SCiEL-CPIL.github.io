"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_valid(x, y, r, tol=1e-9):
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    if np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    n = len(r)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol)**2:
                return False
    return True

def _score(x, y, r):
    if _is_valid(x, y, r):
        return float(np.sum(r))
    return 0.0

def _penalty_and_grad(x, y, r, lam=100.0):
    n = len(r)
    penalty = 0.0
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Boundary penalties
    for i in range(n):
        if x[i] < 0:
            penalty += x[i]*x[i]
            gx[i] += 2 * x[i]
        elif x[i] > 1:
            penalty += (x[i]-1)**2
            gx[i] += 2 * (x[i]-1)
        if y[i] < 0:
            penalty += y[i]*y[i]
            gy[i] += 2 * y[i]
        elif y[i] > 1:
            penalty += (y[i]-1)**2
            gy[i] += 2 * (y[i]-1)
        if r[i] < 0:
            penalty += r[i]*r[i]
            gr[i] += 2 * r[i]
    
    # Pairwise overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-15)
            sum_r = r[i] + r[j]
            overlap = sum_r - dist
            if overlap > 0:
                penalty += overlap * overlap
                factor = 2.0 * overlap / dist
                gx[i] -= lam * factor * dx
                gx[j] += lam * factor * dx
                gy[i] -= lam * factor * dy
                gy[j] += lam * factor * dy
                gr[i] += lam * 2.0 * overlap
                gr[j] += lam * 2.0 * overlap
    
    loss = -np.sum(r) + lam * penalty
    gr = -1.0 + gr
    return loss, gx, gy, gr

def _optimize(x0, y0, r0, max_iters=3000, lr_init=0.002, deadline=None, lam=100.0):
    n = len(r0)
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    lr = lr_init
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    mu = 0.9  # momentum coefficient
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, gx, gy, gr = _penalty_and_grad(x, y, r, lam=lam)
        
        # Momentum update
        momentum_x = mu * momentum_x - lr * gx
        momentum_y = mu * momentum_y - lr * gy
        momentum_r = mu * momentum_r - lr * gr
        
        x += momentum_x
        y += momentum_y
        r += momentum_r
        
        # Clamp
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 1e-6, 0.5)
        
        cur_score = _score(x, y, r)
        if cur_score > best_score:
            best_score = cur_score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Decay learning rate
        if it % 200 == 0:
            lr = max(1e-6, lr * 0.9)
    
    return best_x, best_y, best_r, best_score

def _uniform_grow(x, y, r, max_steps=300, deadline=None):
    scale = 1.0
    for _ in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        new_scale = scale * 1.001
        new_r = r * new_scale
        if _is_valid(x, y, new_r):
            scale = new_scale
            r = new_r
        else:
            break
    return r

def _init_layout(seed=0):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append(((i + 0.5) / side, (j + 0.5) / side))
    
    positions = positions[:N]
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    
    x += rng.uniform(-0.003, 0.003, N)
    y += rng.uniform(-0.003, 0.003, N)
    
    r = np.full(N, 0.5 / side - 0.003)
    
    return x, y, r

def construct():
    deadline = time.time() + 135
    
    best_x, best_y, best_r = None, None, None
    best_score = 0.0
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        x, y, r = _init_layout(seed=restart * 97 + 3)
        
        opt_x, opt_y, opt_r, opt_score = _optimize(
            x, y, r,
            max_iters=2500,
            lr_init=0.001,
            deadline=deadline
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_x, best_y, best_r = opt_x.copy(), opt_y.copy(), opt_r.copy()
        
        if time.time() < deadline - 5:
            grown_r = _uniform_grow(best_x, best_y, best_r, max_steps=200, deadline=deadline)
            grown_score = _score(best_x, best_y, grown_r)
            if grown_score > best_score:
                best_score = grown_score
                best_r = grown_r
    
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_x, best_y, best_r])
# EVOLVE-BLOCK-END
