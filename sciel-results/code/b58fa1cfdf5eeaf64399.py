"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

Uses a physics-based relaxation with softplus penalty, Delaunay-constrained
optimization, and SLSQP polishing.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9

def _elapsed(t_start):
    return time.time() - t_start

def _remaining(t_start, deadline):
    return max(0.0, deadline - _elapsed(t_start))

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x)
    return np.where(x > 0, x + np.log1p(np.exp(-beta * x)) / beta,
                    np.log1p(np.exp(beta * x)) / beta)

def _softplus_grad(x, beta=50.0):
    """Gradient of softplus."""
    x = np.asarray(x)
    return np.where(x > 0, 1.0 - np.exp(-beta * x),
                    np.exp(beta * x) / (1.0 + np.exp(beta * x)))

def _validate(pos, r, tol=1e-7):
    """Check validity: in square, no overlaps."""
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _enforce_validity(pos, r):
    """Shrink radii to ensure validity."""
    n = len(r)
    for i in range(n):
        # Wall constraints
        r[i] = min(r[i], pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        r[i] = max(0.0, r[i])
    # Overlap constraints (iterative)
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if r[i] + r[j] > d:
                    excess = r[i] + r[j] - d
                    r[i] -= excess * 0.5
                    r[j] -= excess * 0.5
                    r[i] = max(0.0, r[i])
                    r[j] = max(0.0, r[j])
                    changed = True
        if not changed:
            break
    return r

def _delaunay_edges(pos):
    """Build Delaunay triangulation of centers, return edge list."""
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _objective_grad(pos, r, beta=50.0):
    """Compute objective and gradient for physics-based relaxation.
    
    Objective: -sum(r) + penalty for violations (softplus-based)
    """
    n = len(r)
    score = np.sum(r)
    
    # Wall violations
    wall_viol = np.zeros(4 * n)
    for i in range(n):
        wall_viol[4*i] = r[i] - pos[i, 0]        # left
        wall_viol[4*i+1] = pos[i, 0] - (1.0 - r[i])  # right
        wall_viol[4*i+2] = r[i] - pos[i, 1]        # bottom
        wall_viol[4*i+3] = pos[i, 1] - (1.0 - r[i])  # top
    
    # Overlap violations
    overlap_viol = []
    overlap_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            overlap_viol.append(r[i] + r[j] - d)
            overlap_pairs.append((i, j))
    overlap_viol = np.array(overlap_viol)
    
    # Softplus penalty
    penalty = np.sum(_softplus(wall_viol, beta)) + np.sum(_softplus(overlap_viol, beta))
    obj = -score + penalty
    
    # Gradient
    grad = np.zeros(3 * n)
    
    # Wall gradient contributions
    for i in range(n):
        for k in range(4):
            g = _softplus_grad(wall_viol[4*i+k], beta)
            if g < 1e-12:
                continue
            if k == 0:  # left wall: r_i - x_i
                grad[2*i] -= g      # x_i
                grad[2*n+i] += g    # r_i
            elif k == 1:  # right wall: x_i - (1 - r_i)
                grad[2*i] += g
                grad[2*n+i] += g
            elif k == 2:  # bottom: r_i - y_i
                grad[2*i+1] -= g
                grad[2*n+i] += g
            elif k == 3:  # top: y_i - (1 - r_i)
                grad[2*i+1] += g
                grad[2*n+i] += g
    
    # Overlap gradient contributions
    for idx, (i, j) in enumerate(overlap_pairs):
        g = _softplus_grad(overlap_viol[idx], beta)
        if g < 1e-12:
            continue
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.hypot(dx, dy)
        if d < 1e-12:
            continue
        ux, uy = dx / d, dy / d
        # d(overlap)/dx_i = -ux, d(overlap)/dx_j = ux
        grad[2*i] -= g * ux
        grad[2*i+1] -= g * uy
        grad[2*j] += g * ux
        grad[2*j+1] += g * uy
        # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
        grad[2*n+i] += g
        grad[2*n+j] += g
    
    # Score gradient: d(-sum(r))/dr_i = -1
    grad[2*n:] -= 1.0
    
    return obj, grad

def _physics_relax(pos, r, t_start, deadline, beta=60.0, max_iter=200):
    """Physics-based relaxation using gradient descent with line search."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    x = np.concatenate([pos.flatten(), r])
    lr = 0.01
    
    for it in range(max_iter):
        if _elapsed(t_start) > deadline:
            break
        
        obj, grad = _objective_grad(pos, r, beta)
        
        # Gradient descent step with line search
        best_step_obj = obj
        best_step_x = x.copy()
        step_lr = lr
        
        for ls in range(20):
            trial_x = x - step_lr * grad
            trial_pos = trial_x[:2*n].reshape(n, 2)
            trial_r = trial_x[2*n:]
            
            # Clip to bounds
            trial_pos[:, 0] = np.clip(trial_pos[:, 0], 0.0, 1.0)
            trial_pos[:, 1] = np.clip(trial_pos[:, 1], 0.0, 1.0)
            trial_r = np.clip(trial_r, 0.0, 0.5)
            
            trial_obj, _ = _objective_grad(trial_pos, trial_r, beta)
            
            if trial_obj < best_step_obj:
                best_step_obj = trial_obj
                best_step_x = trial_x.copy()
            
            if trial_obj < obj - 1e-8:
                break
            step_lr *= 0.5
        
        x = best_step_x
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        
        # Enforce validity
        r = _enforce_validity(pos, r)
        x = np.concatenate([pos.flatten(), r])
        
        score = np.sum(r)
        if _validate(pos, r) and score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
        
        # Adaptive learning rate
        if best_step_obj < obj - 1e-6:
            lr = min(lr * 1.1, 0.05)
        else:
            lr = max(lr * 0.8, 0.001)
    
    return best_pos, best_r

def _polish_analytic(pos, r, t_start, deadline, beta=80.0):
    """High-precision polish using SLSQP with tight tolerances."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    edges = _delaunay_edges(pos)
    
    for _ in range(15):
        if _elapsed(t_start) > deadline:
            break
        
        x0 = np.concatenate([pos.flatten(), r])
        
        def objective(p):
            return -np.sum(p[2*n:])
        
        cons = []
        for i in range(n):
            xi, yi, ri = 2*i, 2*i+1, 2*n+i
            # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda p, xi=xi, ri=ri: p[xi] - p[ri],
                         'jac': lambda p, xi=xi, ri=ri: np.concatenate([[1.0]*xi, [-1.0], [0.0]*(2*n-1-xi), [0.0]*i, [1.0], [0.0]*(n-1-i)])})
            # x_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda p, xi=xi, ri=ri: 1.0 - p[xi] - p[ri],
                         'jac': lambda p, xi=xi, ri=ri: np.concatenate([[-1.0]*xi, [1.0], [0.0]*(2*n-1-xi), [0.0]*i, [-1.0], [0.0]*(n-1-i)])})
            # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda p, yi=yi, ri=ri: p[yi] - p[ri],
                         'jac': lambda p, yi=yi, ri=ri: np.concatenate([[1.0]*yi, [-1.0], [0.0]*(2*n-1-yi), [0.0]*i, [1.0], [0.0]*(n-1-i)])})
            # y_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda p, yi=yi, ri=ri: 1.0 - p[yi] - p[ri],
                         'jac': lambda p, yi=yi, ri=ri: np.concatenate([[-1.0]*yi, [1.0], [0.0]*(2*n-1-yi), [0.0]*i, [-1.0], [0.0]*(n-1-i)])})
            # r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, ri=ri: p[ri],
                         'jac': lambda p, ri=ri: np.concatenate([[0.0]*(2*n+i), [1.0], [0.0]*(n-1-i)])})
        
        for (i, j) in edges:
            xi, yi = 2*i, 2*i+1
            xj, yj = 2*j, 2*j+1
            ri, rj = 2*n+i, 2*n+j
            
            def c_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
                dx = p[xi] - p[xj]
                dy = p[yi] - p[yj]
                d = np.hypot(dx, dy)
                return d - p[ri] - p[rj]
            
            def j_overlap(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
                dx = p[xi] - p[xj]
                dy = p[yi] - p[yj]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    d = 1e-12
                v = np.zeros(3*n)
                v[xi] = dx/d
                v[yi] = dy/d
                v[xj] = -dx/d
                v[yj] = -dy/d
                v[ri] = -1.0
                v[rj] = -1.0
                return v
            
            cons.append({'type': 'ineq', 'fun': c_overlap, 'jac': j_overlap})
        
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 150, 'ftol': 1e-12})
            if res.success or res.fun < objective(x0):
                new_pos = res.x[:2*n].reshape(n, 2)
                new_r = res.x[2*n:]
                if _validate(new_pos, new_r):
                    score = np.sum(new_r)
                    if score > best_score:
                        best_score = score
                        best_pos = new_pos.copy()
                        best_r = new_r.copy()
                    pos = best_pos
                    r = best_r
        except Exception:
            pass
        
        edges = _delaunay_edges(pos)
    
    return best_pos, best_r

def _initial_config_grid(rng, side=5):
    """Grid initial configuration."""
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side
        pos[i, 1] = (row + 0.5) / side
        r[i] = 0.36 / side
    return pos, r

def _initial_config_hex(rng):
    """Hexagonal packing initial configuration."""
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    row_counts = [5, 6, 5, 6, 4]
    idx = 0
    for row_idx, count in enumerate(row_counts):
        if idx >= N:
            break
        row_y = (row_idx + 0.5) / 5.0
        for col_idx in range(count):
            if idx >= N:
                break
            offset = 0.5 / 6.0 if row_idx % 2 == 1 else 0.0
            pos[idx, 0] = (col_idx + 0.5 + offset) / 6.0
            pos[idx, 1] = row_y
            r[idx] = 0.35 / 6.0
            idx += 1
    return pos, r

def _initial_config_perturbed_grid(rng, side=5, noise=0.02):
    """Perturbed grid initial configuration."""
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.normal(0, noise)
        pos[i, 1] = (row + 0.5) / side + rng.normal(0, noise)
        r[i] = 0.34 / side
    pos = np.clip(pos, 0.05, 0.95)
    return pos, r

def _initial_config_random(rng):
    """Random initial configuration."""
    pos = rng.random((N, 2)) * 0.8 + 0.1
    r = np.full(N, 0.02)
    return pos, r

def _simulated_annealing(pos, r, t_start, deadline, seed=42):
    """Simulated annealing to escape local minima."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    rng = np.random.default_rng(seed)
    T = 0.03
    T_min = 1e-6
    cooling = 0.9995
    
    current_obj, _ = _objective_grad(pos, r, beta=50.0)
    
    while _elapsed(t_start) < deadline and T > T_min:
        if rng.random() < 0.6:
            i = rng.integers(0, n)
            pos[i, 0] += rng.normal(0, T * 0.4)
            pos[i, 1] += rng.normal(0, T * 0.4)
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
        else:
            i = rng.integers(0, n)
            r[i] = np.clip(r[i] + rng.normal(0, T * 0.25), 0.0, 0.4)
        
        new_obj, _ = _objective_grad(pos, r, beta=50.0)
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_obj = new_obj
            if _validate(pos, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_r = r.copy()
        
        T *= cooling
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    best_score = -1.0
    best_result = None
    
    num_restarts = 4
    
    for restart in range(num_restarts):
        if _elapsed(t_start) > DEADLINE - 25.0:
            break
        
        rng = np.random.default_rng(3000 + restart)
        
        if restart == 0:
            pos, r = _initial_config_grid(rng)
        elif restart == 1:
            pos, r = _initial_config_hex(rng)
        elif restart == 2:
            pos, r = _initial_config_perturbed_grid(rng, noise=0.025)
        else:
            pos, r = _initial_config_random(rng)
        
        # Ensure initial validity
        r = _enforce_validity(pos, r)
        
        # Phase 1: Physics-based relaxation
        pos, r = _physics_relax(pos, r, t_start, DEADLINE - 40.0, beta=60.0)
        
        # Phase 2: Simulated annealing to escape local minima
        if _elapsed(t_start) < DEADLINE - 25.0:
            pos, r = _simulated_annealing(pos, r, t_start, DEADLINE - 25.0, seed=100+restart)
        
        # Phase 3: Polish with SLSQP
        if _elapsed(t_start) < DEADLINE - 10.0:
            pos, r = _polish_analytic(pos, r, t_start, DEADLINE - 10.0, beta=80.0)
        
        # Phase 4: Final physics polish
        if _elapsed(t_start) < DEADLINE - 3.0:
            pos, r = _physics_relax(pos, r, t_start, DEADLINE - 3.0, beta=100.0)
        
        score = np.sum(r)
        if _validate(pos, r) and score > best_score:
            best_score = score
            best_result = np.column_stack([pos, r])
        
        if best_score > 3.0:
            break
    
    if best_result is None:
        pos, r = _initial_config_grid(np.random.default_rng(0))
        r = _enforce_validity(pos, r)
        best_result = np.column_stack([pos, r])
    
    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
