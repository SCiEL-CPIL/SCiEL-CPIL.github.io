import numpy as np
from scipy.optimize import minimize
import math
import time
import random

def _log_score(pts):
    """Compute log(Delta / n^n) where Delta is product of ordered-pair distances
    after scaling to diameter 2. Returns -inf on degenerate configs."""
    n = 8
    # Compute all pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dij = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            d[i, j] = dij
            d[j, i] = dij
    # Diameter
    D = np.max(d)
    if D < 1e-12:
        return -float('inf')
    # Check for any zero distances
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < 1e-12:
                return -float('inf')
    # Scale factor
    s = 2.0 / D
    # Log product of all ordered-pair distances after scaling
    # There are n*(n-1) = 56 ordered pairs
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                log_prod += math.log(d[i, j] * s)
    # Score = exp(log_prod) / n^n, so log(score) = log_prod - n*log(n)
    log_score_val = log_prod - n * math.log(n)
    return log_score_val

def _score(pts):
    """Return the actual score Delta/n^n."""
    ls = _log_score(pts)
    if ls == -float('inf'):
        return 0.0
    val = math.exp(ls)
    if val == float('inf'):
        return 1e100
    return val

def _regular_octagon():
    return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]

def _nelder_mead_optimize(pts, time_limit):
    """Optimize using Nelder-Mead on all 16 coordinates."""
    n = 8
    x0 = np.array([pts[i][0] for i in range(n)] + [pts[i][1] for i in range(n)])
    
    def objective(x):
        p = [[x[i], x[n + i]] for i in range(n)]
        ls = _log_score(p)
        if ls == -float('inf'):
            return 1e10
        return -ls
    
    start = time.time()
    maxiter = 3000
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9,
                                'adaptive': True})
        if time.time() - start < time_limit:
            p = [[res.x[i], res.x[n + i]] for i in range(n)]
            return p
    except Exception:
        pass
    return pts

def _simulated_annealing(pts, time_limit, seed):
    """Simulated annealing on all 16 coordinates."""
    random.seed(seed)
    n = 8
    current = [p[:] for p in pts]
    current_ls = _log_score(current)
    if current_ls == -float('inf'):
        return pts, -float('inf')
    
    best = [p[:] for p in current]
    best_ls = current_ls
    
    start = time.time()
    step_size = 0.05
    temp = 1.0
    cooling = 0.99995
    
    while time.time() - start < time_limit:
        idx = random.randint(0, n - 1)
        new_pts = [p[:] for p in current]
        new_pts[idx][0] += random.gauss(0, step_size)
        new_pts[idx][1] += random.gauss(0, step_size)
        
        new_ls = _log_score(new_pts)
        
        if new_ls == -float('inf'):
            step_size = max(step_size * 0.98, 0.001)
            temp *= cooling
            continue
        
        delta = new_ls - current_ls
        if delta > 0 or (temp > 1e-9 and random.random() < math.exp(delta / (temp * 0.01))):
            current = new_pts
            current_ls = new_ls
            if current_ls > best_ls:
                best_ls = current_ls
                best = [p[:] for p in current]
            step_size = min(step_size * 1.005, 0.3)
        else:
            step_size = max(step_size * 0.995, 0.001)
        
        temp *= cooling
    
    return best, best_ls

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Start with regular octagon
    best_pts = _regular_octagon()
    best_ls = _log_score(best_pts)
    
    # Phase 1: Nelder-Mead from regular octagon (fast, deterministic)
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5:
        t_limit = min(20.0, remaining * 0.3)
        nm_pts = _nelder_mead_optimize(best_pts, t_limit)
        nm_ls = _log_score(nm_pts)
        if nm_ls > best_ls:
            best_ls = nm_ls
            best_pts = nm_pts
    
    # Phase 2: Simulated annealing from best so far
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 3:
        t_limit = min(30.0, remaining * 0.4)
        sa_pts, sa_ls = _simulated_annealing(best_pts, t_limit, seed=42)
        if sa_ls > best_ls:
            best_ls = sa_ls
            best_pts = sa_pts
    
    # Phase 3: More restarts with random perturbations
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    restart_count = 0
    while remaining > 5:
        # Create perturbed initial guess
        perturbed = [p[:] for p in best_pts]
        for i in range(8):
            perturbed[i][0] += random.gauss(0, 0.15)
            perturbed[i][1] += random.gauss(0, 0.15)
        
        # Quick Nelder-Mead
        t_nm = min(8.0, remaining * 0.25)
        nm_pts = _nelder_mead_optimize(perturbed, t_nm)
        nm_ls = _log_score(nm_pts)
        
        # Quick SA
        remaining = deadline - time.time()
        if remaining > 3:
            t_sa = min(10.0, remaining * 0.4)
            sa_pts, sa_ls = _simulated_annealing(nm_pts, t_sa, seed=100 + restart_count)
            if sa_ls > nm_ls:
                nm_ls = sa_ls
                nm_pts = sa_pts
        
        if nm_ls > best_ls:
            best_ls = nm_ls
            best_pts = nm_pts
        
        restart_count += 1
        remaining = deadline - time.time()
        if restart_count >= 6:
            break
    
    # Final: ensure distinctness and coordinate bounds
    # Nudge any too-close points
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.hypot(best_pts[i][0] - best_pts[j][0], best_pts[i][1] - best_pts[j][1])
            if d < 1e-6:
                best_pts[i][0] += 1e-4 * random.random()
                best_pts[i][1] += 1e-4 * random.random()
    
    # Clamp coordinates
    for i in range(8):
        best_pts[i][0] = max(-999999.0, min(999999.0, best_pts[i][0]))
        best_pts[i][1] = max(-999999.0, min(999999.0, best_pts[i][1]))
    
    return best_pts

# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Log score: {_log_score(pts)}")
    for p in pts:
        print(f"  ({p[0]:.6f}, {p[1]:.6f})")
