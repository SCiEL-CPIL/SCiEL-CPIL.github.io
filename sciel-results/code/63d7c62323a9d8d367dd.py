"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Bisection on the target overlap T, solving a quadratic feasibility
problem at each step using SLSQP. The feasibility check asks whether there
exists h in [0,1]^N with sum(h)=N/2 such that for all k, the linear
autocorrelation r_k >= N/2 - N*T/2.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _feasibility_objective(h, target_r):
    """
    Objective for SLSQP: maximize min(r_k - target_r) subject to constraints.
    We minimize the negative: -min(r_k - target_r).
    Actually, we want all r_k >= target_r. Use a penalty: sum of violations.
    """
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    violations = np.maximum(0, target_r - r)
    return np.sum(violations) ** 2


def _check_feasible(target_r, h0):
    """
    Check if there exists h in [0,1]^N, sum(h)=N/2, with all r_k >= target_r.
    Returns (feasible, h) where h is a feasible point if found.
    """
    n = N
    target_sum = n / 2.0
    
    # Objective: minimize sum of squared violations of r_k >= target_r
    # Constraints: sum(h) = target_sum, 0 <= h_i <= 1
    
    def obj(h):
        r = np.zeros(n)
        for k in range(n):
            r[k] = np.dot(h[k:], h[:n-k])
        viol = np.maximum(0, target_r - r)
        return np.sum(viol ** 2)
    
    def grad_obj(h):
        r = np.zeros(n)
        for k in range(n):
            r[k] = np.dot(h[k:], h[:n-k])
        viol = np.maximum(0, target_r - r)
        grad = np.zeros(n)
        for k in range(n):
            if viol[k] > 0:
                # d/dh_j of r_k = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)
                if k > 0:
                    grad[k:] += 2 * viol[k] * h[:n-k]
                grad[:n-k] += 2 * viol[k] * h[k:]
        return grad
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum,
         'jac': lambda h: np.ones(n)}
    ]
    bounds = [(0.0, 1.0)] * n
    
    result = minimize(obj, h0, method='SLSQP', jac=grad_obj,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-12})
    
    if result.success:
        # Verify feasibility
        r = np.zeros(n)
        for k in range(n):
            r[k] = np.dot(result.x[k:], result.x[:n-k])
        if np.all(r >= target_r - 1e-9):
            # Ensure sum constraint
            diff = target_sum - np.sum(result.x)
            if abs(diff) > 1e-10:
                mask = (result.x > 1e-8) & (result.x < 1 - 1e-8)
                if np.any(mask):
                    result.x[mask] += diff / np.sum(mask)
                    result.x = np.clip(result.x, 0.0, 1.0)
            return True, result.x
    return False, None





def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses bisection on the target overlap T, with SLSQP feasibility checks.
    """
    t0 = time.time()
    np.random.seed(42)
    
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_score = _score(best_h)
    
    # Bisection bounds for T (the max overlap)
    # Lower bound: best_score (we can achieve at least this)
    # Upper bound: 1.0 (trivial)
    lo_T = 0.0
    hi_T = 1.0
    
    # First, get a better initial score from a few structured candidates
    i = np.arange(N)
    candidates = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.cos(2.0 * np.pi * i / N),
        0.5 + 0.3 * np.cos(4.0 * np.pi * i / N),
        0.5 + 0.3 * np.sin(2.0 * np.pi * i / N),
    ]
    for h in candidates:
        h = np.clip(h, 0.0, 1.0)
        # Adjust sum
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    lo_T = best_score * 0.5  # We know we can do at least this well
    
    # Bisection: for each candidate T, check if feasible
    # r_k >= N/2 - N*T/2
    for iteration in range(20):
        if time.time() > _DEADLINE - 10:
            break
        
        mid_T = (lo_T + hi_T) / 2.0
        target_r = N / 2.0 - N * mid_T / 2.0
        
        feasible, h_feas = _check_feasible(target_r, best_h)
        
        if feasible:
            # We can achieve overlap <= mid_T
            hi_T = mid_T
            if h_feas is not None:
                s = _score(h_feas)
                if s < best_score:
                    best_score = s
                    best_h = h_feas.copy()
        else:
            # Cannot achieve overlap <= mid_T
            lo_T = mid_T
    
    # Final refinement: use the best_h found
    # Do a few more SLSQP iterations to polish
    if time.time() < _DEADLINE - 5:
        # Try to improve by directly minimizing max overlap
        def _max_overlap_obj(h):
            cs = _compute_overlaps(h)
            return np.max(cs)
        
        def _max_overlap_grad(h):
            cs = _compute_overlaps(h)
            k_max = np.argmax(cs)
            r_k = np.dot(h[k_max:], h[:N-k_max])
            grad = np.zeros(N)
            if k_max > 0:
                grad[k_max:] = h[:N-k_max]
            grad[:N-k_max] += h[k_max:]
            return (2.0 / N) * grad
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0,
             'jac': lambda h: np.ones(N)}
        ]
        bounds = [(0.0, 1.0)] * N
        
        result = minimize(_max_overlap_obj, best_h, method='SLSQP',
                         jac=_max_overlap_grad, bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 300, 'ftol': 1e-14})
        
        if result.success:
            s = _score(result.x)
            if s < best_score:
                best_score = s
                best_h = result.x.copy()
    
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-10) & (best_h < 1 - 1e-10)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
