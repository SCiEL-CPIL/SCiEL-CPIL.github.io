"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 120.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _find_2block_init():
    """
    Find a good initial guess using a 2-block structure.
    h = [a]*k + [b]*(N-k)
    Constraint: k*a + (N-k)*b = N/2
    Overlap max is max(a*(1-a), a*(1-b), b*(1-b), b)
    """
    best_score = float('inf')
    best_h = None
    
    for k in range(1, N):
        # b = (N/2 - k*a) / (N-k)
        # b in [0,1] => 0 <= N/2 - k*a <= N-k
        # k*a <= N/2 => a <= N/(2k)
        # N/2 - k*a <= N-k => k*a >= N/2 - N + k => a >= (N/2 - N + k)/k
        
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
            
        # Sample a few points
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, 50)
        for a in a_vals:
            b = (N / 2.0 - k * a) / (N - k)
            if b < -1e-6 or b > 1.0 + 1e-6:
                continue
            b = np.clip(b, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a
            h[k:] = b
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                
    return best_h, best_score


def _slsqp_optimize(h0, M0, deadline):
    """
    Optimize using SLSQP with epigraph formulation.
    Minimize M
    Subject to:
    h_i(1-h_{i+1}) <= M for i=1..N-1
    h_N <= M
    sum(h) = N/2
    0 <= h_i <= 1
    """
    n = N
    # Variables: h_1..h_N, M
    x0 = np.concatenate([h0, [M0]])
    
    def objective(x):
        return x[-1] # Minimize M
        
    def gradient(x):
        grad = np.zeros(n + 1)
        grad[-1] = 1.0
        return grad
        
    constraints = []
    
    # Inequality constraints: h_i(1-h_{i+1}) - M <= 0
    # and h_N - M <= 0
    # We can define them as a single function returning an array
    def ineq_constraints(x):
        h = x[:n]
        M = x[-1]
        c = np.zeros(n)
        for i in range(n - 1):
            c[i] = h[i] * (1.0 - h[i+1]) - M
        c[n-1] = h[n-1] - M
        return c
        
    def ineq_jacobian(x):
        h = x[:n]
        M = x[-1]
        # Jacobian is (n x (n+1))
        J = np.zeros((n, n + 1))
        for i in range(n - 1):
            # c_i = h_i - h_i h_{i+1} - M
            # d c_i / d h_i = 1 - h_{i+1}
            # d c_i / d h_{i+1} = -h_i
            # d c_i / d M = -1
            J[i, i] = 1.0 - h[i+1]
            J[i, i+1] = -h[i]
            J[i, -1] = -1.0
        # c_{n-1} = h_{n-1} - M
        J[n-1, n-1] = 1.0
        J[n-1, -1] = -1.0
        return J
        
    constraints.append({
        'type': 'ineq',
        'fun': lambda x: -ineq_constraints(x), # scipy wants f(x) >= 0 for 'ineq'
        'jac': lambda x: -ineq_jacobian(x)
    })
    
    # Equality constraint: sum(h) - N/2 = 0
    def eq_constraint(x):
        return np.sum(x[:n]) - TARGET_SUM
        
    def eq_jacobian(x):
        J = np.zeros(n + 1)
        J[:n] = 1.0
        return J
        
    constraints.append({
        'type': 'eq',
        'fun': eq_constraint,
        'jac': eq_jacobian
    })
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)]
    
    start_time = time.time()
    try:
        res = minimize(objective, x0, jac=gradient, method='SLSQP', 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
        if res.success:
            h_opt = res.x[:n]
            M_opt = res.x[-1]
            # Verify
            h_opt = _fix_sum(h_opt)
            score = _score(h_opt)
            return h_opt, score
        else:
            return None, float('inf')
    except Exception:
        return None, float('inf')


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    best_h = None
    best_score = float('inf')

    # Phase 1: 2-block search
    h_init, s_init = _find_2block_init()
    if h_init is not None:
        best_h = h_init
        best_score = s_init

    # Phase 2: SLSQP optimization from 2-block init
    if best_h is not None and time.time() < deadline - 20.0:
        M_init = best_score
        h_opt, s_opt = _slsqp_optimize(best_h, M_init, deadline)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt

    # Phase 3: Try SLSQP from uniform init if time permits
    if time.time() < deadline - 10.0:
        h_uniform = np.full(N, 0.5)
        M_uniform = _score(h_uniform)
        h_opt, s_opt = _slsqp_optimize(h_uniform, M_uniform, deadline)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt

    if best_h is None:
        # Fallback
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)

    return best_h
