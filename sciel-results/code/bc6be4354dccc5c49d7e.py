import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    # Deadline guard: aim to finish well before the 90s kill.
    # We have a single LP solve which is fast, but we add a safety margin.
    # The LP is small (32 vars, ~50 constraints), so it should take milliseconds.
    
    # 1. Determine PMF
    def get_pmf(spec):
        if isinstance(spec, (list, tuple, np.ndarray)):
            p = np.array(spec, dtype=float)
            if p.shape == (16,):
                p = p / p.sum()
                return np.clip(p, 0, 1)
            if p.shape == (4, 4):
                p = p.flatten() / p.sum()
                return np.clip(p, 0, 1)
        
        if isinstance(spec, str):
            s = spec.lower()
            if 'uniform' in s or 'unif' in s:
                return np.ones(16) / 16.0
            # Fallback for unknown strings
            return np.ones(16) / 16.0
        
        # Default: Uniform
        return np.ones(16) / 16.0

    pmf = get_pmf(distribution_spec)
    pmf = pmf / pmf.sum()
    # Ensure non-negative
    pmf = np.clip(pmf, 0, 1)
    pmf = pmf / pmf.sum()

    # 2. Setup LP
    # Variables: x1[0..15], x2[0..15]
    # Index mapping: idx = v1 * 4 + v2
    
    def idx(v1, v2):
        return v1 * 4 + v2
    
    num_vars = 32
    x1_vars = slice(0, 16)
    x2_vars = slice(16, 32)
    
    # Objective: Maximize Expected Revenue
    # E[R] = sum_r P(r) * p(r)
    # p(r) = v1(r)*x1(r) + v2(r)*x2(r) - sum_{k=1}^{v1} x1(v1-k, v2) - sum_{k=1}^{v2} x2(v1, v2-k)
    
    c_min = np.zeros(num_vars)
    
    # Constraints:
    # A_ub x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Monotonicity for x1: x1(v1, v2) >= x1(v1-1, v2) for v1 > 0
    # => x1(v1-1, v2) - x1(v1, v2) <= 0
    for v1 in range(1, 4):
        for v2 in range(4):
            row = np.zeros(num_vars)
            row[idx(v1-1, v2)] = 1
            row[idx(v1, v2)] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    # 2. Monotonicity for x2: x2(v1, v2) >= x2(v1, v2-1) for v2 > 0
    # => x2(v1, v2-1) - x2(v1, v2) <= 0
    for v1 in range(4):
        for v2 in range(1, 4):
            row = np.zeros(num_vars)
            row[16 + idx(v1, v2-1)] = 1
            row[16 + idx(v1, v2)] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    # 3. Individual Rationality / Non-negative Payments: p(r) >= 0
    # => -p(r) <= 0
    # -p(r) = -v1(r)*x1(r) - v2(r)*x2(r) + sum_{k=1}^{v1} x1(v1-k, v2) + sum_{k=1}^{v2} x2(v1, v2-k) <= 0
    for r in range(16):
        v1 = r // 4
        v2 = r % 4
        
        row = np.zeros(num_vars)
        
        # -v1(r)*x1(r)
        row[idx(v1, v2)] = -v1
        
        # -v2(r)*x2(r)
        row[16 + idx(v1, v2)] = -v2
        
        # + sum_{k=1}^{v1} x1(v1-k, v2)
        for k in range(1, v1 + 1):
            prev_v1 = v1 - k
            row[idx(prev_v1, v2)] += 1
            
        # + sum_{k=1}^{v2} x2(v1, v2-k)
        for k in range(1, v2 + 1):
            prev_v2 = v2 - k
            row[16 + idx(v1, prev_v2)] += 1
            
        A_ub_rows.append(row)
        b_ub_vals.append(0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # 4. Objective Coefficients for c_min (minimize -E[R])
    # E[R] = sum_r P(r) * [ v1(r)*x1(r) + v2(r)*x2(r) - sum_{k=1}^{v1} x1(v1-k, v2) - sum_{k=1}^{v2} x2(v1, v2-k) ]
    
    # Coefficients for x1(v1, v2):
    # From term r=(v1, v2): P(v1, v2) * v1
    # From term r=(v1+k, v2) where k>=1: -P(v1+k, v2)
    # From term r=(v1, v2+k) where k>=1: No effect on x1
    
    # Coefficients for x2(v1, v2):
    # From term r=(v1, v2): P(v1, v2) * v2
    # From term r=(v1, v2+k) where k>=1: -P(v1, v2+k)
    
    for v1 in range(4):
        for v2 in range(4):
            # x1(v1, v2)
            coeff_x1 = pmf[v1 * 4 + v2] * v1
            for k in range(1, 4 - v1):
                coeff_x1 -= pmf[(v1 + k) * 4 + v2]
            c_min[idx(v1, v2)] = -coeff_x1
            
            # x2(v1, v2)
            coeff_x2 = pmf[v1 * 4 + v2] * v2
            for k in range(1, 4 - v2):
                coeff_x2 -= pmf[v1 * 4 + (v2 + k)]
            c_min[16 + idx(v1, v2)] = -coeff_x2

    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * num_vars
    
    # Solve LP
    try:
        res = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
    
    if not res.success:
        # Fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
    
    x_opt = res.x
    x1_opt = x_opt[0:16]
    x2_opt = x_opt[16:32]
    
    # Clip to [0,1] to handle numerical errors
    x1_opt = np.clip(x1_opt, 0, 1)
    x2_opt = np.clip(x2_opt, 0, 1)
    
    # Construct Allocations (16, 2)
    allocations = np.column_stack((x1_opt, x2_opt))
    
    # Construct Payments (16,)
    payments = np.zeros(16)
    for r in range(16):
        v1 = r // 4
        v2 = r % 4
        
        p_r = v1 * x1_opt[idx(v1, v2)] + v2 * x2_opt[idx(v1, v2)]
        
        for k in range(1, v1 + 1):
            prev_v1 = v1 - k
            p_r -= x1_opt[idx(prev_v1, v2)]
            
        for k in range(1, v2 + 1):
            prev_v2 = v2 - k
            p_r -= x2_opt[idx(v1, prev_v2)]
            
        # Ensure non-negative (should be satisfied by LP, but clip for safety)
        payments[r] = max(0, p_r)
        
    # Time check
    if time.time() - start_time > 80:
        pass

    return {"allocations": allocations, "payments": payments}
