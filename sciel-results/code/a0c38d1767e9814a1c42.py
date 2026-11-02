"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _max_overlap(h):
    """Compute maximum overlap of h with shifted 1-h over all shifts."""
    n = len(h)
    dw = 2.0 / n
    max_ov = 0.0
    for k in range(n):
        valid = n - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_ov:
            max_ov = overlap
    return max_ov

def _solve_lp_for_n(n):
    """Solve LP for given n to minimize max overlap."""
    # Candidate shift values where overlap function has breakpoints
    s_candidates = set()
    s_candidates.add(0.0)
    s_candidates.add(1.0)
    s_candidates.add(2.0)
    for k in range(n + 1):
        s1 = 2.0 * k / n
        s2 = 1.0 - 2.0 * k / n
        if 0.0 <= s1 <= 2.0:
            s_candidates.add(s1)
        if 0.0 <= s2 <= 2.0:
            s_candidates.add(s2)
    
    s_vals = sorted(s_candidates)
    num_constraints = len(s_vals)
    
    A_ub = np.zeros((num_constraints, n + 1))
    b_ub = np.zeros(num_constraints)
    
    step_width = 2.0 / n
    
    for idx, s in enumerate(s_vals):
        d = 0.0
        for i in range(n):
            step_start = 2.0 * i / n
            step_end = 2.0 * (i + 1) / n
            
            if s <= 1.0 + 1e-12:
                # overlap(s) = ∫₀^{1-s} h dx + ∫_{1+s}² (1-h) dx
                lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = m1 - m2
                A_ub[idx, i] = w_i
                d += m2
            else:
                # overlap(s) = ∫₀^{s-1} (1-h) dx + ∫_{2-s}² h dx
                lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = -m1 + m2
                A_ub[idx, i] = w_i
                d += m1
            
            # Constraint: sum_i w_i * h_i + d - t <= 0
        A_ub[idx, n] = -1.0
        b_ub[idx] = -d
    
    # Equality constraint: sum h_i = n/2
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([n / 2.0])
    
    # Bounds: 0 <= h_i <= 1, t >= 0
    bounds = [(0, 1)] * n + [(0, None)]
    
    # Objective: minimize t
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        if result.success:
            t_val = result.x[n]
            h_val = result.x[:n]
            h_val = np.clip(h_val, 0.0, 1.0)
            # Adjust to ensure exact sum
            diff = n / 2.0 - np.sum(h_val)
            if abs(diff) > 1e-12:
                h_val += diff / n
                h_val = np.clip(h_val, 0.0, 1.0)
                # Re-correct if needed
                diff2 = n / 2.0 - np.sum(h_val)
                if abs(diff2) > 1e-12:
                    idx = np.argmin(np.abs(h_val - 0.5))
                    h_val[idx] += diff2
                    h_val[idx] = np.clip(h_val[idx], 0.0, 1.0)
            return h_val, t_val
    except Exception:
        pass
    return None, float('inf')

def _anti_periodic_optimize(n):
    """Fallback: optimize anti-periodic structure."""
    half = n // 2
    h_first_half = np.full(half, 0.5)
    
    def build_full_h(first_half):
        h = np.empty(n)
        h[:half] = first_half
        h[half:] = 1.0 - first_half
        return h
    
    def objective(x):
        h = build_full_h(x)
        return _max_overlap(h)
    
    bounds = [(0.0, 1.0)] * half
    best_x = h_first_half.copy()
    best_score = objective(best_x)
    
    try:
        result = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
        if result.fun < best_score:
            best_x = result.x
            best_score = result.fun
    except Exception:
        pass
    
    h_final = build_full_h(best_x)
    current_sum = np.sum(h_final)
    if abs(current_sum - n/2.0) > 1e-12:
        diff = n/2.0 - current_sum
        h_final = h_final + diff / n
        h_final = np.clip(h_final, 0.0, 1.0)
        current_sum = np.sum(h_final)
        if abs(current_sum - n/2.0) > 1e-12:
            diff = n/2.0 - current_sum
            idx = np.argmin(np.abs(h_final - 0.5))
            h_final[idx] += diff
            h_final[idx] = np.clip(h_final[idx], 0.0, 1.0)
    
    return h_final, _max_overlap(h_final)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Try multiple N values with LP
    for n in [16, 32, 64, 100, 128, 256]:
        if time.time() > deadline - 10.0:
            break
        
        h_val, t_val = _solve_lp_for_n(n)
        if h_val is not None:
            if t_val < best_score:
                best_score = t_val
                best_h = h_val
    
    # If LP didn't give good results, try anti-periodic optimization
    if best_score > 0.5 or best_h is None:
        if time.time() > deadline - 20.0:
            pass
        else:
            h_ap, score_ap = _anti_periodic_optimize(N)
            if score_ap < best_score:
                best_score = score_ap
                best_h = h_ap
    
    # Final fallback
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure the returned array has length N as expected
    if len(best_h) != N:
        # Resample to length N
        if len(best_h) > N:
            # Take first N elements
            best_h = best_h[:N]
        else:
            # Pad with 0.5
            pad = N - len(best_h)
            best_h = np.concatenate([best_h, np.full(pad, 0.5)])
    
    # Final sum correction
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-12:
        diff = N/2.0 - current_sum
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        current_sum = np.sum(best_h)
        if abs(current_sum - N/2.0) > 1e-12:
            diff = N/2.0 - current_sum
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h[idx] = np.clip(best_h[idx], 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
