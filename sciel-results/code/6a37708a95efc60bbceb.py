"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple N values, pick the best
    best_h = None
    best_score = float('inf')
    
    for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if time.time() > deadline:
            break
        
        # Build LP:
        # Variables: h_0, ..., h_{N-1}, t
        # Minimize: t
        # Constraints:
        #   0 <= h_i <= 1
        #   sum h_i = N/2
        #   For each shift s in [0,2], overlap(s) <= t
        
        # Overlap for shift s:
        # If s in [0,1]: overlap = sum_{i: x_i < 1-s} h_i + sum_{i: x_i >= 1+s} (1 - h_i)
        #   where x_i is the center or left edge of step i
        # For equal-width steps on [0,2], step i covers [2i/N, 2(i+1)/N)
        # A step is fully in [0, 1-s] if 2(i+1)/N <= 1-s, i.e., i <= N(1-s)/2 - 1
        # A step is fully in [1+s, 2] if 2i/N >= 1+s, i.e., i >= N(1+s)/2
        
        # For simplicity, use step centers: x_i = 2(i+0.5)/N
        # Step i is in [0, 1-s] if x_i < 1-s (approximation)
        # Step i is in [1+s, 2] if x_i >= 1+s
        
        # Let's use exact integration:
        # overlap(s) = ∫₀^{1-s} h(x)dx + ∫_{1+s}² (1-h(x))dx for s in [0,1]
        # For step function, ∫₀^{1-s} h(x)dx = sum over steps fully/partially in [0,1-s]
        
        # For equal-width steps, this is piecewise linear in s.
        # The maximum over s occurs at s = 0, s = 1, or at breakpoints.
        # Breakpoints occur when 1-s = 2k/N or 1+s = 2k/N, i.e., s = 1 - 2k/N or s = 2k/N - 1
        
        # Candidate s values: s = 0, s = 1, s = 2, and s = 2k/N for k=0..N, s = 1 - 2k/N for k=0..N
        # Filter to [0,2]
        
        s_candidates = set()
        s_candidates.add(0.0)
        s_candidates.add(1.0)
        s_candidates.add(2.0)
        for k in range(N + 1):
            s1 = 2.0 * k / N
            s2 = 1.0 - 2.0 * k / N
            if 0.0 <= s1 <= 2.0:
                s_candidates.add(s1)
            if 0.0 <= s2 <= 2.0:
                s_candidates.add(s2)
        
        s_vals = sorted(s_candidates)
        
        # Build constraint matrix A_ub * x <= b_ub
        # x = [h_0, ..., h_{N-1}, t]
        # For each s: overlap(s) - t <= 0
        
        # Compute overlap(s) as linear function of h:
        # overlap(s) = sum_i c_i(s) * h_i + d(s)
        # where c_i(s) = 1 if step i is in [0,1-s], -1 if step i is in [1+s,2] (for s in [0,1])
        # and d(s) = measure of [1+s, 2] (since 1-h contributes -h + 1)
        
        # More carefully:
        # For s in [0,1]:
        #   overlap(s) = ∫₀^{1-s} h(x)dx + ∫_{1+s}² (1-h(x))dx
        #              = ∫₀^{1-s} h(x)dx + (1 - s) - ∫_{1+s}² h(x)dx
        #              = sum_i [w_i(s) * h_i] + (1-s)
        #   where w_i(s) = measure of overlap of step i with [0,1-s] - measure of overlap of step i with [1+s,2]
        
        # For s in [1,2]:
        #   overlap(s) = ∫₀^{s-1} (1-h(x))dx + ∫_{2-s}² h(x)dx
        #              = (s-1) - ∫₀^{s-1} h(x)dx + ∫_{2-s}² h(x)dx
        #              = sum_i [w_i(s) * h_i] + (s-1)
        #   where w_i(s) = -measure of overlap of step i with [0,s-1] + measure of overlap of step i with [2-s,2]
        
        # For equal-width steps, compute w_i(s) exactly
        
        num_constraints = len(s_vals)
        A_ub = np.zeros((num_constraints, N + 1))
        b_ub = np.zeros(num_constraints)
        
        step_width = 2.0 / N
        
        for idx, s in enumerate(s_vals):
            d = 0.0
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                
                if s <= 1.0:
                    # overlap(s) = ∫₀^{1-s} h dx + ∫_{1+s}² (1-h) dx
                    # w_i = measure(step ∩ [0,1-s]) - measure(step ∩ [1+s,2])
                    lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                    lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                    m1 = max(0.0, hi1 - lo1)
                    m2 = max(0.0, hi2 - lo2)
                    w_i = m1 - m2
                    A_ub[idx, i] = w_i
                    d += m2  # from (1-h) integral: contributes +1 per unit measure
                else:
                    # overlap(s) = ∫₀^{s-1} (1-h) dx + ∫_{2-s}² h dx
                    # w_i = -measure(step ∩ [0,s-1]) + measure(step ∩ [2-s,2])
                    lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                    lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                    m1 = max(0.0, hi1 - lo1)
                    m2 = max(0.0, hi2 - lo2)
                    w_i = -m1 + m2
                    A_ub[idx, i] = w_i
                    d += m1  # from (1-h) integral: contributes +1 per unit measure
            
            # Constraint: sum_i w_i * h_i + d - t <= 0
            # i.e., sum_i w_i * h_i - t <= -d
            A_ub[idx, N] = -1.0
            b_ub[idx] = -d
        
        # Equality constraint: sum h_i = N/2
        A_eq = np.zeros((1, N + 1))
        A_eq[0, :N] = 1.0
        b_eq = np.array([N / 2.0])
        
        # Bounds: 0 <= h_i <= 1, t >= 0
        bounds = [(0, 1)] * N + [(0, None)]
        
        # Objective: minimize t
        c = np.zeros(N + 1)
        c[N] = 1.0
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
            if result.success:
                t_val = result.x[N]
                h_val = result.x[:N]
                # Clamp to [0,1] and ensure sum is exactly N/2
                h_val = np.clip(h_val, 0.0, 1.0)
                # Adjust to ensure exact sum
                diff = N / 2.0 - np.sum(h_val)
                if abs(diff) > 1e-12:
                    # Distribute adjustment
                    h_val += diff / N
                    h_val = np.clip(h_val, 0.0, 1.0)
                
                if t_val < best_score:
                    best_score = t_val
                    best_h = h_val
        except Exception:
            pass
    
    if best_h is not None:
        return best_h
    else:
        # Fallback to constant
        return np.full(100, 0.5)

# EVOLVE-BLOCK-END
