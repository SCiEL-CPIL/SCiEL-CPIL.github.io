"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Parse inputs
    # distribution_spec: likely a list of 16 probabilities or a dict mapping report->prob
    # support: list of 16 tuples (v1, v2)
    # constraints: dict with various constraint specs
    
    # Default support: all 16 combinations
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        # Convert support items to tuples to ensure they are hashable and consistent
        support = [tuple(s) for s in support]
    
    n_reports = len(support)
    if n_reports != 16:
        # Try to handle if support is given differently
        # Assume it's a list of 16 entries
        if n_reports == 0:
            support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
            n_reports = 16
        else:
            # If length is not 16, pad or truncate to 16? 
            # The task says "Every allowed report is enumerated" and "16 possible reports".
            # We assume the input is well-formed but defensive coding helps.
            # For now, if it's not 16, we try to map to the standard 16 grid if possible, 
            # otherwise we just use what we have but warn (can't warn in this context, so just proceed).
            # However, the return shape must be (16,2) and (16,). 
            # If support is wrong size, the LP will be wrong size. 
            # We force n_reports to 16 by defaulting if mismatch.
            support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
            n_reports = 16
    
    # Extract probabilities from distribution_spec
    probs = np.ones(n_reports) / n_reports  # uniform default
    if distribution_spec is not None:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            dist_arr = np.array(distribution_spec, dtype=float).flatten()
            if len(dist_arr) == n_reports:
                probs = dist_arr
                # Normalize
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(n_reports) / n_reports
            else:
                # If length doesn't match, stick with uniform
                probs = np.ones(n_reports) / n_reports
        elif isinstance(distribution_spec, dict):
            # Map report indices or tuples to probabilities
            # We need to safely check keys. 
            # Keys might be tuples (v1, v2) or indices.
            for i, rep in enumerate(support):
                # Check if tuple key exists
                if rep in distribution_spec:
                    probs[i] = distribution_spec[rep]
                # Check if index key exists
                elif i in distribution_spec:
                    probs[i] = distribution_spec[i]
                # Check if string key exists? Unlikely, skip.
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(n_reports) / n_reports
    
    # Ensure probs are valid
    probs = np.clip(probs, 0, 1)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_reports) / n_reports
    
    # Extract values for each report
    # support[i] = (v1_i, v2_i)
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        # rep is now a tuple of length 2
        values[i, 0] = rep[0]
        values[i, 1] = rep[1]
    
    # For a single bidder with IC (incentive compatibility), the allocation rule x(t)
    # and payment rule p(t) must satisfy:
    # 1. Individual Rationality (IR): For each type t, u(t,t) >= 0 where u(t,t) = v(t)*x(t) - p(t)
    #    Since the bidder always reports their true type (truthfulness), we need v(t)*x(t) - p(t) >= 0
    # 2. Incentive Compatibility (IC): For each pair of types (t, t'), 
    #    v(t)*x(t) - p(t) >= v(t)*x(t') - p(t')
    #    This means: the utility from reporting t when true type is t is >= utility from reporting t'
    #
    # For a single bidder, the standard result is that IC + IR implies:
    # - The allocation x(t) must be monotone in the value (in some sense)
    # - The payment is determined by the allocation via the envelope theorem
    #
    # Actually, for a single bidder with multi-dimensional types, the problem is more complex.
    # Let's use the LP formulation.
    #
    # Variables:
    # - x[i][j] for i in 0..15, j in 0..1: allocation of good j to bidder with report i (16*2 = 32 vars)
    # - p[i] for i in 0..15: payment for report i (16 vars)
    # Total: 48 variables
    #
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i][j] <= 1 for all i, j
    # 2. IR: For each i, values[i] @ x[i] - p[i] >= 0  =>  p[i] <= values[i] @ x[i]
    # 3. IC: For each pair (i, i'), values[i] @ x[i] - p[i] >= values[i] @ x[i'] - p[i']
    #    => values[i] @ (x[i] - x[i']) + p[i'] - p[i] >= 0
    #    => p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    #    This gives 16*16 = 256 constraints (but we can reduce to 16*15/2 = 120 if we consider both directions)
    #    Actually, IC requires for all i, i': u(i,i) >= u(i,i') where u(i,i') = values[i] @ x[i'] - p[i']
    #    So: values[i] @ x[i] - p[i] >= values[i] @ x[i'] - p[i']
    #    => values[i] @ (x[i] - x[i']) >= p[i] - p[i']
    #    This is 16*16 = 256 constraints, but many are redundant.
    
    # Let's set up the LP
    # Variables: [x[0][0], x[0][1], x[1][0], x[1][1], ..., x[15][0], x[15][1], p[0], p[1], ..., p[15]]
    # Index: for i in 0..15, x[i][0] is at index 2*i, x[i][1] is at index 2*i+1
    #        p[i] is at index 32 + i
    
    n_vars = 2 * n_reports + n_reports  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog, we minimize, so negate
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[32 + i] = -probs[i]  # negative because we're maximizing
    
    # Add a small epsilon to the objective to encourage non-zero allocations/payments if degenerate?
    # No, standard LP should handle it.
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # 1. Feasibility: 0 <= x[i][j] <= 1
    # x[i][j] <= 1: already handled by bounds
    # x[i][j] >= 0: handled by bounds
    
    # 2. IR: p[i] <= values[i] @ x[i]  =>  -values[i][0]*x[i][0] - values[i][1]*x[i][1] + p[i] <= 0
    #    For each i: -values[i][0]*x[i][0] - values[i][1]*x[i][1] + p[i] <= 0
    
    # 3. IC: For each pair (i, i'): p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    #    => values[i][0]*x[i][0] + values[i][1]*x[i][1] - values[i][0]*x[i'][0] - values[i][1]*x[i'][1] - p[i] + p[i'] >= 0
    #    => -values[i][0]*x[i][0] - values[i][1]*x[i][1] + values[i][0]*x[i'][0] + values[i][1]*x[i'][1] + p[i] - p[i'] <= 0
    
    # Let's build the constraint matrix
    # Start with IR constraints (16 constraints)
    A_rows = []
    b_rows = []
    
    # IR: for each i
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[2*i] = -values[i][0]
        row[2*i+1] = -values[i][1]
        row[32+i] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    # IC: for each pair (i, i') where i != i'
    # p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    # => -values[i] @ x[i] + values[i] @ x[i'] + p[i] - p[i'] <= 0
    for i in range(n_reports):
        for ip in range(n_reports):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            # -values[i] @ x[i]
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            # +values[i] @ x[ip]
            row[2*ip] = values[i][0]
            row[2*ip+1] = values[i][1]
            # +p[i] - p[ip]
            row[32+i] = 1
            row[32+ip] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[i][j] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x[i][0]
        bounds.append((0, 1))  # x[i][1]
    for i in range(n_reports):
        bounds.append((0, None))  # p[i] >= 0
    
    # Solve LP
    try:
        # Use method='highs' which is robust and fast
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:
            # Extract solution
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[32+i]
            
            # Clip to ensure feasibility due to floating point errors
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Post-processing: Ensure strict IR and IC
            # 1. Enforce IR: p[i] <= values[i] @ x[i]
            for i in range(n_reports):
                max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
                if p[i] > max_pay:
                    p[i] = max_pay
            
            # 2. Enforce IC: 
            # For single bidder, IC is equivalent to:
            # u(i) = v_i . x_i - p_i
            # u(i) >= u(i') for all i, i' is NOT the correct condition.
            # The condition is: u(i, i) >= u(i, i') for all i, i'
            # u(i, i) = v_i . x_i - p_i
            # u(i, i') = v_i . x_i' - p_i'
            # So: v_i . x_i - p_i >= v_i . x_i' - p_i'
            # => p_i - p_i' <= v_i . (x_i - x_i')
            
            # Since we solved the LP, this should hold. 
            # However, numerical errors might cause violations.
            # We can fix by adjusting payments. 
            # A common way to fix IC violations in single-bidder settings is to use the 
            # "envelope" formula or simply re-project.
            # Given the small size (16 types), we can check and fix.
            
            # Actually, if we just clip p[i] to be <= v_i.x_i, we might break IC.
            # Let's verify IC and if violated, fallback to a known IC mechanism.
            # A simple IC mechanism is the "All-or-Nothing" or "Uniform" allocation.
            # But we want to keep the LP result if possible.
            
            ic_violated = False
            for i in range(n_reports):
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    # Utility of reporting i when type is i
                    u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                    # Utility of reporting ip when type is i
                    u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if u_ii < u_iip - 1e-7:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if not ic_violated:
                return {
                    'allocations': x,
                    'payments': p
                }
            else:
                # Fallback to a safe IC mechanism: 
                # Allocate goods probabilistically or deterministically?
                # The task allows fractional allocations (probabilities).
                # A safe mechanism: x[i] = 1 for all i (allocate all), p[i] = max_j (v_j . x_j) ? No.
                # Let's use the "Vickrey-Clarke-Groves" like logic for single bidder?
                # For single bidder, a dominant strategy mechanism requires monotonicity.
                # Simplest valid mechanism: x[i] = 1 (allocate both goods always).
                # Then p[i] must be such that v_i . 1 - p_i >= v_i . 1 - p_j => p_j >= p_i for all j?
                # If x is constant, IC implies p_i is constant.
                # IR implies p_i <= v_i . 1.
                # To max revenue, set p_i = min_j (v_j . 1).
                # This is valid.
                
                x_safe = np.ones((n_reports, 2))
                min_val = min(values[i][0] + values[i][1] for i in range(n_reports))
                p_safe = np.full(n_reports, min_val)
                
                # Check IR for safe
                for i in range(n_reports):
                    if p_safe[i] > values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]:
                         p_safe[i] = values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]
                
                return {
                    'allocations': x_safe,
                    'payments': p_safe
                }
        else:
            # LP failed, return a safe default
            x_safe = np.ones((n_reports, 2))
            min_val = min(values[i][0] + values[i][1] for i in range(n_reports))
            p_safe = np.full(n_reports, min_val)
            
            # Ensure IR
            for i in range(n_reports):
                if p_safe[i] > values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]:
                     p_safe[i] = values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]
                     
            return {
                'allocations': x_safe,
                'payments': p_safe
            }
    
    except Exception as e:
        # Fallback: simple truthful mechanism
        # Allocate both goods, charge the minimum possible value to ensure IR and IC
        x_safe = np.ones((n_reports, 2))
        min_val = min(values[i][0] + values[i][1] for i in range(n_reports))
        p_safe = np.full(n_reports, min_val)
        
        # Ensure IR
        for i in range(n_reports):
            if p_safe[i] > values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]:
                 p_safe[i] = values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]
                 
        return {
            'allocations': x_safe,
            'payments': p_safe
        }
# EVOLVE-BLOCK-END
