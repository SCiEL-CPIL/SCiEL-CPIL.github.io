"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the support: 16 type profiles (4 values x 4 values for 2 goods)
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    # types[i] = (v1_i, v2_i), i = 0..15
    
    # Parse distribution_spec
    if isinstance(distribution_spec, dict):
        if 'type_probs' in distribution_spec:
            probs = np.array(distribution_spec['type_probs'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
        else:
            probs = None
            for k, v in distribution_spec.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == 16:
                    probs = np.array(v, dtype=float)
                    break
            if probs is None:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    # For a single additive bidder, the optimal IC mechanism decomposes by good.
    # For each good j, choose q_j(v) in [0,1] non-decreasing in v to maximize:
    #   sum_{v=0}^{3} q_j(v) * [v * P(v_j = v) - P(v_j > v)]
    # where P(v_j = v) is the marginal probability of value v for good j.
    #
    # The optimal q_j(v) is:
    #   q_j(v) = 1 if the "virtual value" coefficient is positive, 0 otherwise.
    # Since q_j must be non-decreasing, we need the optimal non-decreasing
    # solution. For a single good with 4 types, this is a small problem.
    #
    # Actually, for the single-bidder case with additive values, the optimal
    # allocation for each good is independent and the virtual value formula
    # gives the answer directly IF the resulting q is non-decreasing.
    # If not, we need to solve a small isotonic regression / pool-adjacent-violators.
    
    # Compute marginal probabilities for each good
    p1 = np.zeros(4)  # P(v1 = v)
    p2 = np.zeros(4)  # P(v2 = v)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    # For each good, compute the optimal non-decreasing allocation q(v)
    # using the virtual value decomposition.
    #
    # The objective for good j is:
    #   max sum_{v=0}^{3} q_j(v) * cv_j(v)
    #   where cv_j(v) = v * P_j(v) - P_j(v > v)
    #   subject to 0 <= q_j(0) <= q_j(1) <= q_j(2) <= q_j(3) <= 1
    #
    # This is a linear program with 4 variables and monotonicity constraints.
    # The optimal solution sets q_j(v) = 1 where the "pooled" virtual value
    # is positive, using the pool-adjacent-violators algorithm (PAVA).
    #
    # Actually, since the objective is linear and the feasible set is a
    # polymatroid-like structure, the optimal solution is:
    #   q_j(v) = 1 if the average virtual value from v to 3 is positive
    #   (after applying PAVA to ensure monotonicity)
    #
    # Let's use a simpler approach: solve the 4-variable LP exactly.
    # Variables: q0, q1, q2, q3 with 0 <= q0 <= q1 <= q2 <= q3 <= 1
    # Maximize: cv0*q0 + cv1*q1 + cv2*q2 + cv3*q3
    
    def solve_single_good(cv):
        """Solve max cv·q s.t. 0<=q0<=q1<=q2<=q3<=1 using PAVA-like logic."""
        # The optimal solution to this LP has the form:
        # q(v) = 1 for v >= threshold, q(v) = 0 for v < threshold
        # OR q(v) = alpha for some range (if there's a tie in virtual values)
        #
        # Using the dual / complementary slackness:
        # The optimal q is determined by the "upper envelope" of cumulative
        # virtual values. Specifically, define:
        #   S(v) = sum_{u=v}^{3} cv(u)  (suffix sum of virtual values)
        # The optimal q(v) = 1 if the running minimum of S(u)/ (4-u) from
        # u=v to 3 is positive... 
        #
        # Actually, let's just enumerate the 2^4 = 16 possible 0/1 patterns
        # that are non-decreasing (i.e., thresholds), plus the fractional cases.
        # Non-decreasing 0/1 patterns: q = [0,0,0,0], [0,0,0,1], [0,0,1,1], 
        # [0,1,1,1], [1,1,1,1] — 5 patterns.
        # But fractional solutions can also be optimal.
        #
        # The cleanest approach: use the fact that this is a separable convex
        # optimization over a chain. The optimal solution can be found by
        # checking all "breakpoints" — the solution is of the form:
        # q(v) = 0 for v < a, q(v) = alpha for a <= v < b, q(v) = 1 for v >= b
        # for some 0 <= a <= b <= 4 and 0 <= alpha <= 1.
        #
        # But actually, for a linear objective over the monotone cube, the
        # optimal solution is always at a vertex. The vertices of {q: 0<=q0<=...<=q3<=1}
        # are exactly the 5 threshold patterns above (plus the all-0 and all-1).
        # Wait, no — the vertices are more numerous. Let me think again.
        #
        # The feasible region is a polytope with vertices that are non-decreasing
        # 0/1 vectors. There are C(4+1,1) = 5 such vectors:
        # [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,1,1,1], [1,1,1,1]
        #
        # Wait, that's not right either. A non-decreasing binary vector of length 4
        # is determined by the threshold: q(v)=0 for v<t, q(v)=1 for v>=t.
        # t can be 0,1,2,3,4 giving 5 patterns. Yes, 5 vertices.
        #
        # So we just evaluate the objective at these 5 vertices and pick the best.
        # But wait — can a fractional solution be better? No, because the objective
        # is linear and the feasible set is a convex polytope, so the optimum is
        # at a vertex.
        
        # The 5 threshold patterns:
        # t=0: [1,1,1,1]
        # t=1: [0,1,1,1]
        # t=2: [0,0,1,1]
        # t=3: [0,0,0,1]
        # t=4: [0,0,0,0]
        
        best_val = -np.inf
        best_q = np.zeros(4)
        
        for t in range(5):
            q = np.zeros(4)
            for v in range(t, 4):
                q[v] = 1.0
            val = np.dot(cv, q)
            if val > best_val + 1e-15:
                best_val = val
                best_q = q.copy()
        
        return best_q
    
    # Compute virtual value coefficients for each good
    cv1 = np.zeros(4)
    cv2 = np.zeros(4)
    for v in range(4):
        # cv(v) = v * P(v_j = v) - P(v_j > v)
        cv1[v] = v * p1[v] - np.sum(p1[v+1:4])
        cv2[v] = v * p2[v] - np.sum(p2[v+1:4])
    
    q1 = solve_single_good(cv1)
    q2 = solve_single_good(cv2)
    
    # Build allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]
        
        # Payment via discrete envelope theorem:
        # p(v1, v2) = v1*q1(v1) + v2*q2(v2) - sum_{k=0}^{v1-1} q1(k) - sum_{k=0}^{v2-1} q2(k)
        pay1 = v1 * q1[v1] - np.sum(q1[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2[v2] - np.sum(q2[:v2]) if v2 > 0 else 0.0
        payments[i] = pay1 + pay2
    
    # Ensure nonnegative payments (IR constraint)
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # Verify IC and IR constraints
    ic_ok = True
    ir_ok = True
    for i in range(16):
        v1_i, v2_i = types[i]
        util_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
        
        # IR: util >= 0
        if util_i < -1e-8:
            ir_ok = False
        
        # IC: for all j, utility when reporting j's allocation but having type i
        # should be <= utility when reporting truthfully
        for j in range(16):
            if i == j:
                continue
            util_report_j = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            if util_report_j > util_i + 1e-8:
                ic_ok = False
    
    if ic_ok and ir_ok:
        # Small cleanup for numerical noise
        allocations[allocations < 1e-9] = 0.0
        allocations[allocations > 1.0 - 1e-9] = 1.0
        payments[payments < 1e-9] = 0.0
        
        return {'allocations': allocations, 'payments': payments}
    
    # Fallback: if verification failed, try the LP approach
    if time.time() - t0 < DEADLINE - 5:
        try:
            from scipy.optimize import linprog
            
            n_vars = 48
            c = np.zeros(n_vars)
            for i in range(16):
                c[32 + i] = -probs[i]
                
            A_ub = []
            b_ub = []
            
            # IC Constraints
            for i in range(16):
                for j in range(16):
                    if i == j:
                        continue
                    row = np.zeros(n_vars)
                    row[32 + i] = 1.0
                    row[32 + j] = -1.0
                    
                    v_i = np.array(types[i], dtype=float)
                    for k in range(2):
                        row[i * 2 + k] = v_i[k]
                        row[j * 2 + k] = -v_i[k]
                        
                    A_ub.append(row)
                    b_ub.append(0.0)
                    
            # IR Constraints: p_i <= v_i · x_i  =>  p_i - v_i·x_i <= 0
            for i in range(16):
                row = np.zeros(n_vars)
                row[32 + i] = 1.0
                row[i * 2 + 0] = -types[i][0]
                row[i * 2 + 1] = -types[i][1]
                A_ub.append(row)
                b_ub.append(0.0)
                
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            bounds = []
            for i in range(16):
                for k in range(2):
                    bounds.append((0.0, 1.0))
            for i in range(16):
                bounds.append((0.0, None))
                
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                q_flat = res.x[:32].reshape(16, 2)
                p_vec = res.x[32:]
                
                allocations_lp = np.clip(q_flat, 0.0, 1.0)
                payments_lp = np.maximum(p_vec, 0.0)
                
                # Verify
                ic_ok2 = True
                ir_ok2 = True
                for i in range(16):
                    v1_i, v2_i = types[i]
                    util_i = v1_i * allocations_lp[i, 0] + v2_i * allocations_lp[i, 1] - payments_lp[i]
                    if util_i < -1e-8:
                        ir_ok2 = False
                    for j in range(16):
                        if i == j:
                            continue
                        util_report_j = v1_i * allocations_lp[j, 0] + v2_i * allocations_lp[j, 1] - payments_lp[j]
                        if util_report_j > util_i + 1e-8:
                            ic_ok2 = False
                
                if ic_ok2 and ir_ok2:
                    allocations_lp[allocations_lp < 1e-9] = 0.0
                    allocations_lp[allocations_lp > 1.0 - 1e-9] = 1.0
                    payments_lp[payments_lp < 1e-9] = 0.0
                    return {'allocations': allocations_lp, 'payments': payments_lp}
        except Exception:
            pass
    
    # Final fallback: return the first solution (which should be valid)
    allocations[allocations < 1e-9] = 0.0
    allocations[allocations > 1.0 - 1e-9] = 1.0
    payments[payments < 1e-9] = 0.0
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
