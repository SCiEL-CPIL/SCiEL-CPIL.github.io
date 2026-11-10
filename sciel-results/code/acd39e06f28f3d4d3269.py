import numpy as np
from scipy.optimize import linprog
import time
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
    else:
        probs = probs / total

    def compute_minimal_payment(q1, q2):
        """
        Given monotone allocations q1, q2, compute the minimal feasible payment
        vector p that satisfies IC, IR, and p >= 0.
        
        For a single additive bidder, the minimal payment satisfying IC is:
        p[t] = max over all t' of [v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t']) - p[t'] + ...]
        
        Actually, the standard result: if we set p at the lowest type (0,0) to its
        value (which is 0 since v=0), then for any type t, the IC constraint
        p[t] >= p[t'] + v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t']) for all t'
        means p[t] >= max_{t'} [p[t'] + v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t'])]
        
        The minimal such p is determined by the "envelope" from type (0,0).
        Since v(0,0) = 0, p(0,0) >= 0, and IC from (0,0) to t gives:
        p[t] >= p[0,0] + v1t*q1[t] + v2t*q2[t] - v1t*q1[0,0] - v2t*q2[0,0]
        
        But we also need IR: p[t] <= v1t*q1[t] + v2t*q2[t]
        
        The optimal strategy is to set p as high as possible while maintaining IC.
        Actually, to MAXIMIZE revenue, we want p as HIGH as possible subject to IC and IR.
        
        The maximum feasible payment for type t given monotone q is:
        p[t] = min( v1t*q1[t] + v2t*q2[t],  min_{t' > t in some order} [...] )
        
        Let me think more carefully. The IC constraints are:
        For all t, t': p[t] - p[t'] >= v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t'])
        
        This is equivalent to saying that the utility u[t] = v1t*q1[t] + v2t*q2[t] - p[t]
        satisfies: u[t] <= u[t'] + v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t']) - v1t'*(q1[t]-q1[t']) - v2t'*(q2[t]-q2[t'])
        
        Wait, let me redo. IC for type t reporting t':
        v1t*q1[t] + v2t*q2[t] - p[t] >= v1t*q1[t'] + v2t*q2[t'] - p[t']
        => p[t'] - p[t] >= v1t*(q1[t']-q1[t]) + v2t*(q2[t']-q2[t])
        
        So for all t, t': p[t'] >= p[t] + v1t*(q1[t']-q1[t]) + v2t*(q2[t']-q2[t])
        
        To maximize sum p[t]*probs[t], we want each p[t] as large as possible.
        The binding constraints are IR: p[t] <= v1t*q1[t] + v2t*q2[t]
        
        And IC: p[t'] >= p[t] + v1t*(q1[t']-q1[t]) + v2t*(q2[t']-q2[t])
        
        The maximum feasible p is obtained by setting p at the "lowest" types to their IR bounds
        and propagating upward. But "lowest" depends on the partial order.
        
        Actually, for the grid, the key insight is that if q is monotone (non-decreasing in each coordinate),
        then the IC constraints are automatically satisfied if we set:
        p[t] = v1t*q1[t] + v2t*q2[t] - u[t]
        where u[t] is the utility, and the IC constraint becomes:
        u[t] >= u[t'] + v1t*(q1[t]-q1[t']) + v2t*(q2[t]-q2[t']) - v1t'*(q1[t]-q1[t']) - v2t'*(q2[t]-q2[t'])
        
        Hmm, this is getting complicated. Let me just use the LP for the payment subproblem
        but with a smarter approach: solve it directly using the structure.
        
        Actually, the simplest correct approach: for given monotone q, the optimal p is the
        solution to the LP. But we can compute it more efficiently.
        
        The key fact: for a single bidder with additive values on a product space, if q is
        coordinate-wise monotone, then the IC constraints reduce to adjacent type constraints.
        Specifically, it suffices to check:
        - p[t+1 in v1] - p[t] >= v1t*(q1[t+1]-q1[t]) + v2t*(q2[t+1]-q2[t])  (but q2 doesn't change)
          Actually: p[v1+1,v2] - p[v1,v2] >= (v1+1)*(q1[v1+1,v2]-q1[v1,v2]) + v2*(q2[v1+1,v2]-q2[v1,v2])
        - Similarly for v2 direction
        
        But this is still an LP. Let me just keep the LP approach but make it faster.
        
        For now, return None to indicate we should use the LP.
        """
        return None

    # Strategy: Use the LP but with better numerical handling and a direct computation
    # of the optimal payment given monotone allocations.
    
    # First, solve for optimal monotone allocations using a reduced LP
    # Variables: q1[0:16], q2[16:32] with monotonicity constraints
    # For each v1 in 0..2, v2 in 0..3: q1[(v1+1)*4+v2] >= q1[v1*4+v2]
    # For each v1 in 0..3, v2 in 0..2: q2[v1*4+(v2+1)] >= q2[v1*4+v2]
    
    # But we also need to optimize payments. Let's use the full LP but with
    # tighter tolerances and a post-processing step.
    
    # Actually, let's try a different approach: enumerate the "corner" solutions.
    # For single-additive-bidder auctions with finite types, the optimal mechanism
    # has allocations that are 0 or 1 (by a standard result for single-parameter).
    # But for multi-dimensional, this isn't necessarily true.
    
    # Let's try: solve the LP, then round allocations to nearest 0/1 while maintaining
    # monotonicity, then recompute optimal payments.
    
    # Full LP solve
    c = np.zeros(48)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_ub = []
    b_ub = []

    # IC constraints
    for t in range(n):
        v1t, v2t = types[t]
        for tp in range(n):
            if tp == t:
                continue
            row = np.zeros(48)
            row[t] = -v1t
            row[tp] = v1t
            row[16 + t] = -v2t
            row[16 + tp] = v2t
            row[32 + t] = 1.0
            row[32 + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints
    for t in range(n):
        v1t, v2t = types[t]
        row = np.zeros(48)
        row[t] = -v1t
        row[16 + t] = -v2t
        row[32 + t] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Nonnegative payments
    for t in range(n):
        row = np.zeros(48)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    if time.time() > deadline - 10:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    try:
        options = {'presolve': True,
                   'dual_feasibility_tolerance': 1e-14,
                   'primal_feasibility_tolerance': 1e-14,
                   'time_limit': 25.0}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs', options=options)
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if not result.success or result.x is None:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    q1 = np.clip(result.x[0:16], 0, 1)
    q2 = np.clip(result.x[16:32], 0, 1)
    p = np.clip(result.x[32:48], 0, None)

    # Enforce monotonicity
    for v1 in range(3):
        for v2 in range(4):
            t_curr = v1 * 4 + v2
            t_next = (v1 + 1) * 4 + v2
            if q1[t_curr] > q1[t_next] + 1e-10:
                avg = (q1[t_curr] + q1[t_next]) / 2
                q1[t_curr] = min(q1[t_curr], avg + 1e-10)
                q1[t_next] = max(q1[t_next], avg - 1e-10)

    for v1 in range(4):
        for v2 in range(3):
            t_curr = v1 * 4 + v2
            t_next = v1 * 4 + (v2 + 1)
            if q2[t_curr] > q2[t_next] + 1e-10:
                avg = (q2[t_curr] + q2[t_next]) / 2
                q2[t_curr] = min(q2[t_curr], avg + 1e-10)
                q2[t_next] = max(q2[t_next], avg - 1e-10)

    # Now compute the OPTIMAL payment for these fixed allocations
    # This is a small LP with 16 variables
    if time.time() < deadline - 3:
        c_p = np.zeros(n)
        for t in range(n):
            c_p[t] = -probs[t]

        A_ub_p = []
        b_ub_p = []

        # IC constraints: p[tp] - p[t] >= v1t*(q1[tp]-q1[t]) + v2t*(q2[tp]-q2[t])
        # => -p[tp] + p[t] <= -(v1t*(q1[tp]-q1[t]) + v2t*(q2[tp]-q2[t]))
        for t in range(n):
            v1t, v2t = types[t]
            for tp in range(n):
                if tp == t:
                    continue
                row = np.zeros(n)
                row[t] = 1.0
                row[tp] = -1.0
                A_ub_p.append(row)
                b_ub_p.append(-(v1t * (q1[tp] - q1[t]) + v2t * (q2[tp] - q2[t])))

        # IR: p[t] <= v1t*q1[t] + v2t*q2[t]
        for t in range(n):
            v1t, v2t = types[t]
            row = np.zeros(n)
            row[t] = 1.0
            A_ub_p.append(row)
            b_ub_p.append(v1t * q1[t] + v2t * q2[t])

        # p >= 0: -p[t] <= 0
        for t in range(n):
            row = np.zeros(n)
            row[t] = -1.0
            A_ub_p.append(row)
            b_ub_p.append(0.0)

        A_ub_p = np.array(A_ub_p)
        b_ub_p = np.array(b_ub_p)
        bounds_p = [(0, None)] * n

        try:
            result_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p,
                               method='highs',
                               options={'presolve': True,
                                        'dual_feasibility_tolerance': 1e-15,
                                        'primal_feasibility_tolerance': 1e-15,
                                        'time_limit': 10.0})
            if result_p.success and result_p.x is not None:
                p_new = np.clip(result_p.x, 0, None)
                if np.dot(probs, p_new) >= np.dot(probs, p) - 1e-10:
                    p = p_new
        except Exception:
            pass

    # Final verification and regret fix
    max_regret = 0.0
    for t in range(n):
        v1t, v2t = types[t]
        util_t = v1t * q1[t] + v2t * q2[t] - p[t]
        for tp in range(n):
            if tp == t:
                continue
            util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
            regret = util_tp - util_t
            if regret > max_regret:
                max_regret = regret

    if max_regret > 1e-8:
        for iteration in range(100):
            if time.time() > deadline - 2:
                break
            changed = False
            for t in range(n):
                v1t, v2t = types[t]
                min_p = 0.0
                for tp in range(n):
                    if tp == t:
                        continue
                    required = p[tp] + v1t * (q1[t] - q1[tp]) + v2t * (q2[t] - q2[tp])
                    if required > min_p:
                        min_p = required
                max_p = v1t * q1[t] + v2t * q2[t]
                if min_p > max_p:
                    scale = max_p / max(min_p, 1e-12) * 0.999
                    q1[t] *= scale
                    q2[t] *= scale
                    min_p = v1t * q1[t] + v2t * q2[t]
                if min_p > p[t] + 1e-12:
                    p[t] = min_p
                    changed = True
            if not changed:
                break
        p = np.clip(p, 0, None)

    allocations = np.column_stack([q1, q2])
    payments = p

    return {'allocations': allocations, 'payments': payments}
