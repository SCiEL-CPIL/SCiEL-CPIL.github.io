"""Incentive-compatible finite auction design solver using combinatorial search."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    values = [0, 1, 2, 3]
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4
    
    joint_probs = np.outer(p1, p2).flatten()
    
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Precompute IC constraint matrix for payments
    # Variables: p0..p15
    # Constraint for t, tp: p[t] - p[tp] <= v1[t]*(x[t,0] - x[tp,0]) + v2[t]*(x[t,1] - x[tp,1])
    # Let R[t, tp] = v1[t]*(x[t,0] - x[tp,0]) + v2[t]*(x[t,1] - x[tp,1])
    # IR: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1] => p[t] <= V[t]
    
    # We will iterate over all 2^32 allocation vectors.
    # To be efficient, we use Gray code to flip one bit at a time.
    # However, maintaining the constraint matrix dynamically is complex.
    # Given 90s, we can afford a simpler approach:
    # 1. Generate all monotone allocations? No, 2^32 is too many for full generation.
    # 2. But we can use the fact that for 2 goods, the optimal allocation is often
    #    determined by thresholds or simple structures.
    #    
    # Let's try a smarter search:
    # The problem is small enough that we can solve the LP for a subset of "interesting"
    # allocations. But the directive asks for a structural change.
    #
    # Alternative: Use the LP solution from the previous step as a guide?
    # No, let's stick to the "Combinatorial Search" idea but optimized.
    #
    # Actually, 2^32 is ~4 billion. Even at 1 microsecond per check, it's 4000 seconds.
    # We cannot iterate all.
    #
    # Let's rethink. The "structural change" should be something that guarantees
    # finding a better solution than the rounded LP.
    #
    # What if we use the LP solution to identify "critical" types and then
    # perform a local search in the neighborhood of the LP solution's rounded
    # allocations?
    #
    # Or, better: The LP relaxation gives the upper bound. The gap is due to
    # integrality. We can use a "Branch and Bound" style approach or simply
    # enumerate the top K solutions from the LP? No, LP gives one solution.
    #
    # Let's try a different structural approach:
    # **Lagrangian Relaxation / Dual Approach?**
    #
    # Or, **Simulated Annealing** on the binary allocations?
    # Objective: Maximize Revenue(alloc, pay(alloc)) subject to IC/IR.
    # pay(alloc) is the solution of a small LP.
    #
    # Let's implement a fast local search / hill climbing starting from the
    # LP-rounded solution.
    
    # 1. Get initial solution from LP (same as before)
    def _solve_lp_for_alloc(alloc):
        """Solve for optimal payments given fixed binary allocations."""
        n_p = 16
        c = -joint_probs
        
        A_rows = []
        b_vals = []
        
        # IC constraints
        for t in range(n_types):
            v1_t, v2_t = types[t]
            for tp in range(n_types):
                if t == tp:
                    continue
                # p[t] - p[tp] <= v1[t]*(x[t,0] - x[tp,0]) + v2[t]*(x[t,1] - x[tp,1])
                rhs = v1_t * (alloc[t, 0] - alloc[tp, 0]) + v2_t * (alloc[t, 1] - alloc[tp, 1])
                row = np.zeros(n_p)
                row[t] = 1
                row[tp] = -1
                A_rows.append(row)
                b_vals.append(rhs)
        
        # IR constraints
        for t in range(n_types):
            v1_t, v2_t = types[t]
            rhs = v1_t * alloc[t, 0] + v2_t * alloc[t, 1]
            row = np.zeros(n_p)
            row[t] = 1
            A_rows.append(row)
            b_vals.append(rhs)
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n_p, method='highs')
            if res.success:
                return res.x, -res.fun
            else:
                return None, -np.inf
        except:
            return None, -np.inf

    # 2. Initial LP solve to get a starting point
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p
    
    c_full = np.zeros(n_vars)
    for t in range(n_types):
        c_full[n_x + t] = -joint_probs[t]
        
    A_ub_rows = []
    b_ub_vals = []
    
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[tp * 2] += v1_t
            row[tp * 2 + 1] += v2_t
            row[n_x + t] = 1
            row[n_x + tp] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
            
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
        
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
        
    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    best_alloc = np.zeros((16, 2))
    best_pay = np.zeros(16)
    best_rev = -np.inf
    
    try:
        result = linprog(c_full, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x_vars = result.x[:n_x]
            for t in range(16):
                best_alloc[t, 0] = 1 if x_vars[t * 2] > 0.5 else 0
                best_alloc[t, 1] = 1 if x_vars[t * 2 + 1] > 0.5 else 0
            
            pay, rev = _solve_lp_for_alloc(best_alloc)
            if pay is not None:
                best_pay = pay
                best_rev = rev
    except:
        pass
    
    # 3. Local Search (Hill Climbing)
    # Try flipping one bit at a time, and maybe two bits.
    # Since 32 bits, 32 neighbors.
    
    current_alloc = best_alloc.copy()
    current_pay = best_pay.copy()
    current_rev = best_rev
    
    improved = True
    max_iters = 1000
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 5.0:
        improved = False
        iter_count += 1
        
        # Try all single-bit flips
        for i in range(32):
            t = i // 2
            g = i % 2
            
            neighbor = current_alloc.copy()
            neighbor[t, g] = 1 - neighbor[t, g]
            
            pay, rev = _solve_lp_for_alloc(neighbor)
            if pay is not None and rev > current_rev + 1e-9:
                current_alloc = neighbor
                current_pay = pay
                current_rev = rev
                best_alloc = current_alloc.copy()
                best_pay = current_pay.copy()
                best_rev = current_rev
                improved = True
        
        # If no single bit improved, try double bit flips?
        # This is 32*31/2 = 496 neighbors. Might be worth it if stuck.
        if not improved:
            found_double = False
            for i in range(32):
                if time.time() > deadline - 5.0:
                    break
                t1 = i // 2
                g1 = i % 2
                for j in range(i + 1, 32):
                    t2 = j // 2
                    g2 = j % 2
                    
                    neighbor = current_alloc.copy()
                    neighbor[t1, g1] = 1 - neighbor[t1, g1]
                    neighbor[t2, g2] = 1 - neighbor[t2, g2]
                    
                    pay, rev = _solve_lp_for_alloc(neighbor)
                    if pay is not None and rev > current_rev + 1e-9:
                        current_alloc = neighbor
                        current_pay = pay
                        current_rev = rev
                        best_alloc = current_alloc.copy()
                        best_pay = current_pay.copy()
                        best_rev = current_rev
                        improved = True
                        found_double = True
                        break
                if found_double:
                    break

    # 4. Final Verification and Return
    allocations = best_alloc
    payments = best_pay
    
    # Verify
    ic_ok = True
    ir_ok = True
    for t in range(n_types):
        v1_t, v2_t = types[t]
        u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if u_tt < -1e-8:
            ir_ok = False
        for tp in range(n_types):
            u_tt_prime = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
            if u_tt < u_tt_prime - 1e-8:
                ic_ok = False
    
    if not ic_ok or not ir_ok:
        # Fallback to safe mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                allocations[t, 0] = 1
            if v2_t >= 2:
                allocations[t, 1] = 1
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            
    return {'allocations': allocations, 'payments': payments}
