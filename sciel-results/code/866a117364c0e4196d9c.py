"""
Incentive-compatible finite auction design with high-precision LP polishing.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin before 90s kill
    
    # 1. Parse the problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    if n_types != 16:
        # Fallback to standard 4x4 grid if support parsing fails or is non-standard
        types = [(i, j) for i in values for j in values]
        n_types = 16
    
    # type_to_vals mapping
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)
    
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # 2. Build the Linear Program
    # Variables:
    # x1[0..15] (indices 0-15)
    # x2[0..15] (indices 16-31)
    # p[0..15]  (indices 32-47)
    n_vars = 48
    
    A_rows = []
    b_rows = []
    
    # IC constraints:
    # u(t) >= u(t') for all t, t'
    # v1(t)x1(t) + v2(t)x2(t) - p(t) >= v1(t)x1(t') + v2(t)x2(t') - p(t')
    # => p(t) - p(t') <= v1(t)(x1(t) - x1(t')) + v2(t)(x2(t) - x2(t'))
    # => -v1(t)x1(t) + v1(t)x1(t') - v2(t)x2(t) + v2(t)x2(t') + p(t) - p(t') <= 0
    
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]          # -v1(t)*x1(t)
            row[t2] = v1_all[t]          # +v1(t)*x1(t')
            row[16 + t] = -v2_all[t]     # -v2(t)*x2(t)
            row[16 + t2] = v2_all[t]     # +v2(t)*x2(t')
            row[32 + t] = 1.0            # +p(t)
            row[32 + t2] = -1.0          # -p(t')
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints:
    # u(t) >= 0 => v1(t)x1(t) + v2(t)x2(t) - p(t) >= 0
    # => -v1(t)x1(t) - v2(t)x2(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: Maximize E[p] = sum p(t) * probs[t]
    # linprog minimizes, so c = -probs for p variables
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    # Bounds
    # x1, x2 in [0, 1]
    # p >= 0
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve LP
    result = None
    try:
        # Use HiGHS with tight tolerances
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'time_limit': 30
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception as e:
        # Fallback to default options if specific options fail
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            result = None
    
    if result is not None and result.success:
        x_sol = result.x
        x1 = np.clip(x_sol[0:16], 0, 1)
        x2 = np.clip(x_sol[16:32], 0, 1)
        p = np.maximum(x_sol[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism (valid but low revenue)
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # 3. High-Precision Polish
    # The LP solution might have small numerical errors. We can refine it by
    # identifying the active constraints and solving the linear system exactly.
    
    def refine_solution(x1, x2, p, v1_all, v2_all, A_ub, b_ub, c, probs):
        """
        Refine the solution by solving the active constraint system.
        """
        n_vars = 48
        x_full = np.concatenate([x1, x2, p])
        
        # Calculate slack for all constraints
        slacks = b_ub - A_ub @ x_full
        
        # Identify active constraints (slack close to 0)
        # We need a set of linearly independent active constraints to define the vertex.
        # Since the dimension is 48, we need 48 independent constraints.
        # However, picking the right ones is tricky. A simpler approach:
        # Use the reduced cost / dual information if available, or just
        # re-solve with tighter bounds around the current solution.
        
        # Alternative Polish Strategy: Local NLP/LSQ refinement
        # Since the objective is linear and constraints are linear, the optimum is a vertex.
        # If we assume the current basis is correct, we can solve for the variables
        # using the active constraints.
        
        # Let's try a simpler, robust polish: 
        # Iterate over all pairs of constraints and check if adjusting the solution
        # along the null space of the active set improves feasibility or objective.
        
        # Actually, for this specific small problem, a very effective polish is:
        # 1. Identify constraints with slack < 1e-6 as "active".
        # 2. If we have >= 48 active constraints, pick a subset of 48 linearly independent ones.
        # 3. Solve A_active @ x_new = b_active.
        # 4. Check if x_new is feasible (all slacks >= -tol) and improves objective.
        
        tol = 1e-5
        active_idx = np.where(slacks < tol)[0]
        
        if len(active_idx) < 48:
            # Not enough active constraints to uniquely determine the vertex?
            # Or we are on an edge/face. In that case, the LP solver's solution is likely
            # as good as it gets without more complex basis identification.
            # But we can still try to minimize the violation.
            pass
        else:
            # Select 48 linearly independent rows from A_ub[active_idx]
            A_active = A_ub[active_idx]
            b_active = b_ub[active_idx]
            
            # QR decomposition to find independent rows
            Q, R = np.linalg.qr(A_active.T, mode='complete')
            # The first 48 columns of Q (transposed) correspond to independent rows
            # Actually, rank(A_active) should be 48.
            # Let's use SVD to find the basis.
            U, s, Vt = np.linalg.svd(A_active, full_matrices=False)
            
            # If rank < 48, we can't solve uniquely.
            if s[47] < 1e-8: 
                return x1, x2, p
            
            # Take the first 48 independent rows? No, SVD gives us the basis in column space.
            # We need to find which rows are independent.
            # Simpler: Use np.linalg.lstsq to find the solution that minimizes residual
            # of the active constraints. This effectively projects the solution onto the
            # intersection of the hyperplanes defined by the active constraints.
            
            # We want x_new such that A_active @ x_new = b_active.
            # This is an overdetermined system if len(active_idx) > 48.
            x_new, residuals, rank, sv = np.linalg.lstsq(A_active, b_active, rcond=None)
            
            # Check if the new solution is feasible for ALL constraints
            new_slacks = b_ub - A_ub @ x_new
            if np.min(new_slacks) >= -1e-6:
                # Check if it respects bounds
                if np.all(x_new[0:16] >= -1e-6) and np.all(x_new[0:16] <= 1 + 1e-6) and \
                   np.all(x_new[16:32] >= -1e-6) and np.all(x_new[16:32] <= 1 + 1e-6) and \
                   np.all(x_new[32:48] >= -1e-6):
                    
                    # Check if it improves or maintains objective
                    obj_old = -c @ x_full
                    obj_new = -c @ x_new
                    
                    # Due to numerical noise, obj_new might be slightly worse even if it's the "true" vertex.
                    # We accept it if it's close and feasible.
                    if obj_new >= obj_old - 1e-5:
                        x1_new = np.clip(x_new[0:16], 0, 1)
                        x2_new = np.clip(x_new[16:32], 0, 1)
                        p_new = np.maximum(x_new[32:48], 0)
                        return x1_new, x2_new, p_new
        
        return x1, x2, p

    # Apply Polish
    allocations[:, 0], allocations[:, 1], payments = refine_solution(
        allocations[:, 0], allocations[:, 1], payments, v1_all, v2_all, A_ub, b_ub, c, probs
    )
    
    # 4. Final Verification and Repair
    # Ensure strict IC and IR.
    
    # Check IR
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-9:
            # IR violated. Reduce payment to make util = 0.
            payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    # Check IC
    # u(t) >= u(t') for all t, t'
    # v1(t)x1(t) + v2(t)x2(t) - p(t) >= v1(t)x1(t') + v2(t)x2(t') - p(t')
    # p(t) <= v1(t)x1(t) + v2(t)x2(t) - v1(t)x1(t') - v2(t)x2(t') + p(t')
    
    # If IC is violated, we must reduce payments.
    # The most conservative fix that preserves revenue as much as possible is to
    # lower the payment of the type t that is "overpaying" relative to t'.
    # Specifically, if u(t) < u(t'), we need to increase u(t) or decrease u(t').
    # Increasing u(t) means lowering p(t).
    # Decreasing u(t') means raising p(t'), which might violate IR.
    # So lowering p(t) is safer.
    
    max_violation = 0.0
    violation_pairs = []
    
    for t in range(16):
        for t2 in range(16):
            if t == t2: continue
            v1_t, v2_t = type_to_vals[t]
            v1_t2, v2_t2 = type_to_vals[t2]
            
            u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            u_t2 = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            
            if u_t < u_t2 - 1e-9:
                violation = u_t2 - u_t
                max_violation = max(max_violation, violation)
                violation_pairs.append((t, t2, violation))
    
    if max_violation > 1e-9:
        # We need to fix these violations.
        # A simple, robust strategy: reduce all payments by max_violation.
        # This is suboptimal but safe.
        # Better: Identify the "source" of the violation.
        # If u(t) < u(t'), it means p(t) is too high OR p(t') is too low.
        # Since we want to maximize revenue, we should lower p(t).
        # But lowering p(t) might make u(t) even lower if t is involved in other violations as t'.
        
        # Let's use a loop to fix violations iteratively.
        for _ in range(10): # Max iterations
            current_max_violation = 0.0
            fix_applied = False
            for t in range(16):
                for t2 in range(16):
                    if t == t2: continue
                    v1_t, v2_t = type_to_vals[t]
                    v1_t2, v2_t2 = type_to_vals[t2]
                    
                    u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                    u_t2 = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
                    
                    if u_t < u_t2 - 1e-9:
                        violation = u_t2 - u_t
                        if violation > current_max_violation:
                            current_max_violation = violation
                            # Reduce payment of t to fix this specific violation
                            # We need u_t to increase by 'violation', so p_t must decrease by 'violation'.
                            new_p_t = payments[t] - violation
                            # Ensure new_p_t >= 0 and IR holds
                            min_p_t = 0.0
                            # IR: p_t <= v1_t*x1_t + v2_t*x2_t
                            max_p_t_ir = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                            
                            # If we can lower p_t, do it.
                            if new_p_t >= min_p_t and new_p_t <= max_p_t_ir + 1e-9:
                                payments[t] = new_p_t
                                fix_applied = True
                                break
                if fix_applied:
                    break
            
            if not fix_applied:
                # No single payment reduction can fix the current max violation without breaking IR?
                # This implies we are in a complex cycle or the allocations are the issue.
                # Fallback: Reduce ALL payments by the max violation found in this pass.
                if current_max_violation > 0:
                    payments = np.maximum(payments - current_max_violation, 0.0)
                break
            
            # Check if all violations are fixed
            all_fixed = True
            for t in range(16):
                for t2 in range(16):
                    if t == t2: continue
                    v1_t, v2_t = type_to_vals[t]
                    v1_t2, v2_t2 = type_to_vals[t2]
                    u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                    u_t2 = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
                    if u_t < u_t2 - 1e-9:
                        all_fixed = False
                        break
                if not all_fixed: break
            
            if all_fixed:
                break

    # Final IR check after IC fixes
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-9:
            payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
