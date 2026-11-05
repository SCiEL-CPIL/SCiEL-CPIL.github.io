"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 70  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        probs = np.ones(num_types) / num_types
    else:
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types
    
    # LP: maximize expected payment subject to IC, IR, feasibility
    # Variables: x1[16], x2[16], t[16]
    # Objective: min -sum_i probs[i]*t[i]
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC: u_i(i) >= u_i(j) for all i,j
    # v1_i*x1_i + v2_i*x2_i - t_i >= v1_i*x1_j + v2_i*x2_j - t_j
    # => -v1_i*x1_i - v2_i*x2_i + t_i + v1_i*x1_j + v2_i*x2_j - t_j <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            row = np.zeros(3 * num_types)
            row[i] = -v1_i
            row[num_types + i] = -v2_i
            row[2 * num_types + i] = 1
            row[j] += v1_i
            row[num_types + j] += v2_i
            row[2 * num_types + j] -= 1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR: u_i(i) >= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = np.zeros(3 * num_types)
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = [(0, 1)] * num_types + [(0, 1)] * num_types + [(0, None)] * num_types
    
    def verify_mechanism(x1, x2, t, tol=1e-8):
        for i in range(num_types):
            v1_i, v2_i = types[i]
            u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
            if u_ii < -tol:
                return False
            for j in range(num_types):
                u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                if u_ii < u_ij - tol:
                    return False
        return True
    
    def reconstruct_payments(x1, x2):
        """Reconstruct payments from allocations using monotonicity + IR at origin."""
        # For single additive bidder, IC+IR iff:
        # 1. Allocation is monotone (higher value -> weakly higher allocation in each good)
        # 2. Payment = value*allocation - utility, with utility(0,0)=0
        # We compute utility at each type via path integration from (0,0)
        
        t = np.zeros(num_types)
        
        # Sort types by (v1, v2) to build a monotone path
        # For type i, u_i = max over all paths from (0,0) to i of sum of marginal utilities
        # But for discrete grid, we can use: u_i = max_j { u_j + (v1_i - v1_j)*x1_j + (v2_i - v2_j)*x2_j }
        # This is the Bellman equation for the utility function
        
        # Initialize u[0] = 0 (type (0,0))
        u = np.full(num_types, -np.inf)
        u[0] = 0.0  # type index 0 is (0,0)
        
        # Iterate to convergence (or use topological order since values are bounded)
        for _ in range(10):  # max iterations for convergence
            updated = False
            for i in range(num_types):
                v1_i, v2_i = types[i]
                if v1_i == 0 and v2_i == 0:
                    continue
                for j in range(num_types):
                    v1_j, v2_j = types[j]
                    if v1_j <= v1_i and v2_j <= v2_i and not (v1_j == v1_i and v2_j == v2_i):
                        candidate = u[j] + (v1_i - v1_j) * x1[j] + (v2_i - v2_j) * x2[j]
                        if candidate > u[i] + 1e-15:
                            u[i] = candidate
                            updated = True
            if not updated:
                break
        
        # Compute payments: t[i] = v1_i*x1[i] + v2_i*x2[i] - u[i]
        for i in range(num_types):
            v1_i, v2_i = types[i]
            t[i] = v1_i * x1[i] + v2_i * x2[i] - u[i]
            if t[i] < 0:
                t[i] = 0.0
        
        return t
    
    best_result = None
    best_revenue = -np.inf
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            x1 = np.clip(x[:num_types], 0, 1)
            x2 = np.clip(x[num_types:2*num_types], 0, 1)
            t_lp = np.clip(x[2*num_types:3*num_types], 0, None)
            
            # First try the LP solution directly
            if verify_mechanism(x1, x2, t_lp, tol=1e-8):
                revenue_val = np.sum(probs * t_lp)
                best_result = {'allocations': np.column_stack([x1, x2]), 'payments': t_lp}
                best_revenue = revenue_val
            else:
                # Reconstruct payments from allocations to guarantee IC/IR
                t_recon = reconstruct_payments(x1, x2)
                if verify_mechanism(x1, x2, t_recon, tol=1e-8):
                    revenue_val = np.sum(probs * t_recon)
                    best_result = {'allocations': np.column_stack([x1, x2]), 'payments': t_recon}
                    best_revenue = revenue_val
                else:
                    # Try clipping and re-verifying
                    t_clipped = np.clip(t_recon, 0, None)
                    if verify_mechanism(x1, x2, t_clipped, tol=1e-8):
                        revenue_val = np.sum(probs * t_clipped)
                        best_result = {'allocations': np.column_stack([x1, x2]), 'payments': t_clipped}
                        best_revenue = revenue_val
    
    except Exception:
        pass
    
    # If we have a valid result, try to polish it
    if best_result is not None and time.time() < deadline:
        x1_cur = best_result['allocations'][:, 0].copy()
        x2_cur = best_result['allocations'][:, 1].copy()
        
        # Try a local optimization to improve revenue while maintaining IC/IR
        # Use the reconstructed payments approach: optimize allocations, reconstruct payments
        
        def neg_revenue_from_alloc(z):
            x1_opt = np.clip(z[:num_types], 0, 1)
            x2_opt = np.clip(z[num_types:2*num_types], 0, 1)
            t_opt = reconstruct_payments(x1_opt, x2_opt)
            return -np.sum(probs * t_opt)
        
        z0 = np.concatenate([x1_cur, x2_cur])
        
        try:
            res_opt = minimize(
                neg_revenue_from_alloc, z0, method='SLSQP',
                bounds=[(0, 1)] * (2 * num_types),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res_opt.success:
                x1_new = np.clip(res_opt.x[:num_types], 0, 1)
                x2_new = np.clip(res_opt.x[num_types:2*num_types], 0, 1)
                t_new = reconstruct_payments(x1_new, x2_new)
                
                if verify_mechanism(x1_new, x2_new, t_new, tol=1e-8):
                    revenue_new = np.sum(probs * t_new)
                    if revenue_new > best_revenue + 1e-10:
                        best_result = {'allocations': np.column_stack([x1_new, x2_new]), 'payments': t_new}
                        best_revenue = revenue_new
        
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
