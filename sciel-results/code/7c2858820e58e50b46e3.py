import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    values = [0, 1, 2, 3]
    
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
    
    # Normalize probabilities
    sum_p1 = p1.sum()
    sum_p2 = p2.sum()
    if sum_p1 > 0:
        p1 = p1 / sum_p1
    else:
        p1 = np.ones(4) / 4
    if sum_p2 > 0:
        p2 = p2 / sum_p2
    else:
        p2 = np.ones(4) / 4
        
    joint_probs = np.outer(p1, p2).flatten()
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Precompute value vectors for types
    # v_t = [v1_t, v2_t]
    V = np.array(types, dtype=float) # shape (16, 2)
    
    # Variables:
    # x: (16, 2) allocations -> 32 variables
    # p: (16,) payments -> 16 variables
    # Total variables: 48
    
    # Objective: Maximize Revenue = sum(prob_t * p_t)
    # linprog minimizes, so we minimize -Revenue
    c = np.zeros(48)
    c[32:] = -joint_probs  # Negative probability for payment variables
    
    # Inequality Constraints: A_ub @ z <= b_ub
    # We need to convert our ">= 0" constraints to "<= 0"
    # Constraint: A @ z >= b  =>  -A @ z <= -b
    
    # 1. Individual Rationality (IR):
    # u(t, t) >= 0 => V[t] @ x_t - p_t >= 0
    # -V[t] @ x_t + p_t <= 0
    
    # 2. Incentive Compatibility (IC):
    # u(t, t) >= u(t, tp)
    # V[t] @ x_t - p_t >= V[t] @ x_tp - p_tp
    # V[t] @ x_t - V[t] @ x_tp - p_t + p_tp >= 0
    # -V[t] @ x_t + V[t] @ x_tp + p_t - p_tp <= 0
    
    # Build A_ub matrix
    # Rows: 16 (IR) + 16*15 (IC) = 256 constraints
    n_constraints = 16 + 16 * 15
    A_ub = np.zeros((n_constraints, 48))
    b_ub = np.zeros(n_constraints)
    
    # IR Constraints (rows 0 to 15)
    for t in range(16):
        # -V[t,0] * x_t1 - V[t,1] * x_t2 + 1 * p_t <= 0
        A_ub[t, 2*t] = -V[t, 0]
        A_ub[t, 2*t+1] = -V[t, 1]
        A_ub[t, 32+t] = 1
        b_ub[t] = 0
        
    # IC Constraints (rows 16 onwards)
    idx = 16
    for t in range(16):
        for tp in range(16):
            if t == tp:
                continue
            # -V[t] @ x_t + V[t] @ x_tp + p_t - p_tp <= 0
            # Coeffs for x_t (indices 2*t, 2*t+1): -V[t]
            A_ub[idx, 2*t] = -V[t, 0]
            A_ub[idx, 2*t+1] = -V[t, 1]
            # Coeffs for x_tp (indices 2*tp, 2*tp+1): +V[t]
            A_ub[idx, 2*tp] = V[t, 0]
            A_ub[idx, 2*tp+1] = V[t, 1]
            # Coeff for p_t (index 32+t): +1
            A_ub[idx, 32+t] = 1
            # Coeff for p_tp (index 32+tp): -1
            A_ub[idx, 32+tp] = -1
            b_ub[idx] = 0
            idx += 1
            
    # Bounds:
    # x in [0, 1]
    # p in [0, None] (but effectively bounded by IR/IC, let's set a safe upper bound like 10)
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, 10))
        
    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    
    try:
        # Use Highs solver if available (default in recent scipy), else CBC or HiGHS
        # method='highs' is generally faster and more robust for LPs
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            z = res.x
            x = z[:32].reshape(16, 2)
            p = z[32:]
            
            # Post-processing: 
            # LP solvers might return 0.9999999 or 1.0000001 due to floating point.
            # We must ensure feasibility with regret <= 1e-8.
            # Strategy: Round allocations to nearest integer (0 or 1) if close.
            # Since the problem is an LP, the optimal solution is at a vertex.
            # For this specific structure, vertices are often integral.
            # If not integral, we might need to project.
            # However, standard LP solvers return the exact vertex within tolerance.
            
            # Check if allocations are close to binary
            x_rounded = np.round(x)
            
            # Verify if rounded solution is feasible and better/similar
            # Actually, if the LP returns a fractional solution, rounding might violate IC.
            # But for "Incentive-compatible finite auction design" with 1 additive bidder, 
            # the constraint matrix is often Totally Unimodular, implying integral vertices.
            # Let's check the integrality gap.
            
            max_frac = np.max(np.abs(x - x_rounded))
            
            if max_frac < 1e-5:
                # Safe to round
                x_final = x_rounded
            else:
                # Keep the fractional solution? 
                # The task allows "feasible probabilities" for allocations? 
                # "allocations:(16,2)" usually implies probabilities in [0,1].
                # "Truthfulness holds only on the declared report/value support."
                # If fractional allocations are allowed, we don't need to round.
                # However, "finite auction design" often implies deterministic mechanisms in some contexts, 
                # but the interface says "allocations:(16,2)" and "feasible probabilities", 
                # which strongly suggests randomized mechanisms (probabilities) are allowed.
                # If randomized mechanisms are allowed, the LP solution IS the optimal randomized mechanism.
                # We just need to ensure it satisfies constraints strictly.
                
                x_final = x
                
            # Clip to [0,1] to be safe
            x_final = np.clip(x_final, 0, 1)
            p_final = np.clip(p, 0, 10)
            
            # Final Verification and Repair (Error Repair Step)
            # The directive asks about 'error_repair'. 
            # We implement a strict verification loop. If any constraint is violated by > 1e-8,
            # we attempt a simple repair: 
            # 1. If IR violated (p_t > V[t]@x_t), set p_t = V[t]@x_t.
            # 2. If IC violated, this is harder. 
            #    However, since we solved the LP, violations should be numerical noise (< 1e-9).
            #    If violations are larger, the solver failed.
            
            valid = True
            # Check IR
            for t in range(16):
                util = np.dot(V[t], x_final[t]) - p_final[t]
                if util < -1e-8:
                    valid = False
                    # Repair: Adjust payment down
                    p_final[t] = np.dot(V[t], x_final[t])
            if not valid:
                # Re-check after repair
                valid = True
                for t in range(16):
                    util = np.dot(V[t], x_final[t]) - p_final[t]
                    if util < -1e-8:
                        valid = False
            
            if valid:
                # Check IC
                for t in range(16):
                    for tp in range(16):
                        if t == tp: continue
                        u_tt = np.dot(V[t], x_final[t]) - p_final[t]
                        u_ttp = np.dot(V[t], x_final[tp]) - p_final[tp]
                        if u_tt < u_ttp - 1e-8:
                            valid = False
                            break
                    if not valid: break
                
            if valid:
                best_allocations = x_final
                best_payments = p_final
            else:
                # If invalid, fallback to zero mechanism (safe, revenue 0)
                # Or try to recover by reducing payments further?
                # Reducing payments helps IR but hurts IC (makes u_tt larger relative to u_ttp? 
                # No, u_tt = Vx - p. Decreasing p increases u_tt. 
                # u_ttp = Vx' - p'. Decreasing p' increases u_ttp.
                # IC: u_tt >= u_ttp.
                # If we decrease all p by same amount, IC holds.
                # But IR: p <= Vx.
                # So we can lower p.
                
                # Simple repair: Set p_t = max(0, min(p_t, V[t]@x_final[t]))
                # This fixes IR. 
                # Does it fix IC?
                # u_tt = Vx_t - p_t. u_ttp = Vx_tp - p_tp.
                # If we reduce p_t and p_tp, the difference u_tt - u_ttp changes by -(delta_p_t) + delta_p_tp.
                # This is complex. 
                # Given LP optimality, if it's invalid, it's likely a solver failure.
                # We return the best feasible found or zeros.
                
                # Let's try a more aggressive repair: 
                # Binary search on a scaling factor for payments? No.
                # Just return zeros if we can't verify.
                best_allocations = np.zeros((16, 2))
                best_payments = np.zeros(16)
                
    except Exception as e:
        # Fallback to zeros
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {'allocations': best_allocations, 'payments': best_payments}
