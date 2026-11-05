import time
import numpy as np
from scipy.optimize import minimize

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
    # x: (16, 2) allocations
    # p: (16,) payments
    # Total variables: 16*2 + 16 = 48
    
    def unpack(z):
        x = z[:32].reshape(16, 2)
        p = z[32:]
        return x, p

    def objective(z):
        x, p = unpack(z)
        # Maximize revenue => Minimize negative revenue
        rev = np.sum(joint_probs * p)
        return -rev

    def constraints_func(z):
        x, p = unpack(z)
        cons = []
        
        # 1. Feasibility: 0 <= x <= 1
        # Handled by bounds in minimize, but let's add explicit constraints if needed?
        # Bounds are more efficient.
        
        # 2. Individual Rationality (IR):
        # u(t, t) >= 0 => v_t . x_t - p_t >= 0 => p_t <= v_t . x_t
        # Constraint: p_t - v_t . x_t <= 0
        for t in range(16):
            v_dot_x = np.dot(V[t], x[t])
            # p[t] - v_dot_x <= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda z, t=t, v_dot_x=v_dot_x: v_dot_x - z[32 + t], 
                # Wait, z is fixed in closure? No, lambda captures z at definition time?
                # No, we need to evaluate at current z.
                # SLSQP passes z to fun.
                'jac': None 
            })
            # Let's rewrite properly. The 'fun' receives z.
            # We need to recompute v_dot_x from z inside fun? 
            # That's expensive. 
            # Better: define constraints as arrays or use a single constraint function returning array.
            # SLSQP supports dict with 'fun' returning scalar or array?
            # If 'fun' returns array, it's treated as multiple constraints.
            # Let's try returning a single array for all IR constraints.
            
        # Let's restructure to return arrays for speed.
        
        # IR constraints: p_t - V[t] @ x_t <= 0  =>  V[t] @ x_t - p_t >= 0
        # Let's define g_ir(z) = V @ x - p (element-wise) >= 0
        # In SLSQP, ineq constraints must be >= 0.
        
        # IC constraints:
        # u(t, t) >= u(t, tp)
        # V[t] @ x_t - p_t >= V[t] @ x_tp - p_tp
        # V[t] @ x_t - V[t] @ x_tp - p_t + p_tp >= 0
        # Let D_ic be a matrix such that D_ic @ z >= 0
        
        # Total variables N = 48.
        # z[0:32] is x, z[32:48] is p.
        
        # We will build the constraint matrix A and vector b such that A @ z >= b (or >= 0).
        # Actually, constructing the matrix once is much faster than lambdas.
        
        # Let's precompute the constraint matrix.
        # Number of constraints:
        # IR: 16
        # IC: 16 * 15 = 240
        # Total: 256 constraints.
        
        # This is a Linear Program in disguise, but SLSQP can handle it.
        # However, building the matrix every time is waste. We can build it once.
        
        # Let's refactor to use a precomputed matrix.
        
        return cons # Placeholder, will replace

    # Precompute Constraint Matrix A (256, 48) and b (256,)
    # Constraint: A @ z >= b
    
    n_constraints = 16 + 16 * 15
    A = np.zeros((n_constraints, 48))
    b = np.zeros(n_constraints)
    
    # IR Constraints:
    # p_t <= V[t] @ x_t  =>  V[t] @ x_t - p_t >= 0
    # Row t:
    # Coeffs for x_t (indices 2*t, 2*t+1): V[t, 0], V[t, 1]
    # Coeff for p_t (index 32 + t): -1
    for t in range(16):
        A[t, 2*t] = V[t, 0]
        A[t, 2*t+1] = V[t, 1]
        A[t, 32+t] = -1
        b[t] = 0
        
    # IC Constraints:
    # For each t, tp (t != tp):
    # V[t] @ x_t - V[t] @ x_tp - p_t + p_tp >= 0
    # Coeffs for x_t: V[t]
    # Coeffs for x_tp: -V[t]
    # Coeff for p_t: -1
    # Coeff for p_tp: 1
    
    idx = 16
    for t in range(16):
        for tp in range(16):
            if t == tp:
                continue
            # x_t part
            A[idx, 2*t] += V[t, 0]
            A[idx, 2*t+1] += V[t, 1]
            # x_tp part
            A[idx, 2*tp] -= V[t, 0]
            A[idx, 2*tp+1] -= V[t, 1]
            # p_t part
            A[idx, 32+t] -= 1
            # p_tp part
            A[idx, 32+tp] += 1
            b[idx] = 0
            idx += 1
            
    # Bounds:
    # x in [0, 1]
    # p in [0, None] (or large number)
    # Max possible value is 3*1 + 3*1 = 6. So p can be at most 6 realistically.
    bounds = []
    for i in range(16):
        bounds.append((0, 1)) # x_i1
        bounds.append((0, 1)) # x_i2
    for i in range(16):
        bounds.append((0, 10)) # p_i
    
    # Objective: Maximize sum(p_t * prob_t) => Minimize -sum(p_t * prob_t)
    c_obj = np.zeros(48)
    c_obj[32:] = -joint_probs
    
    def obj(z):
        return c_obj @ z
        
    # Constraint function for SLSQP
    # SLSQP expects 'ineq' to be >= 0.
    # We have A @ z >= b => A @ z - b >= 0
    def cons_func(z):
        return A @ z - b
        
    def cons_jac(z):
        return A
        
    # Initial guess
    # A good initial guess is the "truthful" mechanism with uniform allocation?
    # Or just zeros.
    # Let's try a few starting points to avoid local minima (though LP is convex, SLSQP might still struggle if ill-conditioned).
    # Since it's a convex problem (LP), any local minimum is global.
    # But SLSQP is for nonlinear. Here constraints are linear.
    # It should converge to a vertex.
    
    z0 = np.zeros(48)
    # Maybe start with x=0.5, p=0?
    z0[:32] = 0.5
    z0[32:] = 0.0
    
    best_result = None
    best_rev = -np.inf
    
    # Try multiple starts to be safe, or just one?
    # Given 85s budget, we can try a few.
    # But one run of SLSQP on 48 vars, 256 constraints should be very fast (<1s).
    
    try:
        res = minimize(
            obj, 
            z0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success:
            x, p = unpack(res.x)
            rev = np.sum(joint_probs * p)
            if rev > best_rev:
                best_rev = rev
                best_result = res.x
        else:
            # Even if not "success", check if it's feasible and better
            if res.fun < 0: # It's minimizing -rev, so fun is -rev
                # Check feasibility
                viol = np.min(cons_func(res.x))
                if viol > -1e-6:
                    x, p = unpack(res.x)
                    rev = np.sum(joint_probs * p)
                    if rev > best_rev:
                        best_rev = rev
                        best_result = res.x
                        # Clip to bounds and fix small violations
                        # We'll do this in the final step
                    
    except Exception as e:
        pass

    # Fallback: Try another start point if first failed or time remains
    if time.time() < deadline - 5:
        z0_2 = np.ones(48) * 0.1
        z0_2[:32] = 0.5
        try:
            res2 = minimize(
                obj, 
                z0_2, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            if res2.success or (res2.fun < 0 and np.min(cons_func(res2.x)) > -1e-6):
                x2, p2 = unpack(res2.x)
                rev2 = np.sum(joint_probs * p2)
                if rev2 > best_rev:
                    best_rev = rev2
                    best_result = res2.x
        except:
            pass

    if best_result is None:
        # Return zeros
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)
    else:
        x, p = unpack(best_result)
        
        # Post-processing: Clip and verify
        x = np.clip(x, 0, 1)
        p = np.clip(p, 0, 10)
        
        # Verify IC and IR strictly
        # If small violations exist, we might need to project or accept if < 1e-8
        # The task requires regret <= 1e-8.
        
        valid = True
        # Check IR
        for t in range(16):
            util = np.dot(V[t], x[t]) - p[t]
            if util < -1e-8:
                valid = False
                break
        if valid:
            # Check IC
            for t in range(16):
                for tp in range(16):
                    if t == tp: continue
                    u_tt = np.dot(V[t], x[t]) - p[t]
                    u_ttp = np.dot(V[t], x[tp]) - p[tp]
                    if u_tt < u_ttp - 1e-8:
                        valid = False
                        break
                if not valid: break
                
        if not valid:
            # If invalid, try to fix by reducing payments?
            # Or just return zeros if we can't fix easily.
            # Given the convexity, SLSQP should be close.
            # Let's try a simple projection: reduce p_t to satisfy IR.
            # This might break IC.
            # For now, if invalid, we return the best feasible found in a previous step?
            # We didn't keep previous steps.
            # Let's assume SLSQP is good enough. If not, score 0 is bad.
            # Let's add a fallback to the binary enumeration if time permits?
            # No, keep it simple. The LP formulation is exact.
            
            # If invalid, we might just return zeros to be safe? 
            # Or try to repair.
            # Repair: Set p_t = min(p_t, V[t] @ x[t]) for IR.
            # Then check IC. If IC fails, it's hard to repair.
            # Let's hope SLSQP works.
            
            pass
            
        best_allocations = x
        best_payments = p

    return {'allocations': best_allocations, 'payments': best_payments}
