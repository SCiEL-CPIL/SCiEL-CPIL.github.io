import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0

    # Parse distribution_spec to get probabilities over 16 types
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.zeros(16)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    v1, v2 = k
                    idx = v1 * 4 + v2
                    if 0 <= idx < 16:
                        probs[idx] = v
                elif isinstance(k, int) and 0 <= k < 16:
                    probs[k] = v
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                tmp = np.zeros(16)
                tmp[:len(probs)] = probs
                probs = tmp
    else:
        probs = np.ones(16) / 16.0

    # Normalize
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(16) / 16.0

    # Ensure non-negative
    probs = np.maximum(probs, 0)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(16) / 16.0

    n_types = 16
    values = np.zeros((n_types, 2))
    for i in range(n_types):
        values[i, 0] = i // 4
        values[i, 1] = i % 4

    # Variables:
    # x[0:32] : allocation probabilities x_{ij}, j in {0,1}, i in 0..15
    # u[32:48] : ex-post utilities u_i
    # Total 48 variables.
    
    n_vars = 48
    
    # Objective: Maximize sum_i pi_i * p_i = sum_i pi_i * (sum_k v_ik x_ik - u_i)
    # Minimize -Objective
    c = np.zeros(n_vars)
    for i in range(n_types):
        for k in range(2):
            # Variable index for x_{ik} is i*2 + k
            c[i * 2 + k] = -probs[i] * values[i, k]
        # Variable index for u_i is 32 + i
        c[32 + i] = probs[i]  # Minimize -(-pi_i u_i) -> +pi_i u_i? No.
        # Wait. Objective to max: R = sum pi_i (val_i x_i - u_i)
        # Minimize -R = sum pi_i u_i - sum pi_i val_i x_i
        # So c for u_i is +pi_i. c for x_ik is -pi_i v_ik.
        # Correct.

    A_ub = []
    b_ub = []

    # 1. IR Constraints: u_i >= 0  =>  -u_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # 2. IC Constraints:
    # For true type j and report i:
    # u_j >= sum_k v_jk x_ik - u_i
    # u_j + u_i - sum_k v_jk x_ik >= 0
    # -u_j - u_i + sum_k v_jk x_ik <= 0
    
    for j in range(n_types):
        for i in range(n_types):
            if i == j:
                # u_j + u_j - sum v_jk x_jk >= 0 is the IR/consistency condition?
                # Actually, IC for i=j is u_j >= val_j x_j - u_j => 2u_j >= val_j x_j.
                # This is implied by IR (u_j>=0) and x_j <= 1? No.
                # If u_j=0 and x_j=1, 0 >= val_j - 0 => val_j <= 0. False if val_j > 0.
                # So we MUST include i=j in the IC loop or handle it separately.
                # The standard IC condition is u_j - val_j x_j >= u_i - val_j x_i for all i.
                # Let u_i = val_i x_i - p_i.
                # u_j - u_i >= val_j x_j - val_j x_i.
                # u_j + val_j x_i >= u_i + val_j x_j.
                # This is the constraint we derived: u_j + u_i >= sum v_jk x_ik?
                # Let's re-derive carefully.
                # Utility of reporting i when true type is j: U(j, i) = sum v_jk x_ik - p_i.
                # Truthfulness requires U(j, j) >= U(j, i) for all i.
                # U(j, j) = sum v_jk x_jk - p_j.
                # U(j, i) = sum v_jk x_ik - p_i.
                # So: sum v_jk x_jk - p_j >= sum v_jk x_ik - p_i.
                # p_i - p_j >= sum v_jk (x_ik - x_jk).
                # Substitute p_k = sum v_kl x_kl - u_k:
                # (sum v_il x_il - u_i) - (sum v_jl x_jl - u_j) >= sum v_jk x_ik - sum v_jk x_jk
                # sum v_il x_il - u_i - sum v_jl x_jl + u_j >= sum v_jk x_ik - sum v_jk x_jk
                # This looks complicated. Let's stick to the utility definition.
                # u_k = sum v_kl x_kl - p_k.
                # p_k = sum v_kl x_kl - u_k.
                # IC: p_i - p_j <= sum v_ik (x_ik - x_jk)  (This is the one in current code)
                # Let's check the current code's IC:
                # p_i - p_j - sum v_ik x_ik + sum v_ik x_jk <= 0
                # p_i - p_j <= sum v_ik x_ik - sum v_ik x_jk = sum v_ik (x_ik - x_jk).
                # This is the standard IC condition for quasi-linear utilities.
                
                # Now let's express this in terms of u.
                # p_i = val_i x_i - u_i
                # p_j = val_j x_j - u_j
                # (val_i x_i - u_i) - (val_j x_j - u_j) <= sum v_ik (x_ik - x_jk)
                # val_i x_i - u_i - val_j x_j + u_j <= sum v_ik x_ik - sum v_ik x_jk
                # val_i x_i - val_j x_j - u_i + u_j <= sum v_ik x_ik - sum v_ik x_jk
                # Note val_i x_i is sum v_ik x_ik.
                # sum v_ik x_ik - val_j x_j - u_i + u_j <= sum v_ik x_ik - sum v_ik x_jk
                # - val_j x_j - u_i + u_j <= - sum v_ik x_jk
                # u_j - u_i <= val_j x_j - sum v_ik x_jk
                # u_j - u_i <= sum v_jk x_jk - sum v_ik x_jk  (since val_j x_j is sum v_jk x_jk)
                # u_j - u_i <= sum v_jk (x_jk - x_ik)
                # u_j + sum v_jk x_ik <= u_i + sum v_jk x_jk
                # This is not linear in a simple "u_i + u_j" form.
                
                # Let's use the property that for single-parameter agents, IC is equivalent to:
                # x_i is monotone and u_i = u_0 + integral... 
                # But for finite discrete types, we can just use the pairwise constraints.
                # The constraint is:
                # p_i - p_j <= sum_k v_{ik} (x_{ik} - x_{jk})
                # Substitute p:
                # (sum v_{ik} x_{ik} - u_i) - (sum v_{jk} x_{jk} - u_j) <= sum_k v_{ik} x_{ik} - sum_k v_{ik} x_{jk}
                # sum v_{ik} x_{ik} - u_i - sum v_{jk} x_{jk} + u_j <= sum v_{ik} x_{ik} - sum v_{ik} x_{jk}
                # - u_i - sum v_{jk} x_{jk} + u_j <= - sum v_{ik} x_{jk}
                # u_j - u_i <= sum v_{jk} x_{jk} - sum v_{ik} x_{jk}
                # u_j - u_i <= sum_k (v_{jk} - v_{ik}) x_{jk}  -- Wait, indices are mixed.
                # RHS: sum_k v_{jk} x_{jk} - sum_k v_{ik} x_{jk} = sum_k (v_{jk} - v_{ik}) x_{jk}.
                # LHS: u_j - u_i.
                # So: u_j - u_i - sum_k (v_{jk} - v_{ik}) x_{jk} <= 0.
                
                # This is a linear constraint in u and x.
                # Variables: u_j, u_i, x_{jk} (for k=0,1).
                # Coeffs:
                # u_j: 1
                # u_i: -1
                # x_{j0}: (v_{j0} - v_{i0})
                # x_{j1}: (v_{j1} - v_{i1})
                # Note: x_{ik} does NOT appear! Only x_{jk}.
                # This is a significant simplification. The allocation of the reported type i does not appear in the constraint relating i and j?
                # Let's re-verify.
                # IC: p_i - p_j <= sum_k v_{ik} (x_{ik} - x_{jk})
                # p_i = V_i(x_i) - u_i
                # p_j = V_j(x_j) - u_j
                # V_i(x_i) - u_i - V_j(x_j) + u_j <= V_i(x_i) - V_i(x_j)
                # - u_i - V_j(x_j) + u_j <= - V_i(x_j)
                # u_j - u_i <= V_j(x_j) - V_i(x_j)
                # u_j - u_i <= sum_k (v_{jk} - v_{ik}) x_{jk}
                
                # Yes! The constraint only involves x_j (allocation of the TRUE type j) and utilities u_i, u_j.
                # This is much sparser.
                
                pass # Handled in loop below

    for j in range(n_types):
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[32 + j] = 1.0
            row[32 + i] = -1.0
            for k in range(2):
                # Variable x_{jk} is at index j*2 + k
                coeff = values[j, k] - values[i, k]
                if coeff != 0:
                    row[j * 2 + k] = coeff
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[32:] = 10.0 # Utilities can be bounded by max value

    best_allocations = None
    best_payments = None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
        if res.status == 0 and res.x is not None:
            x_sol = res.x[:32]
            u_sol = res.x[32:]

            x_sol = np.clip(x_sol, 0, 1)
            u_sol = np.maximum(u_sol, 0)

            allocations = x_sol.reshape(16, 2)
            
            # Calculate payments
            payments = np.zeros(16)
            for i in range(n_types):
                payments[i] = np.sum(values[i] * allocations[i]) - u_sol[i]
                if payments[i] < 0:
                    # Should not happen if IR is satisfied (u_i <= val_i x_i)
                    # But due to numerical noise, clamp to 0?
                    # If u_i > val_i x_i, IR is violated.
                    # IR is u_i >= 0.
                    # Wait, IR is u_i >= 0.
                    # Payment p_i = val_i x_i - u_i.
                    # If u_i is very large, p_i becomes negative.
                    # Is p_i >= 0 required? "nonnegative payments".
                    # Yes. So we need p_i >= 0 => u_i <= val_i x_i.
                    # We did NOT add this constraint explicitly!
                    # We added u_i >= 0.
                    # We need u_i <= val_i x_i.
                    # This is an additional set of constraints: u_i - sum v_ik x_ik <= 0.
                    
                    pass

            # We missed the constraint u_i <= val_i x_i (which ensures p_i >= 0).
            # Let's add it.
            # Actually, if p_i < 0, we can set p_i = 0 and adjust u_i?
            # No, u_i is a variable.
            # We should have added: u_i - sum v_ik x_ik <= 0.
            
            # Let's re-run with this constraint added.
            # Since we are in a try block, let's just build the full correct LP in the next step or fix it here.
            # For now, let's assume the LP solver found a valid point.
            # If p_i < 0, we can clamp p_i to 0. This reduces revenue but maintains validity.
            
            valid_payments = np.maximum(payments, 0)
            
            # Verify IC with valid_payments
            # If we clamped, IC might be broken.
            # It is better to add the constraint u_i <= val_i x_i to the LP.
            
            pass

    except Exception:
        pass

    # Corrected LP Formulation
    # Variables: x (32), u (16)
    # Obj: Max sum pi_i (val_i x_i - u_i)
    # Constraints:
    # 1. u_i >= 0
    # 2. u_i <= val_i x_i  (ensures p_i >= 0)
    # 3. IC: u_j - u_i <= sum_k (v_jk - v_ik) x_jk
    
    c = np.zeros(n_vars)
    for i in range(n_types):
        for k in range(2):
            c[i * 2 + k] = -probs[i] * values[i, k]
        c[32 + i] = probs[i]

    A_ub = []
    b_ub = []

    # IR: u_i >= 0  => -u_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Non-negative Payment: p_i >= 0 => val_i x_i - u_i >= 0 => u_i - val_i x_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        for k in range(2):
            row[i * 2 + k] = -values[i, k]
        A_ub.append(row)
        b_ub.append(0.0)

    # IC: u_j - u_i - sum_k (v_jk - v_ik) x_jk <= 0
    for j in range(n_types):
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[32 + j] = 1.0
            row[32 + i] = -1.0
            for k in range(2):
                coeff = values[j, k] - values[i, k]
                if coeff != 0:
                    row[j * 2 + k] = coeff
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[32:] = 10.0

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
        if res.status == 0 and res.x is not None:
            x_sol = res.x[:32]
            u_sol = res.x[32:]

            x_sol = np.clip(x_sol, 0, 1)
            u_sol = np.maximum(u_sol, 0)

            allocations = x_sol.reshape(16, 2)
            payments = np.zeros(16)
            for i in range(n_types):
                payments[i] = np.sum(values[i] * allocations[i]) - u_sol[i]
                payments[i] = max(0.0, payments[i])

            # Full Verification
            valid = True
            # IR
            for i in range(n_types):
                if payments[i] > np.sum(values[i] * allocations[i]) + 1e-8:
                    valid = False
                    break
                if payments[i] < -1e-8:
                    valid = False
                    break
            
            # IC
            if valid:
                for i in range(n_types):
                    for j in range(n_types):
                        if i == j: continue
                        lhs = payments[i] - payments[j]
                        rhs = np.sum(values[i] * (allocations[i] - allocations[j]))
                        if lhs > rhs + 1e-8:
                            valid = False
                            break
                    if not valid: break

            if valid:
                best_allocations = allocations
                best_payments = payments
            else:
                # Fallback: Project to nearest valid point?
                # Simple fallback: Zero mechanism
                best_allocations = np.zeros((16, 2))
                best_payments = np.zeros(16)
                
    except Exception:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    if best_allocations is None:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {"allocations": best_allocations, "payments": best_payments}
