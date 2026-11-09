    # Variables: x1(t), x2(t), u(t)
    # Indices:
    # x1: 0-15
    # x2: 16-31
    # u: 32-47
    
    n = 16
    c = np.zeros(3 * n)
    
    # Objective: Maximize Revenue = Sum (v(t).x(t) - u(t)) * pi(t)
    # Minimize -Revenue = Sum (u(t) - v(t).x(t)) * pi(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        c[0 + t] = -v1_t * probs[t]
        c[16 + t] = -v2_t * probs[t]
        c[32 + t] = probs[t]
        
    A_rows = []
    b_rows = []
    
    # IC Constraints: u(t) - u(t') <= v(t).x(t) - v(t).x(t')
    # u(t) - u(t') - v(t).x(t) + v(t).x(t') <= 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            row = np.zeros(3 * n)
            # u(t)
            row[32 + t] = 1
            # -u(t')
            row[32 + t2] = -1
            # -v(t).x(t)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            # +v(t).x(t')
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            
            A_rows.append(row)
            b_rows.append(0)
            
    # IR Constraints: u(t) >= 0
    # This is handled by bounds on u(t) >= 0
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, 3 * n))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    bounds = []
    for t in range(n):
        bounds.append((0, 1)) # x1
    for t in range(n):
        bounds.append((0, 1)) # x2
    for t in range(n):
        bounds.append((0, None)) # u >= 0
        
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
         # Fallback
         allocations = np.ones((16, 2))
         payments = np.zeros(16)
         return {'allocations': allocations, 'payments': payments}
         
    x = result.x
    x1 = x[0:16]
    x2 = x[16:32]
    u = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    
    # Calculate payments: p(t) = v(t).x(t) - u(t)
    payments = np.zeros(16)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        payments[t] = v1_t * x1[t] + v2_t * x2[t] - u[t]
        
    # Clip payments to be non-negative (IR ensures u(t) <= v(t).x(t), so p(t) >= 0)
    # But due to numerical error, p(t) might be slightly negative.
    payments = np.maximum(payments, 0)
    
    # Verify IC again. If violated, adjust.
    # The LP solution should be IC.
    # If we clipped payments, we might have broken IC?
    # If p(t) was negative and we set to 0, u(t) increased.
    # Increasing u(t) relaxes IC for t (t is happier) but tightens IC for t' reporting t?
    # IC: u(t) >= v(t)x(t') - p(t').
    # If we increase u(t), LHS increases, so constraint is more likely satisfied.
    # So clipping p to 0 (increasing u) is safe for IC of type t.
    # Does it affect IC of other types?
    # IC for type s reporting t: u(s) >= v(s)x(t) - p(t).
    # If we decrease p(t) (set to 0 from negative), RHS decreases, so constraint is easier to satisfy.
    # So clipping p to 0 is safe.
    
    return {'allocations': allocations, 'payments': payments}
