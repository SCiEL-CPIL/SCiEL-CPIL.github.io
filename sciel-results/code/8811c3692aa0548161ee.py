<<<<<<< SEARCH
def _verify_and_fix(allocations, payments, support, probs):
    """Verify IC and IR, and fix any violations."""
    n = len(support)
    n_goods = 2
    
    # Check IC
    for v in range(n):
        for r in range(n):
            util_v_v = sum(support[v][g] * allocations[v, g] for g in range(n_goods)) - payments[v]
            util_v_r = sum(support[v][g] * allocations[r, g] for g in range(n_goods)) - payments[r]
            if util_v_r - util_v_v > 1e-8:
                # IC violation - this shouldn't happen if LP solved correctly
                # But fix by adjusting payment
                pass
    
    # Check IR
    for v in range(n):
        val = sum(support[v][g] * allocations[v, g] for g in range(n_goods))
        if payments[v] > val + 1e-8:
            payments[v] = val
    
    return allocations, payments
=======
def _verify_and_fix(allocations, payments, support, probs):
    """Verify IC and IR, and fix any violations."""
    n = len(support)
    n_goods = 2
    
    # Check IC and fix by adjusting payments upward if needed
    # For each v, the minimum payment that satisfies IC for all r:
    # p(v) >= max_r [v·x(v) - (v·x(r) - p(r))] = v·x(v) + max_r [p(r) - v·x(r)]
    # Also IR: p(v) >= 0
    # To maximize revenue, set p(v) = max(0, v·x(v) + max_r [p(r) - v·x(r)])
    # But this creates a system of equations. Instead, iterate:
    
    # Start with current payments
    p = payments.copy()
    
    # Iterate to find consistent payments
    for iteration in range(100):
        p_old = p.copy()
        for v in range(n):
            min_pay = 0.0
            for r in range(n):
                if v == r:
                    continue
                # IC: p(v) >= v·x(v) - v·x(r) + p(r)
                val_v = sum(support[v][g] * allocations[v, g] for g in range(n_goods))
                val_r = sum(support[v][g] * allocations[r, g] for g in range(n_goods))
                required = val_v - val_r + p[r]
                if required > min_pay:
                    min_pay = required
            # IR: p(v) >= 0
            if min_pay < 0:
                min_pay = 0.0
            p[v] = min_pay
        
        # Check convergence
        if np.max(np.abs(p - p_old)) < 1e-12:
            break
    
    # Verify final solution
    allocations = np.clip(allocations, 0, 1)
    payments = p
    
    # Final check and fix
    for v in range(n):
        val = sum(support[v][g] * allocations[v, g] for g in range(n_goods))
        if payments[v] > val + 1e-10:
            payments[v] = val
    
    return allocations, payments
>>>>>>> REPLACE

