# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105.0

    # ... (keep build_base and verify) ...
    
    # Try GD for ranks 45 to 53
    for R_target in range(45, 54):
        if time.time() > deadline - 10.0:
            break
        try:
            U_gd, V_gd, W_gd, success = try_gradient_descent(R_target, deadline - 5.0)
            if success:
                # Verify
                if verify(U_gd, V_gd, W_gd, R_target):
                    return {"U": U_gd.tolist(), "V": V_gd.tolist(), "W": W_gd.tolist()}
        except:
            pass

    # Fallback to base 54
    # ... (keep existing logic) ...
