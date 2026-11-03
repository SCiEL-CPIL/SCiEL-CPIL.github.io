<<<<<<< SEARCH
    # Try to find lower ranks
    for R in range(35, 25, -1):
=======
    # First, verify and return the rank-19 solution if valid
    if verify_solution(U19, V19, W19, tol=1e-9):
        U_snap, V_snap, W_snap = snap_to_half_integers(U19, V19, W19)
        if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try to find lower ranks
    for R in range(18, 25, -1):
>>>>>>> REPLACE

