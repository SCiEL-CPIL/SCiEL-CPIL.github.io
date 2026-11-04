<<<<<<< SEARCH
    # Try ranks
    # Start with lower ranks, go up to schoolbook
    for r in [45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 71]:
        if time.time() > deadline:
            break
        sol, _ = search_rank(r, deadline)
        if sol:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_school, "V": V_school, "W": W_school}
=======
    # Try ranks
    # Start with lower ranks, go up to schoolbook
    for r in [45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 71]:
        if time.time() > deadline:
            break
        sol, _ = search_rank(r, deadline)
        if sol:
            U, V, W = sol
            # Verify the solution
            err = compute_error(U, V, W, r)
            if max(abs(e) for e in err) < 1e-6:
                return {"U": U, "V": V, "W": W}
            
    return {"U": U_school, "V": V_school, "W": W_school}
>>>>>>> REPLACE
