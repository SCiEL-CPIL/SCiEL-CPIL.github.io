<<<<<<< SEARCH
    # Step 4: Final polish pass
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=50)
            if polished is not None:
                best_result = {
                    "U": polished[0].tolist(),
                    "V": polished[1].tolist(),
                    "W": polished[2].tolist(),
                }
        except Exception:
            pass

    return best_result
=======
    # Step 4: Final aggressive elimination pass
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN - 10 and best_rank > 30:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            cur_rank = len(cur_U[0])
            
            # Try to eliminate one more column
            U_new, V_new, W_new, new_rank = targeted_elimination(
                cur_U, cur_V, cur_W, cur_rank - 1, rng, tol=1e-8, max_rounds=50
            )
            if new_rank < best_rank:
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = new_rank
        except Exception:
            pass

    # Step 5: Final polish pass
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=50)
            if polished is not None:
                best_result = {
                    "U": polished[0].tolist(),
                    "V": polished[1].tolist(),
                    "W": polished[2].tolist(),
                }
        except Exception:
            pass

    return best_result
>>>>>>> REPLACE
