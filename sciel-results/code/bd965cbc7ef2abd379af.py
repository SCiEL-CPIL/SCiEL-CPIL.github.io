<<<<<<< SEARCH
    # Final validation
    valid, best_centers, best_sides, final_score = clamp_and_fix(best_centers, best_sides)
    if not valid:
        return centers_init, sides_init

    return best_centers, best_sides
=======
    # Final high-precision polish
    if time.time() < deadline - 5:
        x_flat = to_flat(best_centers, best_sides)
        res = minimize(penalty_function, x_flat, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        centers_res, sides_res = from_flat(res.x)
        valid, c_final, s_final, score_final = clamp_and_fix(centers_res, sides_res)
        if valid and score_final > best_score:
            best_score = score_final
            best_centers = c_final
            best_sides = s_final

    # Final validation
    valid, best_centers, best_sides, final_score = clamp_and_fix(best_centers, best_sides)
    if not valid:
        return centers_init, sides_init

    return best_centers, best_sides
>>>>>>> REPLACE
