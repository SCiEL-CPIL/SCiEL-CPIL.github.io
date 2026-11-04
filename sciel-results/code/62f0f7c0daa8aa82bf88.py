<<<<<<< SEARCH
    # Phase 3: BFGS Polish (replaces Riemannian CG and BB GD)
    if best_x is not None and time.time() < deadline - 5.0:
        polish_time = min(10.0, deadline - time.time() - 2.0)
        if polish_time > 1.0:
            x_bfgs, e_bfgs = _polish_bfgs(best_x, polish_time)
            if e_bfgs < best_e:
                best_e = e_bfgs
                best_x = x_bfgs

    # Phase 4: Final High-Precision BFGS Polish
    if best_x is not None and time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 0.5
        if polish_time > 0.5:
            x_final, e_final = _polish_newton(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
=======
    # Phase 3: BB Gradient Descent Polish (adaptive step sizes)
    if best_x is not None and time.time() < deadline - 8.0:
        polish_time = min(8.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            x_bb, e_bb = _bb_polish(best_x, polish_time)
            if e_bb < best_e:
                best_e = e_bb
                best_x = x_bb

    # Phase 4: BFGS Polish
    if best_x is not None and time.time() < deadline - 4.0:
        polish_time = min(6.0, deadline - time.time() - 2.0)
        if polish_time > 1.0:
            x_bfgs, e_bfgs = _polish_bfgs(best_x, polish_time)
            if e_bfgs < best_e:
                best_e = e_bfgs
                best_x = x_bfgs

    # Phase 5: Final High-Precision BFGS Polish
    if best_x is not None and time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 0.5
        if polish_time > 0.5:
            x_final, e_final = _polish_newton(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final
>>>>>>> REPLACE
