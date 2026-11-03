<<<<<<< SEARCH
    # SA-based refinement on best aspect ratio
    if best_result is not None and time_left() > 25:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=1500, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value, best_result, best_w, best_h = val, circles_sa.copy(), w_sa, h_sa
        if time_left() > 12:
            circles_p = multi_phase_polish(circles_sa, w_sa, h_sa)
            nbrs = delaunay_neighbors(circles_p[:, :2])
            if fast_valid(circles_p, w_sa, h_sa, nbrs, tol=1e-8):
                val_p = evaluate(circles_p, w_sa, h_sa, tol=1e-8)
                if val_p > best_value:
                    best_value, best_result, best_w, best_h = val_p, circles_p.copy(), w_sa, h_sa

    # Final high-precision polish on the best result
    if best_result is not None and time_left() > 8:
        circles_hp = high_precision_polish(best_result, best_w, best_h)
        nbrs_hp = delaunay_neighbors(circles_hp[:, :2])
        if fast_valid(circles_hp, best_w, best_h, nbrs_hp, tol=1e-9):
            val_hp = evaluate(circles_hp, best_w, best_h, tol=1e-9)
            if val_hp > best_value:
                best_value, best_result = val_hp, circles_hp.copy()
=======
    # SA-based refinement on best aspect ratio
    if best_result is not None and time_left() > 30:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=1500, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value, best_result, best_w, best_h = val, circles_sa.copy(), w_sa, h_sa
        if time_left() > 15:
            circles_p = contact_graph_polish(circles_sa, w_sa, h_sa, n_passes=2,
                                             phases=((100, 200), (300, 400), (600, 800)))
            nbrs = delaunay_neighbors(circles_p[:, :2])
            if fast_valid(circles_p, w_sa, h_sa, nbrs, tol=1e-8):
                val_p = evaluate(circles_p, w_sa, h_sa, tol=1e-8)
                if val_p > best_value:
                    best_value, best_result, best_w, best_h = val_p, circles_p.copy(), w_sa, h_sa

    # Joint aspect-ratio optimization on best result
    if best_result is not None and time_left() > 20:
        circles_j, w_j, h_j = joint_aspect_polish(best_result, best_w, best_h,
                                                  max_iter=2000, beta=300.0)
        val_j = evaluate(circles_j, w_j, h_j)
        if val_j > best_value:
            best_value, best_result, best_w, best_h = val_j, circles_j.copy(), w_j, h_j
        # Polish the joint result
        if time_left() > 10:
            circles_jp = contact_graph_polish(circles_j, w_j, h_j, n_passes=2,
                                              phases=((200, 300), (500, 600), (1000, 1000)))
            nbrs_jp = delaunay_neighbors(circles_jp[:, :2])
            if fast_valid(circles_jp, w_j, h_j, nbrs_jp, tol=1e-8):
                val_jp = evaluate(circles_jp, w_j, h_j, tol=1e-8)
                if val_jp > best_value:
                    best_value, best_result, best_w, best_h = val_jp, circles_jp.copy(), w_j, h_j

    # Final high-precision polish on the best result
    if best_result is not None and time_left() > 8:
        circles_hp = high_precision_polish(best_result, best_w, best_h)
        nbrs_hp = delaunay_neighbors(circles_hp[:, :2])
        if fast_valid(circles_hp, best_w, best_h, nbrs_hp, tol=1e-9):
            val_hp = evaluate(circles_hp, best_w, best_h, tol=1e-9)
            if val_hp > best_value:
                best_value, best_result = val_hp, circles_hp.copy()
>>>>>>> REPLACE
