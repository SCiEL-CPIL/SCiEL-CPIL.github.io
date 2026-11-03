<<<<<<< SEARCH
    # Phase 4: Fine-tuning from the best solution found
    if time_left() > 3.0 and best_bary is not None:
        # Try multiple fine-tuning runs from the best with very small perturbations
        for ft in range(8):
            if time_left() < 2.0:
                break
            scale = 0.01 + 0.01 * (ft / 8.0)
            bary_ft = _make_initial_bary_perturb(best_bary, rng, scale)
            x0 = bary_ft.flatten()
            
            # Use standard objective for high-precision polish
            x_opt = _local_optimize(x0, bounds, maxiter=500, gtol=1e-11)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
        
        # Final polish with even tighter tolerance
        if time_left() > 2.0:
            x0 = best_bary.flatten()
            x_opt = _local_optimize(x0, bounds, maxiter=1000, gtol=1e-12)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
=======
    # Phase 4: Fine-tuning from the best solution found
    if time_left() > 3.0 and best_bary is not None:
        # Try multiple fine-tuning runs from the best with very small perturbations
        for ft in range(8):
            if time_left() < 2.0:
                break
            scale = 0.01 + 0.01 * (ft / 8.0)
            bary_ft = _make_initial_bary_perturb(best_bary, rng, scale)
            x0 = bary_ft.flatten()
            
            # Use Nelder-Mead with tighter tolerances for polish
            x_opt = _nelder_mead_optimize(x0, maxiter=500)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
        
        # Final polish
        if time_left() > 2.0:
            x0 = best_bary.flatten()
            x_opt = _nelder_mead_optimize(x0, maxiter=1000)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
>>>>>>> REPLACE
