<<<<<<< SEARCH
    if best_x is None:
        best_x = _structured_init_fps()
        
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
=======
    if best_x is None:
        best_x = _structured_init_fps()
    
    # Final high-precision polish pass
    if time_left() > 5:
        try:
            # Tighter SLSQP polish
            x_polish = best_x.flatten()
            res = minimize(
                _objective_slsqp, x_polish, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 50}
            )
            bary_polish = _project_to_bary_simplex(res.x.reshape(N, 2))
            score_polish = _min_area_ratio_from_bary(bary_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_x = bary_polish
        except Exception:
            pass
        
        # Additional L-BFGS-B polish with very high beta
        if time_left() > 3:
            try:
                current_x = best_x.flatten()
                for beta in [1000.0, 5000.0]:
                    if time_left() < 2:
                        break
                    res = minimize(
                        _objective_smooth,
                        current_x,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    x_opt = res.x.reshape(N, 2)
                    x_opt = _project_to_bary_simplex(x_opt)
                    score = _min_area_ratio_from_bary(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
                        current_x = x_opt.flatten()
            except Exception:
                pass
    
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
>>>>>>> REPLACE
