    best_obj_val = objective(best_params)
    # ...
    if res.fun < best_obj_val:
        best_params = res.x
        # fix endpoints
        best_params[0] = 0.0
        best_params[1] = 0.0
        best_params[-2] = -2.0
        best_params[-1] = 0.0
        best_obj_val = objective(best_params)
        best_path = get_path(best_params.reshape(-1, 2))
        best_score = evaluate_area(best_path)

    # ...
            if res2.fun < best_obj_val:
                best_params = res2.x
                # fix endpoints
                best_params[0] = 0.0
                best_params[1] = 0.0
                best_params[-2] = -2.0
                best_params[-1] = 0.0
                best_obj_val = objective(best_params)
                best_path = get_path(best_params.reshape(-1, 2))
                best_score = evaluate_area(best_path)
