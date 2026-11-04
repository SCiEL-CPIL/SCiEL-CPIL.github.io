<<<<<<< SEARCH
    # Strategy 3: Polish the best result found so far
    if best_result is not None and time.time() < deadline - 20.0:
        # First try high-precision SLSQP polish
        polished = _high_precision_polish(best_result, best_w, best_h, deadline)
        if polished is not None:
            value = _total_radius(polished)
            if value > best_value:
                best_value = value
                best_result = polished

    # Strategy 4: Additional penalty-based polish with higher beta
    if best_result is not None and time.time() < deadline - 5.0:
        result = _penalty_optimize(best_result, best_w, best_h, deadline, 
                                   beta_start=100.0, beta_end=50000.0, n_beta_steps=10)
        if result is not None:
            if _validate(result, best_w, best_h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result
=======
    # Strategy 3: Polish the best result found so far
    if best_result is not None and time.time() < deadline - 20.0:
        # First try high-precision SLSQP polish
        polished = _high_precision_polish(best_result, best_w, best_h, deadline)
        if polished is not None:
            value = _total_radius(polished)
            if value > best_value:
                best_value = value
                best_result = polished

    # Strategy 4: Additional penalty-based polish with higher beta
    if best_result is not None and time.time() < deadline - 5.0:
        result = _penalty_optimize(best_result, best_w, best_h, deadline, 
                                   beta_start=100.0, beta_end=50000.0, n_beta_steps=12)
        if result is not None:
            if _validate(result, best_w, best_h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result

    # Strategy 5: Final fine-tuning pass with very high beta
    if best_result is not None and time.time() < deadline - 2.0:
        result = _penalty_optimize(best_result, best_w, best_h, deadline,
                                   beta_start=5000.0, beta_end=100000.0, n_beta_steps=5)
        if result is not None:
            if _validate(result, best_w, best_h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result
>>>>>>> REPLACE

