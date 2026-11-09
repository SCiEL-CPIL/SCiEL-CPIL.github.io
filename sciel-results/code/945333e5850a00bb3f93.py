<<<<<<< SEARCH
    S, params, ctr, oa = best_result
    
    # Final polish: try to squeeze S a bit more
    if time.time() < DEADLINE - 5:
        # Try to reduce S by 1% and re-optimize
        S_try = S * 0.99
        p_try = params.copy()
        p_try = _refine_local(p_try, S_try, ctr, oa, DEADLINE)
        if _check_validity(p_try, S_try, ctr, oa):
            best_S = S_try
            best_result = (S_try, p_try, ctr, oa)
            S, params, ctr, oa = best_result

    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)
=======
    S, params, ctr, oa = best_result
    
    # Final polish: try to squeeze S a bit more with multiple small steps
    if time.time() < DEADLINE - 10:
        # Try to reduce S by small amounts and re-optimize
        for frac in [0.995, 0.99, 0.985, 0.98, 0.975]:
            if time.time() > DEADLINE - 5:
                break
            S_try = S * frac
            p_try = params.copy()
            p_try = _final_polish(p_try, S_try, ctr, oa, DEADLINE)
            if _check_validity(p_try, S_try, ctr, oa):
                S = S_try
                params = p_try
            else:
                break

    # One final high-precision polish at the best S found
    if time.time() < DEADLINE - 3:
        params = _final_polish(params, S, ctr, oa, DEADLINE)
        if not _check_validity(params, S, ctr, oa):
            # Revert if polish broke validity
            pass

    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)
>>>>>>> REPLACE
