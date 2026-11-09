def _greedy_peak_shave(f, max_iters=2000, deadline=None):
    """
    Greedy peak shaving using incremental score updates.
    """
    best = f.copy()
    ac = _autoconv(best)
    best_score = _score(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        
        if peak_idx < N:
            i_start = max(0, peak_idx - (N - 1))
            i_end = min(N - 1, peak_idx)
        else:
            i_start = max(0, peak_idx - N + 1)
            i_end = min(N - 1, peak_idx)
        
        if i_start > i_end:
            break
        
        valid_i = np.arange(i_start, i_end + 1)
        k_vals = peak_idx - valid_i
        products = best[valid_i] * best[k_vals]
        
        same_sign_mask = np.sign(products) == np.sign(peak_val)
        
        if not np.any(same_sign_mask):
            # Try random flip
            idx = np.random.randint(N)
            sc_cand = _score_flip(best, ac, idx)
            if sc_cand < best_score:
                best[idx] *= -1
                ac = _autoconv(best)
                best_score = sc_cand
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        # Try top-3 candidates by |product|
        n_try = min(3, len(candidates_idx))
        order = np.argsort(np.abs(candidates_prod))[::-1][:n_try]
        
        improved = False
        for trial in range(n_try):
            fi = candidates_idx[order[trial]]
            sc_cand = _score_flip(best, ac, fi)
            if sc_cand < best_score:
                best[fi] *= -1
                ac = _autoconv(best)
                best_score = sc_cand
                improved = True
                break
                
    return best, best_score
