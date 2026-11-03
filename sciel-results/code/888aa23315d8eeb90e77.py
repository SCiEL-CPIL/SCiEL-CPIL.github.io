<<<<<<< SEARCH
    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 8:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.4, 200), (0.15, 200), (0.05, 200)):
            if _now() > deadline - 8:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 128),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_cont = np.tanh(result.x)
                sc = _score_sequence(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
                    x_start = result.x.copy()
            except Exception:
                continue
=======
    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 5:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.3, 300), (0.1, 300), (0.03, 300), (0.01, 200)):
            if _now() > deadline - 5:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 256),
                    options={'maxiter': iters, 'ftol': 1e-13, 'gtol': 1e-11},
                )
                s_cont = np.tanh(result.x)
                sc = _score_sequence(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
                    x_start = result.x.copy()
            except Exception:
                continue
        
        # Final discrete polish after continuous optimization
        if _now() < deadline - 3:
            cur = best_s.copy()
            cur_ac = fftconvolve(cur, cur, mode='full')
            cur_sum = float(np.sum(cur))
            cur_score = best_score
            for _ in range(500):
                if _now() > deadline - 3:
                    break
                all_scores = _batched_flip_eval(cur, cur_ac, cur_sum)
                best_j = int(np.argmin(all_scores))
                best_js = float(all_scores[best_j])
                if best_js >= cur_score - 1e-12:
                    break
                val = cur[best_j]
                delta = -2.0 * val
                cur_ac = cur_ac.copy()
                cur_ac[best_j:best_j+N] += 2.0 * delta * cur
                cur_ac[best_j] += delta * delta
                cur[best_j] = -val
                cur_sum += delta
                cur_score = best_js
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()
>>>>>>> REPLACE
