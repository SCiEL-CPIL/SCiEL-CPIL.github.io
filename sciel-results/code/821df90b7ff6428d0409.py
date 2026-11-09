"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok(margin=2.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _greedy_fill(ow, ov, cap, skip1=-1, skip2=-1, skip3=-1):
    """Greedy fill in density order, optionally skipping up to 3 positions."""
    total = 0.0
    n = len(ow)
    if skip3 < 0:
        for k in range(n):
            if k == skip1 or k == skip2:
                continue
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
                if cap <= 0:
                    break
    else:
        for k in range(n):
            if k == skip1 or k == skip2 or k == skip3:
                continue
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
                if cap <= 0:
                    break
    return total


def _lp_bound(fw, fv, cap, skip1=-1, skip2=-1, skip3=-1):
    """Compute fractional knapsack upper bound for remaining capacity."""
    total = 0.0
    n = len(fw)
    for k in range(n):
        if k == skip1 or k == skip2 or (skip3 >= 0 and k == skip3):
            continue
        wk = fw[k]
        if wk <= 0:
            total += fv[k]
            continue
        if wk <= cap:
            cap -= wk
            total += fv[k]
        else:
            total += fv[k] * (cap / wk)
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    C = float(remaining_capacity)
    if C <= 0:
        return -1
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    if n == 0:
        return -1

    feasible = w <= C
    if not np.any(feasible):
        return -1

    feas_idx = np.where(feasible)[0]
    fw = w[feas_idx]
    fv = v[feas_idx]
    m = len(fw)
    if m == 0:
        return -1

    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / np.where(fw > 0, fw, 1.0), np.inf)
    order = np.argsort(-dens, kind='stable')
    ow = fw[order]
    ov = fv[order]
    pos = np.empty(m, dtype=int)
    pos[order] = np.arange(m)

    if m <= 2:
        best = -1
        best_val = -np.inf
        for i in range(m):
            if fw[i] <= C:
                if fv[i] > best_val:
                    best_val = fv[i]
                    best = i
        if best == -1:
            return -1
        return int(feas_idx[best])

    # 1-step lookahead with LP gap scoring
    scores1 = np.empty(m)
    for i in range(m):
        cap1 = C - fw[i]
        if cap1 < 0:
            scores1[i] = -np.inf
        else:
            greedy_val = fv[i] + _greedy_fill(ow, ov, cap1, skip1=pos[i])
            lp_val = _lp_bound(ow, ov, cap1, skip1=pos[i])
            # Score = greedy value + small bonus for low LP gap (tighter bound)
            gap = lp_val - greedy_val
            scores1[i] = greedy_val + 0.1 * max(0, -gap)  # reward tight bounds

    if not _time_ok(margin=5.0):
        best = int(np.argmax(scores1))
        return int(feas_idx[best])

    # 2-step lookahead for top candidates
    if m <= 40:
        cand = list(range(m))
    else:
        B = min(30, m)
        top = np.argsort(-scores1, kind='stable')[:B]
        cand = [int(t) for t in top]

    best_score = -np.inf
    best_i = cand[0]
    all_scores = {}

    for i in cand:
        cap1 = C - fw[i]
        if cap1 < 0:
            continue
        pi = pos[i]
        base = fv[i]

        best2 = base + _greedy_fill(ow, ov, cap1, skip1=pi)

        for j in range(m):
            if j == i:
                continue
            wj = fw[j]
            if wj > cap1:
                continue
            cap2 = cap1 - wj
            if cap2 < 0:
                continue
            val = base + fv[j] + _greedy_fill(ow, ov, cap2, skip1=pi, skip2=pos[j])
            if val > best2:
                best2 = val

        all_scores[i] = best2
        if best2 > best_score:
            best_score = best2
            best_i = i

        if not _time_ok(margin=2.0):
            break

    # UCB-style tie-breaker: if top scores are within tolerance, prefer items
    # with higher "exploration potential" (lower density rank = more synergy)
    if len(all_scores) >= 2 and _time_ok(margin=3.0):
        sorted_cand = sorted(all_scores.keys(), key=lambda x: -all_scores[x])
        top_score = all_scores[sorted_cand[0]]
        tol = 0.01 * max(1.0, top_score)
        tied = [i for i in sorted_cand if top_score - all_scores[i] < tol]
        if len(tied) > 1:
            # UCB bonus: items with lower density rank (higher pos[i]) get bonus
            # This rewards weight-synergy exploration
            dens_rank = np.argsort(np.argsort(dens))  # rank 0 = highest density
            for i in tied:
                rank_bonus = dens_rank[i] / m  # 0 for top density, 1 for lowest
                all_scores[i] += 0.005 * rank_bonus  # small smooth bonus
            best_i = max(tied, key=lambda x: all_scores[x])

    # Final polish: 3-step lookahead on top-3 if time permits
    if _time_ok(margin=10.0) and len(cand) >= 3:
        top3 = sorted(cand, key=lambda x: -all_scores.get(x, -np.inf))[:3]
        for i in top3:
            cap1 = C - fw[i]
            if cap1 < 0:
                continue
            pi = pos[i]
            base = fv[i]
            # 3-step: try adding a third item
            for j in range(m):
                if j == i:
                    continue
                wj = fw[j]
                if wj > cap1:
                    continue
                cap2 = cap1 - wj
                if cap2 < 0:
                    continue
                pj = pos[j]
                for k in range(m):
                    if k == i or k == j:
                        continue
                        wk = fw[k]
                        if wk > cap2:
                            continue
                        cap3 = cap2 - wk
                        if cap3 < 0:
                            continue
                        val = base + fv[j] + fv[k] + _greedy_fill(ow, ov, cap3, skip1=pi, skip2=pj, skip3=pos[k])
                        if val > all_scores.get(i, -np.inf):
                            all_scores[i] = val
                if not _time_ok(margin=5.0):
                    break
            if not _time_ok(margin=5.0):
                break
        best_i = max(all_scores.keys(), key=lambda x: all_scores[x])

    return int(feas_idx[best_i])
# EVOLVE-BLOCK-END
