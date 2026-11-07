import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin under 60s (evaluator kills at 90s)

    # Precompute full comparability matrix (4096 x 4096)
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = (cond1 | cond2)
    np.fill_diagonal(comp, False)
    comp_int = comp.astype(np.int32)  # 0/1 for arithmetic
    base_deg = comp_int.sum(axis=1).astype(np.int32)

    def calc_deg(fam):
        fam_arr = np.array(fam, dtype=np.int32)
        return comp_int[:, fam_arr].sum(axis=1).astype(np.int32)

    def get_score(fam, deg):
        return int(np.sum(deg) // 2)

    def polish(fam, deg):
        """Exact full-4096 single-swap until no positive improvement remains."""
        fam = list(fam)
        deg = deg.copy()
        score = get_score(fam, deg)
        fam_set = set(fam)
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        fam_arr = np.array(fam, dtype=np.int32)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # candidate masks not in family
            cand = np.where(~in_fam_mask)[0]
            if len(cand) == 0:
                break
            cand_arr = cand.astype(np.int32)
            comp_cand_fam = comp_int[np.ix_(cand_arr, fam_arr)]  # (C, m)
            deg_cand = deg[cand_arr]
            deg_fam = deg[fam_arr]
            delta = deg_cand[:, None] - comp_cand_fam - deg_fam[None, :]
            mi = np.unravel_index(np.argmax(delta), delta.shape)
            best_delta = int(delta[mi])
            if best_delta > 0:
                best_new = int(cand_arr[mi[0]])
                best_old = int(fam_arr[mi[1]])
                deg -= comp_int[:, best_old]
                deg += comp_int[:, best_new]
                fam[fam.index(best_old)] = best_new
                fam_set.remove(best_old)
                fam_set.add(best_new)
                in_fam_mask[best_old] = False
                in_fam_mask[best_new] = True
                fam_arr = np.array(fam, dtype=np.int32)
                score += best_delta
                improved = True
        return fam, score

    def local_search(fam):
        fam = list(fam)
        deg = calc_deg(fam)
        score = get_score(fam, deg)
        fam_set = set(fam)
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        not_in_fam_mask = ~in_fam_mask
        fam_arr = np.array(fam, dtype=np.int32)

        T = 10.0
        cooling_rate = 0.95
        max_no_improve = 40
        no_improve_count = 0
        best_score_so_far = score
        best_fam_so_far = list(fam)

        while time.time() < deadline - 3:
            improved = False

            # ---- smooth (entropic) max-delta single swap over top candidates ----
            deg_candidates = deg.copy()
            deg_candidates[in_fam_mask] = -10**6
            K = 200
            if K > total_masks:
                K = total_masks
            top_indices = np.argpartition(deg_candidates, -K)[-K:]
            top_not_in = np.array([not_in_fam_mask[t] for t in top_indices])
            top_arr = np.array(top_indices, dtype=np.int32)[top_not_in]

            if len(top_arr) > 0:
                comp_top_fam = comp_int[np.ix_(top_arr, fam_arr)]  # (C, m)
                deg_top = deg[top_arr]
                deg_fam = deg[fam_arr]
                delta_matrix = deg_top[:, None] - comp_top_fam - deg_fam[None, :]
                best_delta = int(delta_matrix.max())

                if best_delta > 0:
                    # greedy argmax
                    mi = np.unravel_index(np.argmax(delta_matrix), delta_matrix.shape)
                    best_new = int(top_arr[mi[0]])
                    best_old = int(fam_arr[mi[1]])
                else:
                    # smooth surrogate: entropic (log-sum-exp) to diversify away from plateau
                    beta = max(0.5, 1.0 / (T + 1e-9))
                    dm = delta_matrix
                    dm_max = dm.max()
                    z = np.exp(beta * (dm - dm_max))
                    zsum = z.sum()
                    if zsum <= 0 or not np.isfinite(zsum):
                        best_new = best_old = None
                    else:
                        p = z / zsum
                        flat = p.reshape(-1)
                        idx = int(np.random.choice(flat.size, p=flat))
                        mi = np.unravel_index(idx, delta_matrix.shape)
                        best_new = int(top_arr[mi[0]])
                        best_old = int(fam_arr[mi[1]])
                        best_delta = int(delta_matrix[mi])

                if best_new is not None and best_new != best_old and best_new not in fam_set:
                    deg -= comp_int[:, best_old]
                    deg += comp_int[:, best_new]
                    fam[fam.index(best_old)] = best_new
                    fam_set.remove(best_old)
                    fam_set.add(best_new)
                    in_fam_mask[best_old] = False
                    in_fam_mask[best_new] = True
                    not_in_fam_mask[best_old] = True
                    not_in_fam_mask[best_new] = False
                    fam_arr = np.array(fam, dtype=np.int32)
                    score += best_delta
                    improved = True
                    no_improve_count = 0
                    if score > best_score_so_far:
                        best_score_so_far = score
                        best_fam_so_far = list(fam)

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

                # ---- clique-guided worst-member swap (escape move) ----
                if T > 1.0:
                    fam_arr = np.array(fam, dtype=np.int32)
                    comp_fam = comp_int[np.ix_(fam_arr, fam_arr)]
                    np.fill_diagonal(comp_fam, 0)
                    member_deg = comp_fam.sum(axis=1).astype(np.int32)  # comparable count per member
                    worst = int(np.argmin(member_deg))
                    old = int(fam_arr[worst])

                    # best full-4096 candidate for this slot
                    cand = np.where(~in_fam_mask)[0].astype(np.int32)
                    if len(cand) > 0:
                        comp_cand_fam = comp_int[np.ix_(cand, fam_arr)]
                        comp_cand_fam[:, worst] = 0  # self not comparable
                        gain = comp_cand_fam.sum(axis=1)
                        new = int(cand[np.argmax(gain)])

                        if new != old and new not in fam_set:
                            delta = int(gain.max()) - int(member_deg[worst])
                            accept = delta > 0 or (T > 1.0 and np.random.random() < np.exp(delta / T))
                            if accept:
                                deg -= comp_int[:, old]
                                deg += comp_int[:, new]
                                fam[worst] = new
                                fam_set.remove(old)
                                fam_set.add(new)
                                in_fam_mask[old] = False
                                in_fam_mask[new] = True
                                not_in_fam_mask[old] = True
                                not_in_fam_mask[new] = False
                                fam_arr = np.array(fam, dtype=np.int32)
                                score = get_score(fam, deg)
                                if score > best_score_so_far:
                                    best_score_so_far = score
                                    best_fam_so_far = list(fam)
            T *= cooling_rate

        return best_fam_so_far, best_score_so_far

    def greedy_chain_init(rng):
        fam = []
        used = np.zeros(total_masks, dtype=bool)
        for _ in range(m):
            candidates = np.where(~used)[0]
            if len(candidates) == 0:
                break
            if len(fam) > 0:
                fam_arr = np.array(fam, dtype=np.int32)
                comp_with_fam = comp_int[np.ix_(candidates, fam_arr)].sum(axis=1)
                score_cand = comp_with_fam + base_deg[candidates] * 0.1
                best_cand = candidates[np.argmax(score_cand)]
            else:
                best_cand = candidates[rng.integers(len(candidates))]
            fam.append(int(best_cand))
            used[best_cand] = True
        return fam

    best_score = 0
    best_fam = None
    rng = np.random.default_rng(42)

    while time.time() < deadline - 2:
        if rng.random() < 0.5:
            fam = rng.choice(total_masks, size=m, replace=False).tolist()
        else:
            fam = greedy_chain_init(rng)

        cur_fam, cur_score = local_search(fam)

        # final high-precision polish over the full 4096 candidate set
        cur_deg = calc_deg(cur_fam)
        cur_fam, cur_score = polish(cur_fam, cur_deg)

        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam

        if best_score > 700:
            break

    if best_fam is None:
        best_fam = list(range(m))

    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    return result
