"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Bitset greedy init + iterative dominance pruning + ILS local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    # General fallback: greedy set cover
    return _greedy_fallback(q, n, radius, start_time, deadline)


def _solve_ternary_7_1(start_time, deadline):
    q, n = 3, 7
    N = q ** n  # 2187

    # Precompute all words and neighbors (ball of radius 1 = 15 points)
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    neighbors = np.empty((N, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        neighbors[i, 0] = i
        k = 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    nw = list(w)
                    nw[pos] = val
                    neighbors[i, k] = word_to_idx[tuple(nw)]
                    k += 1

    # coverers[point] = list of codewords whose ball contains point
    coverers = [[] for _ in range(N)]
    for cw in range(N):
        for pt in neighbors[cw]:
            coverers[pt].append(cw)

    # Ball bitsets for fast coverage
    ball_bitsets = np.empty(N, dtype=object)
    for i in range(N):
        m = 0
        for j in neighbors[i]:
            m |= (1 << j)
        ball_bitsets[i] = m
    all_mask = (1 << N) - 1

    def greedy_init():
        """Greedy set cover using bitsets."""
        codewords = []
        is_chosen = [False] * N
        cov_mask = 0
        while cov_mask != all_mask:
            if time.time() > deadline - 60:
                break
            best_i, best_count = -1, 0
            for i in range(N):
                if is_chosen[i]:
                    continue
                nb = ball_bitsets[i] & ~cov_mask
                if nb:
                    c = nb.bit_count()
                    if c > best_count:
                        best_count = c
                        best_i = i
            if best_i == -1:
                break
            codewords.append(best_i)
            is_chosen[best_i] = True
            cov_mask |= ball_bitsets[best_i]
        # ensure validity
        if cov_mask != all_mask:
            for i in range(N):
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                cov_mask |= ball_bitsets[i]
                if cov_mask == all_mask:
                    break
        return codewords

    def build_cover(codewords):
        cov = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            cov[neighbors[cw]] += 1
        return cov

    def try_remove_redundant(cw_list, cov):
        """Remove codewords whose ball is fully covered by others."""
        removed = 0
        i = 0
        while i < len(cw_list):
            if time.time() > deadline - 15:
                break
            cw = cw_list[i]
            if np.min(cov[neighbors[cw]]) > 1:
                cov[neighbors[cw]] -= 1
                cw_list.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    def local_search(cw_list, cov, budget):
        """Perturbation-based iterated local search."""
        best = list(cw_list)
        best_size = len(cw_list)
        no_improve = 0
        rng = random.Random(12345)

        while time.time() < budget:
            # 1. remove redundancies
            if try_remove_redundant(cw_list, cov) > 0:
                no_improve = 0
                if len(cw_list) < best_size:
                    best_size = len(cw_list)
                    best = list(cw_list)
                continue

            # 2. pick a random codeword, try to swap/remove
            if not cw_list:
                break
            idx = rng.randrange(len(cw_list))
            old_cw = cw_list[idx]
            b_old = neighbors[old_cw]
            critical = [int(pt) for pt in b_old if cov[pt] == 1]

            if not critical:
                # removable (shouldn't happen after redundancy pass, but safe)
                cov[b_old] -= 1
                cw_list.pop(idx)
                no_improve = 0
                if len(cw_list) < best_size:
                    best_size = len(cw_list)
                    best = list(cw_list)
                continue

            # candidate set = intersection of coverers of critical points
            cand = set(coverers[critical[0]])
            for p in critical[1:]:
                cand.intersection_update(coverers[p])
            cand.discard(old_cw)
            for cw in cw_list:
                cand.discard(cw)

            if cand:
                # pick candidate minimizing redundancy cost
                cand_list = list(cand)
                if len(cand_list) > 40:
                    rng.shuffle(cand_list)
                    cand_list = cand_list[:40]
                best_new, best_cost = -1, float('inf')
                for nc in cand_list:
                    cost = int(np.sum(cov[neighbors[nc]]))
                    if cost < best_cost:
                        best_cost = cost
                        best_new = nc
                if best_new >= 0:
                    cov[b_old] -= 1
                    cov[neighbors[best_new]] += 1
                    cw_list[idx] = best_new
                    no_improve = 0
                    try_remove_redundant(cw_list, cov)
                    if len(cw_list) < best_size:
                        best_size = len(cw_list)
                        best = list(cw_list)
                    continue

            # 3. perturbation: kick to escape local minimum
            no_improve += 1
            if no_improve > 25:
                # random restart of local search from a perturbed state
                if len(cw_list) > 200:
                    for _ in range(3):
                        ri = rng.randrange(len(cw_list))
                        if rng.random() < 0.5:
                            nc = rng.randrange(N)
                            if nc not in cw_list:
                                cw_list[ri] = nc
                                cov[neighbors[cw_list[ri]]] += 1
                no_improve = 0

        return best, best_size

    # Phase 1: greedy initialization
    cw_list = greedy_init()
    cov = build_cover(cw_list)
    try_remove_redundant(cw_list, cov)

    best_solution = list(cw_list)
    best_size = len(cw_list)

    # Phase 2: ILS with multiple restarts, using spare time
    while time.time() < deadline - 10:
        # budget for this ILS round
        round_budget = min(deadline - 10, time.time() + 30)
        sol, sz = local_search(list(cw_list), build_cover(cw_list), round_budget)
        if sz < best_size:
            best_size = sz
            best_solution = sol
            cw_list = sol
        else:
            # perturb current best to diversify
            cw_list = list(best_solution)
            for _ in range(2):
                if cw_list:
                    ri = random.randrange(len(cw_list))
                    nc = random.randrange(N)
                    if nc not in cw_list:
                        cw_list[ri] = nc

    # Final cleanup
    final = list(best_solution)
    final_cov = build_cover(final)
    try_remove_redundant(final, final_cov)
    if np.min(final_cov) >= 1:
        return [list(all_words[idx]) for idx in final]
    return [list(all_words[idx]) for idx in best_solution]


def _greedy_fallback(q, n, radius, start_time, deadline):
    all_words = list(itertools.product(range(q), repeat=n))

    def ball(cw):
        s = {cw}
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    nw = list(cw)
                    nw[i] = val
                    s.add(tuple(nw))
        return s

    codewords = []
    covered = set()
    while time.time() < deadline - 10:
        best_cw, best_new = None, None
        best_count = -1
        for cand in all_words:
            new = ball(cand) - covered
            if len(new) > best_count:
                best_count = len(new)
                best_cw = cand
                best_new = new
        if best_count <= 0:
            break
        codewords.append(best_cw)
        covered |= best_new
        if len(covered) == len(all_words):
            break

    # redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw = codewords[i]
        can = True
        for w in ball(cw):
            if not any(w in ball(o) for j, o in enumerate(codewords) if j != i):
                can = False
                break
        if can:
            codewords.pop(i)
        else:
            i += 1

    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
