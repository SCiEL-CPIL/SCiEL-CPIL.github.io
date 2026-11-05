import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Generic fallback: greedy
        all_words = [list(t) for t in product(range(q), repeat=n)]
        N = len(all_words)
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        def dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for nv in range(q):
                    if nv == w[pos]:
                        continue
                    nw = w.copy(); nw[pos] = nv
                    neighbors[i].append(word_to_idx[tuple(nw)])
        uncovered = set(range(N))
        code = []
        while uncovered:
            best, bc = -1, -1
            for c in range(N):
                if c in code:
                    continue
                cnt = sum(1 for nb in neighbors[c] if nb in uncovered)
                if cnt > bc:
                    bc, best = cnt, c
            if best == -1:
                break
            code.append(best)
            for nb in neighbors[best]:
                uncovered.discard(nb)
        return [all_words[c] for c in code]

    start = time.time()
    deadline = start + 140  # safety margin under 150s

    # ---- Precompute structure (vectorized) ----
    N = 3 ** 7  # 2187
    # ball_matrix: N x N boolean, ball_matrix[c, w] = 1 if w within dist 1 of c
    # Build via index arithmetic
    def idx_to_tuple(i):
        t = [0] * n
        for pos in range(n - 1, -1, -1):
            t[pos] = i % 3
            i //= 3
        return tuple(t)

    all_words = [list(idx_to_tuple(i)) for i in range(N)]

    # For each position pos and value change, the index delta
    # index = d0*3^6 + d1*3^5 + ... + d6*3^0
    pow3 = [3 ** (n - 1 - pos) for pos in range(n)]  # pow3[pos] = 3^(7-1-pos)
    # neighbors list: for each c, the 15 indices in its ball
    ball = np.zeros((N, N), dtype=bool)
    ball_list = []
    for c in range(N):
        w = all_words[c]
        idxs = [c]
        for pos in range(n):
            base = w[pos]
            for nv in range(3):
                if nv == base:
                    continue
                delta = (nv - base) * pow3[pos]
                idxs.append(c + delta)
        ball_list.append(idxs)
        for j in idxs:
            ball[c, j] = True

    # ball as list of arrays for fast ops
    ball_arr = [np.array(x, dtype=np.int32) for x in ball_list]

    # ---- Greedy with randomization + vectorized gain ----
    def greedy_repair(initial_code=None, rng=None):
        if rng is None:
            rng = random
        if initial_code is None:
            initial_code = []
        code_set = set(initial_code)
        # cover_count: how many codewords cover each word
        cc = np.zeros(N, dtype=np.int32)
        for c in code_set:
            cc[ball_arr[c]] += 1
        while True:
            if time.time() > deadline:
                break
            # uncovered words
            unc = np.where(cc == 0)[0]
            if len(unc) == 0:
                break
            unc_set = set(unc.tolist())
            # gain for each candidate = number of uncovered words in its ball
            # vectorize: for each c, count of unc in ball_arr[c]
            best_c = -1
            best_gain = 0
            # iterate candidates not in code
            # Use the ball matrix: gain[c] = ball[c, :].dot(unc_mask)
            unc_mask = np.zeros(N, dtype=np.int32)
            unc_mask[unc] = 1
            # gains = ball @ unc_mask  (N x N bool @ N int)
            gains = ball @ unc_mask
            # exclude current code
            for c in code_set:
                gains[c] = -1
            # random tie-break among max
            maxg = gains.max()
            if maxg <= 0:
                break
            cands = np.where(gains == maxg)[0]
            if len(cands) == 0:
                break
            chosen = rng.choice(cands.tolist())
            code_set.add(chosen)
            cc[ball_arr[chosen]] += 1
        return code_set, cc

    # ---- Local search: deletion + swap (vectorized) ----
    def local_search(code_set, cc, rng, max_iter=200000):
        best_set = set(code_set)
        best_cc = cc.copy()
        best_size = len(code_set)
        it = 0
        while time.time() < deadline and it < max_iter:
            it += 1
            improved = False
            # Deletion pass: try removing each redundant codeword
            # c is redundant if for all nb in ball[c], cc[nb] >= 2
            # vectorized: min over ball of cc
            for c in list(code_set):
                if time.time() > deadline:
                    break
                b = ball_arr[c]
                if cc[b].min() >= 2:
                    # remove
                    code_set.discard(c)
                    cc[b] -= 1
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_set = set(code_set)
                        best_cc = cc.copy()
                    break
            if improved:
                continue
            # Swap pass: remove c_out, find best c_in to add
            # Try a few random c_out
            if not code_set:
                break
            tried = 0
            while tried < 8 and time.time() < deadline:
                tried += 1
                c_out = rng.choice(list(code_set))
                b_out = ball_arr[c_out]
                # tentatively remove
                cc[b_out] -= 1
                code_set.discard(c_out)
                unc = np.where(cc == 0)[0]
                if len(unc) == 0:
                    # pure deletion found
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_set = set(code_set)
                        best_cc = cc.copy()
                    improved = True
                    break
                unc_mask = np.zeros(N, dtype=np.int32)
                unc_mask[unc] = 1
                gains = ball @ unc_mask
                for c in code_set:
                    gains[c] = -1
                maxg = gains.max()
                if maxg >= len(unc):
                    # full repair -> same size, but we removed one and added one: net same
                    # actually we want to keep size same but maybe enable future deletion
                    c_in = int(np.argmax(gains))
                    code_set.add(c_in)
                    cc[ball_arr[c_in]] += 1
                    # size unchanged; keep as current
                else:
                    # revert
                    code_set.add(c_out)
                    cc[b_out] += 1
            if improved:
                continue
            # Kick: random restart of a few codewords to escape
            # Replace up to 5 random codewords with random new ones (keep covering)
            if len(code_set) > 0:
                kick = min(5, len(code_set))
                to_remove = rng.sample(list(code_set), kick)
                for c in to_remove:
                    cc[ball_arr[c]] -= 1
                    code_set.discard(c)
                # repair
                code_set, cc = greedy_repair(list(code_set), rng)
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_set = set(code_set)
                    best_cc = cc.copy()
        return best_set, best_cc

    # ---- Main: multiple random greedy restarts + local search ----
    best_set = None
    best_size = N  # infinity
    rng = random.Random(12345)
    np.random.seed(42)

    # First, a deterministic-ish greedy to get a baseline fast
    code0, cc0 = greedy_repair([], None)
    code0, cc0 = local_search(code0, cc0, rng, max_iter=50000)
    best_set = set(code0)
    best_size = len(code0)

    # Randomized restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # random greedy: randomize candidate order by permuting gain ties (already random tie-break)
        # Also occasionally start from a random subset to diversify
        init = []
        if restarts % 3 == 0 and best_set is not None:
            # perturb best: remove a few, repair
            init = list(best_set)
            if len(init) > 10:
                to_rm = rng.sample(init, min(8, len(init) - 5))
                init = [c for c in init if c not in set(to_rm)]
        code_r, cc_r = greedy_repair(init, rng)
        code_r, cc_r = local_search(code_r, cc_r, rng, max_iter=30000)
        if len(code_r) < best_size:
            best_size = len(code_r)
            best_set = set(code_r)

    # Final polish: one more local search on best
    if best_set is not None:
        cc_best = np.zeros(N, dtype=np.int32)
        for c in best_set:
            cc_best[ball_arr[c]] += 1
        best_set, cc_best = local_search(best_set, cc_best, rng, max_iter=100000)

    # Verify full coverage (safety)
    cc_final = np.zeros(N, dtype=np.int32)
    for c in best_set:
        cc_final[ball_arr[c]] += 1
    if cc_final.min() < 1:
        # repair any holes
        best_set, cc_final = greedy_repair(list(best_set), rng)

    return [all_words[c] for c in best_set]
