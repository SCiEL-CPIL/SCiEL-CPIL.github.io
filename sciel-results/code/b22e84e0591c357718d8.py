import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # ---- Generic fallbacks for other parameters ----
    if not (q == 3 and n == 7 and radius == 1):
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # =================== q=3, n=7, radius=1 ===================
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute ball (neighborhood) of each word
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(sorted(nb), dtype=np.int32))

    # Precompute a boolean ball matrix for fast gain computation:
    # ball_matrix[c, w] = True iff w in ball(c)
    ball_matrix = np.zeros((N, N), dtype=bool)
    for c in range(N):
        ball_matrix[c, neighbors_list[c]] = True

    def greedy_construct(seed_val):
        """Randomized greedy set-cover. Returns (code_list, coverage, cod_set)."""
        rng = random.Random(seed_val)
        coverage = np.zeros(N, dtype=np.int32)
        code = []
        cod_set = set()

        # gain[c] = number of currently-uncovered words in ball(c)
        gain = ball_matrix.sum(axis=1).astype(np.int32)  # initially all uncovered
        covered_any = np.zeros(N, dtype=bool)

        while True:
            if time.time() > deadline - 30:
                break
            # pick best candidate with positive gain
            g = gain.copy()
            g[np.array(list(cod_set), dtype=np.int32)] = 0 if cod_set else g
            best_c = int(np.argmax(g))
            best_gain = int(g[best_c])
            if best_gain <= 0:
                break
            code.append(best_c)
            cod_set.add(best_c)
            ball = neighbors_list[best_c]
            coverage[ball] += 1
            covered_any[ball] = True
            # update gain: words that just became covered reduce gain of all balls containing them
            new_cov = np.where(coverage == 1)[0]  # words that transitioned 0->1
            gain -= ball_matrix[new_cov].sum(axis=0).astype(np.int32)

        return code, coverage, cod_set

    # ---------- Local search on a (code, coverage, cod_set) state ----------

    def polish(code, coverage, cod_set):
        """Iteratively remove redundants and do best 1-exchange until no improvement."""
        code = list(code)
        coverage = coverage.copy()
        cod_set = set(cod_set)

        def try_removals():
            # remove any redundant codeword (all its ball words have coverage >= 2)
            changed = False
            for c in list(code):
                if np.min(coverage[neighbors_list[c]]) >= 2:
                    coverage[neighbors_list[c]] -= 1
                    code.remove(c)
                    cod_set.remove(c)
                    changed = True
            return changed

        def try_swap():
            # find best 1-for-1 replacement that keeps validity
            # For each c in code, removing c leaves uncovered = words where coverage==1 in ball(c)
            # We need c' covering all those. Try to maximize extra coverage (reduce future redundancy).
            n_code = len(code)
            best = None  # (c, cprime, extra)
            for c in code:
                ball_c = neighbors_list[c]
                # words that become uncovered if c removed: coverage==1 within ball_c
                mask = (coverage == 1)
                uncov = np.where(mask[ball_c])[0]
                if uncov.size == 0:
                    continue
                uncov_words = ball_c[uncov]
                # candidate c' must cover all uncov_words
                # gain_extra[c'] = |ball(c') ∩ {words with coverage>=1}| (extra redundancy created)
                # We want c' that covers uncov_words and maximizes coverage of already-covered words.
                covered_mask = (coverage >= 1)
                # For all c', does it cover uncov_words?
                covers = np.all(ball_matrix[:, uncov_words], axis=1)
                # extra coverage = number of covered words in ball(c')
                extra = (ball_matrix & covered_mask[None, :]).sum(axis=1)
                # exclude c and existing codewords
                valid = covers & (~np.isin(np.arange(N), np.array(list(cod_set), dtype=np.int32)))
                valid[~covers] = False
                if not valid.any():
                    continue
                # choose c' maximizing extra
                cand = np.where(valid)[0]
                if cand.size == 0:
                    continue
                best_cprime = cand[np.argmax(extra[cand])]
                # accept if it adds redundancy (extra > 0) — helps future removals
                if best is None or extra[best_cprime] > best[2]:
                    best = (c, int(best_cprime), int(extra[best_cprime]))
            if best is not None and best[2] > 0:
                c, cprime, _ = best
                coverage[neighbors_list[c]] -= 1
                code.remove(c)
                cod_set.remove(c)
                coverage[neighbors_list[cprime]] += 1
                code.append(cprime)
                cod_set.add(cprime)
                return True
            return False

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            if try_removals():
                improved = True
            if try_swap():
                improved = True

        return code, coverage, cod_set

    def kick_2opt(code, coverage, cod_set):
        """Remove 2 random codewords, greedily repair. Returns True if size decreased."""
        if len(code) < 2:
            return False
        rng = random.Random()
        a, b = rng.sample(code, 2)
        temp_cov = coverage.copy()
        temp_cov[neighbors_list[a]] -= 1
        temp_cov[neighbors_list[b]] -= 1
        uncov = np.where(temp_cov == 0)[0]
        if uncov.size == 0:
            # both redundant together
            coverage[neighbors_list[a]] -= 1
            coverage[neighbors_list[b]] -= 1
            code.remove(a); code.remove(b)
            cod_set.discard(a); cod_set.discard(b)
            return True
        uncov_set = set(uncov.tolist())
        # greedy repair with up to 2 additions
        added = []
        remaining = set(uncov_set)
        for _ in range(2):
            if not remaining:
                break
            # best c covering most remaining
            cand = set()
            for w in remaining:
                cand.update(neighbors_list[w].tolist())
            best_c = -1
            best_cnt = 0
            for c in cand:
                if c in cod_set or c in (a, b) or c in added:
                    continue
                cnt = len(set(neighbors_list[c].tolist()) & remaining)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                if best_cnt == len(remaining):
                    break
            if best_c == -1 or best_cnt == 0:
                break
            added.append(best_c)
            remaining -= set(neighbors_list[best_c].tolist())
        if not remaining and len(added) < 2:
            # apply
            coverage[neighbors_list[a]] -= 1
            coverage[neighbors_list[b]] -= 1
            code.remove(a); code.remove(b)
            cod_set.discard(a); cod_set.discard(b)
            for c in added:
                coverage[neighbors_list[c]] += 1
                code.append(c)
                cod_set.add(c)
            return True
        return False

    # ---------- Main search loop ----------
    best_code = None
    best_size = float('inf')

    max_restarts = 6
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        code, coverage, cod_set = greedy_construct(1000 + restart * 17)

        # polish
        code, coverage, cod_set = polish(code, coverage, cod_set)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # iterated local search: kick to escape local minima, re-polish
        for _ in range(12):
            if time.time() > deadline - 8:
                break
            if not kick_2opt(code, coverage, cod_set):
                # try a single random swap kick
                if len(code) >= 1:
                    rng = random.Random()
                    c = rng.choice(code)
                    # remove and repair with 1
                    temp_cov = coverage.copy()
                    temp_cov[neighbors_list[c]] -= 1
                    uncov = np.where(temp_cov == 0)[0]
                    if uncov.size == 0:
                        coverage[neighbors_list[c]] -= 1
                        code.remove(c); cod_set.discard(c)
                    else:
                        remaining = set(uncov.tolist())
                        cand = set()
                        for w in remaining:
                            cand.update(neighbors_list[w].tolist())
                        best_c2 = -1
                        best_cnt = 0
                        for c2 in cand:
                            if c2 in cod_set or c2 == c:
                                continue
                            cnt = len(set(neighbors_list[c2].tolist()) & remaining)
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_c2 = c2
                        if best_c2 != -1 and best_cnt == len(remaining):
                            coverage[neighbors_list[c]] -= 1
                            code.remove(c); cod_set.discard(c)
                            coverage[neighbors_list[best_c2]] += 1
                            code.append(best_c2); cod_set.add(best_c2)
            # re-polish
            code, coverage, cod_set = polish(code, coverage, cod_set)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            if best_size <= 160:
                break
        if best_size <= 160:
            break

    if best_code is None:
        # fallback: full space
        best_code = list(range(N))

    result = [idx_to_tuple(c) for c in best_code]
    return result
