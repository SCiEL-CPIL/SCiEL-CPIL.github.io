import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before the 120s kill

    total_words = q ** n
    cover_count = 1 + n * (q - 1)  # size of the Hamming ball

    # ---------- index <-> word ----------
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # ---------- neighbors (Hamming ball of each word) ----------
    neighbors = []          # neighbors[i] = list of word indices covered by codeword i
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_word(i)
        idxs = [i]
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                w[pos] = nv
                # compute index of modified word
                j = i
                # recompute via base-q
                digits = [0] * n
                tmp = i
                for k in range(n - 1, -1, -1):
                    digits[k] = tmp % q
                    tmp //= q
                digits[pos] = nv
                j = 0
                for k in range(n):
                    j = j * q + digits[k]
                idxs.append(j)
            w[pos] = orig
        neighbors.append(idxs)
        for j, v in enumerate(idxs):
            neighbors_arr[i, j] = v

    # ---------- reverse cover: which codewords cover each word ----------
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for v in neighbors[i]:
            reverse_cover[v].append(i)

    # ---------- greedy construction (vectorized gain via numpy) ----------
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        while uncovered.any():
            if time.time() > deadline:
                return None
            # gain = number of currently-uncovered words each candidate covers
            gains = uncovered[neighbors_arr].sum(axis=1)
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = np.where(gains == max_gain)[0]
            chosen = int(random.choice(best.tolist()))
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        return code

    # ---------- reduction (remove redundant codewords) ----------
    def reduce_cover(code):
        if not code:
            return code
        cover_count_arr = np.zeros(total_words, dtype=np.int16)
        for c in code:
            cover_count_arr[neighbors_arr[c, :]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                cover_count_arr[neighbors_arr[c, :]] -= 1
                if np.any(cover_count_arr == 0):
                    cover_count_arr[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    # ---------- local search: 1-1 swap, 2-1 move, plus removal ----------
    def local_search(code, budget):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True

        def critical_words(c):
            nb = neighbors_arr[c, :]
            return nb[counts[nb] == 1]

        def find_replacement(crit, exclude):
            # intersection of reverse_cover over critical words
            if not len(crit):
                return None
            cand = None
            for cw in crit:
                rc = reverse_cover[cw]
                if cand is None:
                    cand = set(rc)
                else:
                    cand.intersection_update(rc)
                if cand is None or len(cand) == 0:
                    return None
            if exclude:
                for e in exclude:
                    cand.discard(e)
            if len(cand) == 0:
                return None
            return random.choice(list(cand))

        improved = True
        while improved and time.time() < budget:
            improved = False
            order = list(range(len(code)))
            random.shuffle(order)

            # ---- 1-1: try remove / swap each codeword ----
            for i in order:
                if time.time() > budget:
                    break
                c = code[i]
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    crit = critical_words(c)
                    rep = find_replacement(crit, exclude=[c])
                    if rep is not None and not in_code[rep]:
                        # perform swap
                        counts[neighbors_arr[c, :]] -= 0  # already removed above? recompute
                        # we currently have c removed (counts decremented). Add rep.
                        counts[neighbors_arr[rep, :]] += 1
                        in_code[c] = False
                        in_code[rep] = True
                        code[i] = rep
                        improved = True
                        # quick reduction pass after swap
                        for j in range(len(code) - 1, -1, -1):
                            cj = code[j]
                            counts[neighbors_arr[cj, :]] -= 1
                            if np.any(counts == 0):
                                counts[neighbors_arr[cj, :]] += 1
                            else:
                                counts[neighbors_arr[cj, :]] += 1
                                in_code[cj] = False
                                code.pop(j)
                                break
                        break
                    else:
                        counts[neighbors_arr[c, :]] += 1  # revert
                else:
                    # c is redundant: remove it
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # ---- 2-1 move: remove two codewords, add one that covers both critical sets ----
            if len(code) >= 2:
                pairs = list(range(len(code)))
                random.shuffle(pairs)
                found = False
                for a in range(len(code)):
                    if time.time() > budget:
                        break
                    for b in range(a + 1, len(code)):
                        ca = code[a]
                        cb = code[b]
                        counts[neighbors_arr[ca, :]] -= 1
                        counts[neighbors_arr[cb, :]] -= 1
                        if np.any(counts == 0):
                            counts[neighbors_arr[ca, :]] += 1
                            counts[neighbors_arr[cb, :]] += 1
                            # union of critical words
                            nba = neighbors_arr[ca, :]
                            nbb = neighbors_arr[cb, :]
                            ca_crit = set(nba[counts[nba] == 1].tolist())
                            cb_crit = set(nbb[counts[nbb] == 1].tolist())
                            union = ca_crit | cb_crit
                            rep = find_replacement(list(union), exclude=[ca, cb])
                            if rep is not None and not in_code[rep]:
                                counts[neighbors_arr[rep, :]] += 1
                                in_code[ca] = False
                                in_code[cb] = False
                                in_code[rep] = True
                                code.pop(b)
                                code.pop(a)
                                code.append(rep)
                                found = True
                                improved = True
                                break
                            else:
                                counts[neighbors_arr[ca, :]] += 1
                                counts[neighbors_arr[cb, :]] += 1
                        else:
                            # both redundant together: remove both
                            in_code[ca] = False
                            in_code[cb] = False
                            code.pop(b)
                            code.pop(a)
                            found = True
                            improved = True
                            break
                    if found:
                        break

            # ---- perturb/repair: escape local minima by random perturbation ----
            if not improved and len(code) > 0 and time.time() < budget - 3:
                # drop a random codeword, then repair with greedy additions
                k = random.randrange(len(code))
                c = code[k]
                counts[neighbors_arr[c, :]] -= 1
                in_code[c] = False
                code.pop(k)
                # repair: add codewords until covered again
                covered = np.zeros(total_words, dtype=np.bool_)
                for cc in code:
                    covered[neighbors_arr[cc, :]] = True
                guard = 0
                while not covered.all() and guard < 50 and time.time() < budget - 2:
                    uncovered = ~covered
                    gains = uncovered[neighbors_arr].sum(axis=1)
                    mx = gains.max()
                    if mx <= 0:
                        break
                    best = np.where(gains == mx)[0]
                    ch = int(random.choice(best.tolist()))
                    code.append(ch)
                    in_code[ch] = True
                    covered[neighbors_arr[ch, :]] = True
                    counts[neighbors_arr[ch, :]] += 1
                    guard += 1
                improved = True

        return code

    # ---------- main loop ----------
    best_solution = None
    best_size = float('inf')

    def evaluate(seed):
        nonlocal best_solution, best_size
        code = greedy_construct(seed)
        if code is None:
            return
        code = reduce_cover(code)
        if time.time() < deadline - 8:
            code = local_search(code, deadline - 4)
            code = reduce_cover(code)
        if len(code) < best_size:
            best_size = len(code)
            best_solution = list(code)

    seed = 0
    while time.time() < deadline:
        evaluate(seed)
        seed += 1

    if best_solution is None:
        # fallback: tetracode-based construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
    return result
