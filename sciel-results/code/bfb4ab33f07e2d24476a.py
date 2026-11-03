import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    N = q ** n
    sphere_size = 1 + n * (q - 1)

    # Precompute neighbor indices for every word.
    # Using little-endian base-q: digit j has weight q**j.
    weights = np.array([q ** j for j in range(n)], dtype=np.int64)

    # digits[i, j] = j-th digit (little-endian) of word i
    digits = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        x = i
        for j in range(n):
            digits[i, j] = x % q
            x //= q

    # neighbors[i] = array of indices within distance <= radius from i
    if radius == 1:
        # For each position j, each digit d != original: new_idx = i + (d - orig)*q**j
        neigh_arrays = []
        for i in range(N):
            arr = [i]
            w = digits[i]
            for j in range(n):
                base = w[j]
                wt = q ** j
                for d in range(q):
                    if d != base:
                        arr.append(i + (d - base) * wt)
            neigh_arrays.append(arr)
        neighbors = np.array(neigh_arrays, dtype=np.int32)
    else:
        # General radius (slower, but N is small for this task)
        neigh_arrays = []
        for i in range(N):
            arr = [i]
            seen = {i}
            w = digits[i].copy()
            for dist in range(1, radius + 1):
                for positions in _combinations(range(n), dist):
                    for vals in _product(range(q), repeat=dist):
                        if all(vals[k] == w[positions[k]] for k in range(dist)):
                            continue
                        new_w = w.copy()
                        for k, pos in enumerate(positions):
                            new_w[pos] = vals[k]
                        new_idx = int(np.dot(new_w, weights))
                        if new_idx not in seen:
                            seen.add(new_idx)
                            arr.append(new_idx)
            neigh_arrays.append(arr)
        neighbors = np.array(neigh_arrays, dtype=np.int32)

    # Inverse: for each word, which codewords cover it
    word_to_coders = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbors[c]:
            word_to_coders[w].append(c)

    def greedy_criticality():
        """Greedy that prioritizes the hardest uncovered word (fewest coders),
        then picks the codeword covering it with max new coverage."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        while uncovered.any():
            if time.time() > deadline:
                break
            # find uncovered word with minimum criticality
            unc_idx = np.nonzero(uncovered)[0]
            min_crit = N + 1
            target = -1
            for w in unc_idx:
                crit = len(word_to_coders[w])
                if crit < min_crit:
                    min_crit = crit
                    target = w
                    if min_crit == 1:
                        break
            if target == -1:
                break
            # pick best codeword covering target
            best_c = -1
            best_new = -1
            for c in word_to_coders[target]:
                if c in code_set:
                    continue
                new_count = int(uncovered[neighbors[c]].sum())
                if new_count > best_new:
                    best_new = new_count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            uncovered[neighbors[best_c]] = False
        return code

    def local_search(code):
        """Redundant removal + 2-opt swap to reduce size."""
        code = list(code)
        if not code:
            return code
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_counts[neighbors[c]] += 1
        code_set = set(code)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                c = code[i]
                # words uniquely covered by c
                newly_uncovered = [w for w in neighbors[c] if cover_counts[w] == 1]
                if not newly_uncovered:
                    # remove
                    cover_counts[neighbors[c]] -= 1
                    code.pop(i)
                    code_set.discard(c)
                    improved = True
                    break
                # try 2-opt: remove c, add one c2 that covers all newly_uncovered
                # and enables net size reduction (i.e., after adding c2, some other
                # codeword becomes removable). Simpler: find c2 covering all
                # newly_uncovered; if adding c2 then lets us drop >=2 others, do it.
                best_c2 = -1
                best_c2_gain = 0
                for c2 in word_to_coders[newly_uncovered[0]]:
                    if c2 in code_set:
                        continue
                    if not all(cover_counts[w] >= 1 and w in neighbors[c2] for w in newly_uncovered):
                        continue
                    # count how many existing codewords become removable after adding c2
                    # (i.e., their sphere is fully covered by others + c2)
                    # Approximate: count codewords c3 whose sphere has min coverage >1 after add.
                    # Cheap heuristic: count c3 where all words in neighbors[c3] have cover>=1
                    # and at least one word in neighbors[c3] is in neighbors[c2].
                    gain = 0
                    for c3 in code:
                        if c3 == c:
                            continue
                        # c3 removable if after removing c and adding c2, no word uniquely covered by c3
                        # min coverage over neighbors[c3] excluding c's contribution, including c2
                        cov = cover_counts[neighbors[c3]].copy()
                        cov -= (neighbors[c3] == c).astype(np.int32)  # remove c
                        cov += (neighbors[c3] == c2).astype(np.int32)  # add c2
                        if cov.min() > 1:
                            gain += 1
                    if gain > best_c2_gain:
                        best_c2_gain = gain
                        best_c2 = c2
                if best_c2 != -1 and best_c2_gain >= 1:
                    # apply: remove c, add c2, then remove one newly-removable codeword
                    cover_counts[neighbors[c]] -= 1
                    code.pop(i)
                    code_set.discard(c)
                    code.append(best_c2)
                    code_set.add(best_c2)
                    cover_counts[neighbors[best_c2]] += 1
                    # now remove one redundant
                    for j in range(len(code)):
                        cj = code[j]
                        if all(cover_counts[w] > 1 for w in neighbors[cj]):
                            cover_counts[neighbors[cj]] -= 1
                            code.pop(j)
                            code_set.discard(cj)
                            break
                    improved = True
                    break
        return code

    # Run greedy, then local search
    code = greedy_criticality()
    code = local_search(code)

    # If time remains, try a couple of randomized restarts to beat local minima
    if time.time() < deadline - 20 and len(code) > 0:
        rng = np.random.default_rng(0)
        best_code = list(code)
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            # randomized greedy: perturb by randomizing criticality tie-breaking
            uncovered = np.ones(N, dtype=bool)
            rcode = []
            code_set = set()
            while uncovered.any():
                if time.time() > deadline - 5:
                    break
                unc_idx = np.nonzero(uncovered)[0]
                # pick a few random candidates, choose one with low criticality
                k = min(5, len(unc_idx))
                sample = rng.choice(unc_idx, size=k, replace=False)
                target = min(sample, key=lambda w: len(word_to_coders[w]))
                best_c = -1
                best_new = -1
                for c in word_to_coders[target]:
                    if c in code_set:
                        continue
                    new_count = int(uncovered[neighbors[c]].sum())
                    if new_count > best_new:
                        best_new = new_count
                        best_c = c
                if best_c == -1:
                    break
                rcode.append(best_c)
                code_set.add(best_c)
                uncovered[neighbors[best_c]] = False
            rcode = local_search(rcode)
            if len(rcode) < len(best_code):
                best_code = rcode
        code = best_code

    # Convert indices to base-q digit lists (big-endian, standard representation)
    result = []
    for idx in code:
        word = [0] * n
        x = idx
        for j in range(n):
            word[n - 1 - j] = x % q
            x //= q
        result.append(word)

    return result


def _combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def _product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for r in result:
        yield tuple(r)
