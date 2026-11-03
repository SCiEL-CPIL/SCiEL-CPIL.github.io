"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Fast greedy (candidate sampling from uncovered-word neighborhoods) followed by
    randomized local search that swaps/removes codewords to maximize redundant coverage,
    with periodic redundancy removal.
    """
    if n == 0:
        return [[]]

    total_words = q ** n

    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbor sets (word -> set of words within radius).
    neighbors = [None] * total_words
    for w in range(total_words):
        neigh = [w]
        if radius >= 1:
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
        if radius >= 2:
            word = int_to_word(w, n, q)
            for pos1 in range(n):
                c1 = word[pos1]
                for val1 in range(q):
                    if val1 == c1:
                        continue
                    word[pos1] = val1
                    for pos2 in range(pos1 + 1, n):
                        c2 = word[pos2]
                        for val2 in range(q):
                            if val2 == c2:
                                continue
                            word[pos2] = val2
                            neigh.append(word_to_int(word, q))
                        word[pos2] = c2
                word[pos1] = c1
        neighbors[w] = set(neigh)

    def greedy_init():
        """Fast greedy: candidates drawn from neighborhoods of uncovered words."""
        covered = bytearray(total_words)
        uncovered = set(range(total_words))
        codewords = []

        while uncovered:
            if len(uncovered) > 600:
                sample = random.sample(list(uncovered), 60)
            else:
                sample = list(uncovered)

            candidates = set()
            for u in sample:
                candidates.update(neighbors[u])
            if not candidates:
                break

            best_word = -1
            best_cov = 0
            for w in candidates:
                cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_word = w
                    if best_cov == len(uncovered):
                        break

            if best_word == -1:
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered.discard(u)

        return codewords

    def optimize(codewords, time_limit):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        cw_set = set(codewords)

        def remove_redundant():
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        break

        remove_redundant()

        start_time = time.time()
        last_redundancy_check = start_time

        while time.time() - start_time < time_limit:
            if time.time() - last_redundancy_check > 0.25:
                remove_redundant()
                last_redundancy_check = time.time()
                continue
            if not codewords:
                break

            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            old_neigh = neighbors[old_cw]

            would_uncover = [u for u in old_neigh if coverage_count[u] == 1]
            if not would_uncover:
                for u in old_neigh:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue

            # Candidate pool: neighbors of the first uniquely-covered word.
            first_uncovered = would_uncover[0]
            candidates = list(neighbors[first_uncovered])
            random.shuffle(candidates)
            candidates = candidates[:400]

            best_new_cw = -1
            best_gain = 0
            for new_cw in candidates:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                new_neigh = neighbors[new_cw]
                ok = True
                for u in would_uncover:
                    if u not in new_neigh:
                        ok = False
                        break
                if not ok:
                    continue
                gain = 0
                for u in new_neigh:
                    if u not in old_neigh:
                        if coverage_count[u] >= 1:
                            gain += 1
                    else:
                        if coverage_count[u] == 2:
                            gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw

            if best_new_cw != -1:
                for u in old_neigh:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)

        remove_redundant()
        return codewords

    start_time = time.time()
    time_limit = 115
    best_codewords = []

    while time.time() - start_time < time_limit:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break
        greedy_time = min(1.0, remaining_time * 0.1)
        opt_time = remaining_time - greedy_time - 2
        if opt_time <= 0:
            break

        cw = greedy_init()
        cw = optimize(cw, opt_time)
        if not best_codewords or len(cw) < len(best_codewords):
            best_codewords = cw

        if time.time() - start_time > time_limit - 5:
            break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
