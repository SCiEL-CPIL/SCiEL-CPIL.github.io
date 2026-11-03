"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a fast greedy construction followed by randomized local search that swaps
    codewords to maximize redundant coverage, with periodic redundancy removal.
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

    # Precompute neighbor sets once
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
        covered = [False] * total_words
        codewords = []
        uncovered_count = total_words
        uncovered_set = set(range(total_words))

        while uncovered_count > 0:
            # Sample from neighborhoods of uncovered words for better candidates
            if len(uncovered_set) > 500:
                sample_words = random.sample(list(uncovered_set), 500)
            else:
                sample_words = list(uncovered_set)

            # Collect candidate codewords: neighbors of sampled uncovered words
            candidate_set = set()
            for w in sample_words:
                for u in neighbors[w]:
                    candidate_set.add(u)

            if not candidate_set:
                break

            candidates = list(candidate_set)
            if len(candidates) > 3000:
                candidates = random.sample(candidates, 3000)

            best_word = -1
            best_cov = -1

            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_cov:
                    best_cov = new_cov
                    best_word = w
                    if new_cov == uncovered_count:
                        break

            if best_word == -1:
                # Fallback: pick a random uncovered word
                best_word = random.choice(list(uncovered_set))
                best_cov = 0

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
                    uncovered_set.discard(u)

        return codewords

    def optimize(codewords, time_limit):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1

        cw_set = set(codewords)

        def remove_redundant():
            removed_any = False
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
                        removed_any = True
                        break
            return removed_any

        remove_redundant()

        start_time = time.time()
        last_redundancy_check = start_time
        no_improve_count = 0
        max_no_improve = 500

        while time.time() - start_time < time_limit:
            # Periodically try to remove redundant codewords
            if time.time() - last_redundancy_check > 0.3:
                if remove_redundant():
                    no_improve_count = 0
                last_redundancy_check = time.time()
                continue

            if not codewords:
                break

            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]

            # Words uniquely covered by old_cw
            would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]

            if not would_uncover:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                no_improve_count = 0
                continue

            # Candidate pool: intersection of neighbors of all uniquely covered words
            # Start with neighbors of the first, intersect with others
            candidate_set = set(neighbors[would_uncover[0]])
            for u in would_uncover[1:]:
                candidate_set &= neighbors[u]
                if not candidate_set:
                    break

            if not candidate_set:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Try a random perturbation: remove and re-add
                    break
                continue

            candidates = list(candidate_set)
            # Remove existing codewords and the old one
            candidates = [c for c in candidates if c not in cw_set and c != old_cw]
            random.shuffle(candidates)

            best_new_cw = -1
            best_gain = 0

            # Evaluate candidates
            old_neigh = neighbors[old_cw]
            for new_cw in candidates:
                new_neigh = neighbors[new_cw]

                # Gain: words newly covered or gaining redundancy minus redundancy lost
                gain = 0
                for u in new_neigh:
                    if u in old_neigh:
                        if coverage_count[u] == 2:
                            gain -= 1
                    else:
                        if coverage_count[u] >= 1:
                            gain += 1

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
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > max_no_improve:
                # Perturb: remove a random codeword that has redundancy
                removable = [i for i in range(len(codewords))
                            if all(coverage_count[u] > 1 for u in neighbors[codewords[i]])]
                if removable:
                    i = random.choice(removable)
                    for u in neighbors[codewords[i]]:
                        coverage_count[u] -= 1
                    cw_set.discard(codewords[i])
                    codewords.pop(i)
                    no_improve_count = 0

        remove_redundant()
        return codewords

    # Multiple restarts with adaptive time allocation
    start_time = time.time()
    time_limit = 115
    best_codewords = []

    while time.time() - start_time < time_limit:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break

        # Give greedy a small budget, rest to optimization
        greedy_time = min(2.0, remaining_time * 0.08)
        opt_time = remaining_time - greedy_time - 2
        if opt_time <= 0:
            break

        cw = greedy_init()
        cw = optimize(cw, opt_time)

        if not best_codewords or len(cw) < len(best_codewords):
            best_codewords = cw

        # If we are close to the time limit, stop early to ensure safe return
        if time.time() - start_time > time_limit - 5:
            break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
