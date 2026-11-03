"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

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

        # Use a simple greedy: pick word covering most uncovered words.
        # To keep runtime bounded, we sample a subset when remaining set is large.
        remaining = set(range(total_words))

        while uncovered_count > 0:
            if len(remaining) > 2000:
                sample = random.sample(list(remaining), 2000)
            else:
                sample = list(remaining)

            best_word = -1
            best_cov = -1

            for w in sample:
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
                break

            codewords.append(best_word)
            remaining.discard(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1

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

        # Precompute neighbor arrays for faster set operations
        neighbor_arrays = [np.array(list(neighbors[w]), dtype=np.int32) for w in range(total_words)]
        neighbor_sets = [set(neighbors[w]) for w in range(total_words)]

        while time.time() - start_time < time_limit:
            # Periodically try to remove redundant codewords
            if time.time() - last_redundancy_check > 0.3:
                remove_redundant()
                last_redundancy_check = time.time()
                continue

            if not codewords:
                break

            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]

            # Words uniquely covered by old_cw
            old_neighbors = neighbor_sets[old_cw]
            would_uncover = [u for u in old_neighbors if coverage_count[u] == 1]

            if not would_uncover:
                for u in old_neighbors:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue

            # Candidate pool: neighbors of the first uniquely covered word
            first_uncovered = would_uncover[0]
            candidates = list(neighbor_sets[first_uncovered])
            random.shuffle(candidates)

            best_new_cw = -1
            best_gain = 0

            # Evaluate candidates; accept only if they cover all uniquely covered words
            would_uncover_set = set(would_uncover)
            old_neigh_set = old_neighbors

            for new_cw in candidates:
                if new_cw in cw_set or new_cw == old_cw:
                    continue

                # Must cover all uniquely covered words
                new_neigh_set = neighbor_sets[new_cw]
                if not would_uncover_set.issubset(new_neigh_set):
                    continue

                # Gain: number of words that gain redundancy (covered by both old and new)
                # minus words that lose redundancy
                # Net effect: words in new_neigh but not old_neigh gain +1 coverage
                # words in old_neigh but not new_neigh lose -1 coverage
                # We want to maximize words that become more redundantly covered
                # Actually, for a swap, the total count stays the same (we remove old, add new)
                # So we want to minimize unique coverage after the swap
                # Unique words after swap = words in (old_neigh - new_neigh) with count 1 after removing old
                # = words in (old_neigh - new_neigh) with original count 1
                # Plus words in (new_neigh - old_neigh) with original count 0 (but they were uncovered, impossible)
                
                # Better metric: minimize the number of words that would become uniquely covered
                # Words that lose old coverage: old_neigh - new_neigh
                # Of those, count how many have coverage_count == 1 (would become uncovered) -> invalid
                # Count how many have coverage_count == 2 (would become unique) -> bad
                # Words that gain new coverage: new_neigh - old_neigh
                # Of those, count how many have coverage_count == 1 (would become redundant) -> good
                
                lost_words = old_neigh_set - new_neigh_set
                gained_words = new_neigh_set - old_neigh_set
                
                # Check validity: no word in lost_words should have count 1
                valid = True
                for u in lost_words:
                    if coverage_count[u] == 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: prefer swaps that reduce unique coverage
                # unique_after = sum(1 for u in all if coverage_count[u] == 1 after swap)
                # Change in unique count:
                # - For u in lost_words: if count was 2, becomes 1 (unique) -> +1 unique
                # - For u in gained_words: if count was 1, becomes 2 (redundant) -> -1 unique
                # - For u in intersection: no change
                # - For u not in either: no change
                
                unique_increase = 0
                for u in lost_words:
                    if coverage_count[u] == 2:
                        unique_increase += 1
                for u in gained_words:
                    if coverage_count[u] == 1:
                        unique_increase -= 1
                
                # We want to minimize unique_increase (prefer negative)
                if -unique_increase > best_gain:
                    best_gain = -unique_increase
                    best_new_cw = new_cw

            if best_new_cw != -1:
                for u in old_neighbors:
                    coverage_count[u] -= 1
                for u in neighbor_sets[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)

        remove_redundant()
        return codewords

    # Multiple restarts with adaptive time allocation
    start_time = time.time()
    time_limit = 118
    best_codewords = []

    while time.time() - start_time < time_limit:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break

        # Give greedy a small budget, rest to optimization
        greedy_time = min(1.5, remaining_time * 0.1)
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
