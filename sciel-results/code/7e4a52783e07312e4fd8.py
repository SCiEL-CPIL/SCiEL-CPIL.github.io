"""Combined program: tetracode direct-sum start + coverage-count removal + swap local search + restarts."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s evaluator limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

    # Precompute index <-> word conversions
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (distance <= radius) for every word in the space
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = [i]  # include self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)

    # Also build a set version for fast membership tests
    neighbor_sets = [set(nb) for nb in neighbors]

    # Precompute: for each codeword index, a bitmask representation of its neighborhood
    # Since space_size=729, we can use a Python int as a bitmask (729 bits)
    # This enables fast intersection checks via bitwise AND
    nb_bitmasks = []
    for i in range(space_size):
        bm = 0
        for nb in neighbors[i]:
            bm |= (1 << nb)
        nb_bitmasks.append(bm)

    # Tetracode direct-sum initial construction: 9 * 3^(6-4) = 81 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]

    def build_counts(codeword_indices):
        """Build coverage count array from a list of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in codeword_indices:
            counts[neighbors[ci]] += 1
        return counts

    def greedy_remove(cw_indices, counts, deadline):
        """Iteratively remove redundant codewords. Returns (list, counts)."""
        current = list(cw_indices)
        cnt = counts.copy()
        while True:
            if time.time() > deadline:
                break
            removed = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ci = current[i]
                # Check if any neighbor has count == 1 (would become uncovered)
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    removed = True
            if not removed:
                break
        return current, cnt

    def local_search_swap(cw_indices, counts, deadline, max_swaps=20000):
        """
        Local search: try swaps to enable further deletions.
        For each codeword, find its 'essential' neighbors (count==1).
        Try replacing it with a candidate that covers all essential neighbors.
        Uses bitmask for fast coverage checks.
        """
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        swaps_made = 0

        # Build a bitmask of currently covered words (cnt > 0)
        # and track essential words per codeword efficiently

        while swaps_made < max_swaps and time.time() < deadline:
            progress = False

            # Phase 1: try simple removals
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
            if progress:
                continue

            # Phase 2: try swaps using bitmask acceleration
            # For each codeword, find essential neighbors
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                essential = [nb for nb in neighbors[ci] if cnt[nb] == 1]
                if not essential:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
                    break

                # Build essential bitmask for fast checking
                essential_bm = 0
                for e in essential:
                    essential_bm |= (1 << e)

                # Try random candidates to replace ci
                tried = 0
                max_tries = 50
                while tried < max_tries and time.time() < deadline:
                    tried += 1
                    cand = random.randint(0, space_size - 1)
                    if cand in current_set:
                        continue
                    # Fast check: does candidate cover all essential neighbors?
                    if (nb_bitmasks[cand] & essential_bm) != essential_bm:
                        continue
                    # Check: no word with cnt==1 in neighbors[ci] is lost
                    # i.e., all cnt==1 words in neighbors[ci] must be in neighbors[cand]
                    ci_essential_bm = 0
                    for nb in neighbors[ci]:
                        if cnt[nb] == 1:
                            ci_essential_bm |= (1 << nb)
                    if (nb_bitmasks[cand] & ci_essential_bm) != ci_essential_bm:
                        continue
                    # Perform swap
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    for nb in neighbors[cand]:
                        cnt[nb] += 1
                    current[i] = cand
                    current_set.discard(ci)
                    current_set.add(cand)
                    progress = True
                    swaps_made += 1
                    break
                if progress:
                    break

            if not progress:
                break

        return current, cnt

    def try_randomized_removal_order(cw_indices, counts, deadline, seed):
        """Try removal in a randomized order, then swap local search."""
        rng = random.Random(seed)
        current = list(cw_indices)
        cnt = counts.copy()

        # Randomized removal passes
        for _ in range(30):
            if time.time() > deadline:
                break
            rng.shuffle(current)
            removed = False
            i = 0
            while i < len(current):
                if time.time() > deadline:
                    break
                ci = current[i]
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    removed = True
                else:
                    i += 1
            if not removed:
                break

        # Then swap local search
        current, cnt = local_search_swap(current, cnt, deadline, max_swaps=5000)
        return current, cnt

    # Main search loop with multiple restarts
    best_size = float('inf')
    best_code = None

    # Initial: deterministic greedy remove + swap
    cw_indices = [word_to_idx(w) for w in initial_codewords]
    counts = build_counts(cw_indices)
    code, counts = greedy_remove(cw_indices, counts, deadline - 25)
    code, counts = local_search_swap(code, counts, deadline - 15, max_swaps=15000)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # Multiple randomized restarts
    remaining_deadline = deadline - 5
    restarts = 0
    while time.time() < remaining_deadline:
        restarts += 1
        seed = restarts * 12345 + 678
        code_r, counts_r = try_randomized_removal_order(cw_indices, counts, remaining_deadline, seed)
        if len(code_r) < best_size:
            best_size = len(code_r)
            best_code = code_r.copy()
        # If we've hit the known optimum, stop
        if best_size <= 73:
            break

    # Final polish: one more greedy remove + swap on the best
    if best_code is not None and time.time() < deadline - 2:
        final_counts = build_counts(best_code)
        final_code, final_counts = greedy_remove(best_code, final_counts, deadline - 1)
        final_code, final_counts = local_search_swap(final_code, final_counts, deadline, max_swaps=8000)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    # Convert back to list of lists
    result = [idx_to_word(ci) for ci in best_code]
    return result


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute coverage sets
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
