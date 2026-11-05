import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # Build the direct-sum construction
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]

    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))

    m = len(code)  # 81 for q=3, n=6
    total_words = q ** n

    # Precompute word indices
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword
    covers = []
    cover_sets = []
    for c in code:
        c_set = set()
        c_set.add(word_to_index(c))
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                c_set.add(word_to_index(tuple(c_list)))
            c_list[pos] = orig_val
        covers.append(np.array(sorted(c_set), dtype=np.int32))
        cover_sets.append(c_set)

    # Precompute coverage_count
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in range(m):
        coverage_count[covers[ci]] += 1

    # Precompute redundancy score: number of words that would be uncovered if this codeword were removed
    # A codeword is "redundant" if all words it covers are covered by at least one other codeword
    def compute_redundancy_scores(active):
        """Compute for each active codeword how many words it uniquely covers."""
        scores = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            if not active[ci]:
                continue
            unique_count = np.sum(coverage_count[covers[ci]] == 1)
            scores[ci] = unique_count
        return scores

    best_code = code[:]
    best_count = m

    def greedy_remove(active, curr_coverage, order=None):
        """Perform greedy removal pass. Returns modified active and curr_coverage."""
        if order is None:
            order = np.random.permutation(m)
        for ci in order:
            if not active[ci]:
                continue
            if np.all(curr_coverage[covers[ci]] > 1):
                active[ci] = False
                curr_coverage[covers[ci]] -= 1
        return active, curr_coverage

    def greedy_remove_by_redundancy(active, curr_coverage):
        """Remove codewords in order of increasing redundancy (most redundant first)."""
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Find codewords with minimum unique coverage
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            min_unique = np.inf
            candidates = []
            for ci in active_indices:
                unique = np.sum(curr_coverage[covers[ci]] == 1)
                if unique < min_unique:
                    min_unique = unique
                    candidates = [ci]
                elif unique == min_unique:
                    candidates.append(ci)
            # Try to remove one of the most redundant codewords
            random.shuffle(candidates)
            for ci in candidates:
                if np.all(curr_coverage[covers[ci]] > 1):
                    active[ci] = False
                    curr_coverage[covers[ci]] -= 1
                    improved = True
                    break
        return active, curr_coverage

    def try_add_remove(active, curr_coverage, inactive_indices, num_tries=5):
        """Try adding random inactive codewords then greedily removing. Returns best found."""
        best_local_active = active.copy()
        best_local_count = int(np.sum(active))
        best_local_coverage = curr_coverage.copy()

        for _ in range(num_tries):
            if time.time() > deadline - 3.0:
                break
            ri = inactive_indices[random.randint(0, len(inactive_indices) - 1)]
            if active[ri]:
                continue

            saved_coverage = curr_coverage.copy()
            saved_active = active.copy()

            active[ri] = True
            curr_coverage[covers[ri]] += 1

            order2 = np.random.permutation(m)
            active, curr_coverage = greedy_remove(active, curr_coverage, order2)
            # Also try redundancy-based removal
            active, curr_coverage = greedy_remove_by_redundancy(active, curr_coverage)

            cnt = int(np.sum(active))
            if cnt < best_local_count:
                best_local_count = cnt
                best_local_active = active.copy()
                best_local_coverage = curr_coverage.copy()

            curr_coverage = saved_coverage
            active = saved_active

        return best_local_active, best_local_coverage, best_local_count

    def try_swaps(active, curr_coverage, max_rounds=3):
        """Try swap-based local search. Returns modified active and curr_coverage."""
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            improved = False
            active_list = np.where(active)[0]
            inactive_list = np.where(~active)[0]

            if len(inactive_list) == 0:
                break

            num_swap_tries = min(50, len(active_list) * len(inactive_list))
            for _ in range(num_swap_tries):
                if time.time() > deadline - 3.0:
                    break
                ai = active_list[random.randint(0, len(active_list) - 1)]
                ri = inactive_list[random.randint(0, len(inactive_list) - 1)]

                ai_unique = covers[ai][curr_coverage[covers[ai]] == 1]
                if len(ai_unique) == 0:
                    active[ai] = False
                    active[ri] = True
                    curr_coverage[covers[ai]] -= 1
                    curr_coverage[covers[ri]] += 1
                    improved = True
                else:
                    ri_set = cover_sets[ri]
                    if all(wi in ri_set for wi in ai_unique):
                        active[ai] = False
                        active[ri] = True
                        curr_coverage[covers[ai]] -= 1
                        curr_coverage[covers[ri]] += 1
                        improved = True

            if not improved:
                order3 = np.random.permutation(m)
                active, curr_coverage = greedy_remove(active, curr_coverage, order3)
                active, curr_coverage = greedy_remove_by_redundancy(active, curr_coverage)
            else:
                # After swaps, do another greedy pass
                order3 = np.random.permutation(m)
                active, curr_coverage = greedy_remove(active, curr_coverage, order3)
                active, curr_coverage = greedy_remove_by_redundancy(active, curr_coverage)

            if not improved:
                break

        return active, curr_coverage

    max_restarts = 50000  # Will break based on time

    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break

        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active = np.ones(m, dtype=bool)

        # Phase 1: Initial greedy removal with random order
        order = np.random.permutation(m)
        active, curr_coverage = greedy_remove(active, curr_coverage, order)
        # Also try redundancy-based removal
        active, curr_coverage = greedy_remove_by_redundancy(active, curr_coverage)

        # Phase 2: Try add-then-remove
        inactive_indices = np.where(~active)[0]
        if len(inactive_indices) > 0:
            active, curr_coverage, _ = try_add_remove(active, curr_coverage, inactive_indices, num_tries=10)

        # Phase 3: Swap-based local search
        active, curr_coverage = try_swaps(active, curr_coverage, max_rounds=8)

        # Phase 4: Final greedy pass
        order_final = np.random.permutation(m)
        active, curr_coverage = greedy_remove(active, curr_coverage, order_final)
        active, curr_coverage = greedy_remove_by_redundancy(active, curr_coverage)

        cnt = int(np.sum(active))
        if cnt < best_count:
            best_count = cnt
            best_code = [code[i] for i in range(m) if active[i]]

    # Convert to lists for output
    return [list(c) for c in best_code]
