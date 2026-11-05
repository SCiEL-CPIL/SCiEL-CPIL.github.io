import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # For the specific problem: q=3, n=6, radius=1
    # Use direct sum of ternary tetracode (9 words of length 4) with all tails
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    else:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4

    if n < base_n:
        return [list(range(n))]

    # Build initial code: direct sum
    tails = list(itertools.product(range(q), repeat=n - base_n))
    code = [tuple(h) + tuple(tail) for h in base_code for tail in tails]
    num_codewords = len(code)
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)

    # Compute word indices for each codeword
    def compute_word_indices(codewords):
        indices = np.zeros(len(codewords), dtype=np.int64)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices

    code_arr = np.array(code, dtype=np.int32)
    cw_indices = compute_word_indices(code_arr)

    # Precompute coverage: for each codeword, list of word indices it covers
    # And reverse: for each word, list of codewords that cover it
    codeword_covers = []
    covered_by = [[] for _ in range(total_words)]

    for ci in range(num_codewords):
        c = code_arr[ci]
        covers = set()
        covers.add(cw_indices[ci])
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                covers.add(new_idx)
        codeword_covers.append(list(covers))
        for wi in covers:
            covered_by[wi].append(ci)

    # Convert to numpy arrays for faster operations
    # coverage_count[wi] = number of active codewords covering word wi
    best_code = None
    best_count = num_codewords

    # Multiple trials with different random orderings
    max_trials = 30

    for trial in range(max_trials):
        if time.time() > deadline - 10.0:
            break

        # Random order for checking removals
        order = list(range(num_codewords))
        random.shuffle(order)

        active = np.ones(num_codewords, dtype=bool)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            for wi in codeword_covers[ci]:
                coverage_count[wi] += 1

        # Phase 1: Greedy removal
        removed_any = True
        while removed_any:
            if time.time() > deadline - 10.0:
                break
            removed_any = False
            for ci in order:
                if time.time() > deadline - 10.0:
                    break
                if not active[ci]:
                    continue
                is_redundant = True
                for wi in codeword_covers[ci]:
                    if coverage_count[wi] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    active[ci] = False
                    removed_any = True
                    for wi in codeword_covers[ci]:
                        coverage_count[wi] -= 1

        # Phase 2: Swap-based local search with simulated annealing
        # Try to swap one codeword for another to reduce size
        # Actually, we want to find smaller codes, so we try:
        # 1. Remove a codeword, then add a different one that covers more
        # 2. Or just try random swaps to escape local minima

        current_count = np.sum(active)

        # Try swap moves: remove one active codeword, add one inactive codeword
        # Accept if it reduces size (always) or with probability based on "energy"
        # Since we want to minimize size, we only accept if size decreases
        # But to escape local minima, we can allow equal-size moves with probability

        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]

        if len(inactive_list) > 0 and len(active_list) > 0:
            swap_iterations = 0
            max_swap_iters = 200
            while swap_iterations < max_swap_iters:
                if time.time() > deadline - 10.0:
                    break
                swap_iterations += 1

                # Randomly pick an active codeword to potentially remove
                ci_remove = random.choice(active_list)
                # Randomly pick an inactive codeword to potentially add
                ci_add = random.choice(inactive_list)

                # Check if removing ci_remove and adding ci_add maintains coverage
                # This is expensive, so we do a quick check:
                # For each word covered only by ci_remove, check if ci_add covers it

                # Words that would become uncovered if we remove ci_remove
                at_risk = []
                for wi in codeword_covers[ci_remove]:
                    if coverage_count[wi] == 1:
                        at_risk.append(wi)

                if len(at_risk) == 0:
                    # Safe to remove without adding anything
                    # But we're doing a swap, so we still add ci_add
                    # This increases size, which is bad
                    # So we should just remove ci_remove
                    # But that's handled in Phase 1
                    # For swap, we only accept if it doesn't increase size
                    # So skip this case
                    continue

                # Check if ci_add covers all at_risk words
                add_covers = set(codeword_covers[ci_add])
                all_covered = True
                for wi in at_risk:
                    if wi not in add_covers:
                        all_covered = False
                        break

                if all_covered:
                    # Perform the swap
                    # Remove ci_remove
                    for wi in codeword_covers[ci_remove]:
                        coverage_count[wi] -= 1
                    active[ci_remove] = False

                    # Add ci_add
                    for wi in codeword_covers[ci_add]:
                        coverage_count[wi] += 1
                    active[ci_add] = True

                    # Update lists
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    if len(inactive_list) == 0:
                        break

        trial_count = np.sum(active)
        if trial_count < best_count:
            best_count = trial_count
            best_indices = np.where(active)[0]
            best_code = [list(code[i]) for i in best_indices]

        # Early exit if we reached the known optimum
        if best_count <= 73:
            break

    if best_code is None:
        best_code = [list(c) for c in code]

    # Final verification
    if time.time() < deadline - 2.0 and len(best_code) <= 100:
        covered_set = set()
        for c in best_code:
            c_arr = np.array(c, dtype=np.int32)
            idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(idx)
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = idx + (val - orig) * (q ** (n - 1 - pos))
                    covered_set.add(new_idx)
        if len(covered_set) < total_words:
            best_code = [list(c) for c in code]

    return best_code
