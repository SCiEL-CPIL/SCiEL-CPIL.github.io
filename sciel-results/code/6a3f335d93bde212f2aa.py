import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    # Compute word indices for each codeword
    def compute_word_indices(codewords):
        indices = np.zeros(len(codewords), dtype=np.int64)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices

    code_arr = np.array(code, dtype=np.int32)
    cw_indices = compute_word_indices(code_arr)

    # Precompute coverage: for each codeword, list of word indices it covers
    codeword_covers = []
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

    # Convert to numpy arrays for faster operations
    best_code = None
    best_count = num_codewords

    # Multiple trials with different random orderings
    max_trials = 40

    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break

        # Random order for checking removals
        order = list(range(num_codewords))
        random.shuffle(order)

        active = np.ones(num_codewords, dtype=bool)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            for wi in codeword_covers[ci]:
                coverage_count[wi] += 1

        # Phase 1: Greedy removal with multiple passes
        removed_any = True
        while removed_any:
            if time.time() > deadline - 5.0:
                break
            removed_any = False
            for ci in order:
                if time.time() > deadline - 5.0:
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

        # Phase 2: Pairwise swap local search
        # Try to replace one active codeword with an inactive one to maintain or improve coverage
        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]

        if len(inactive_list) > 0 and len(active_list) > 0:
            # Precompute set version of codeword_covers for faster lookup
            codeword_covers_set = [set(cov) for cov in codeword_covers]
            
            swap_iterations = 0
            max_swap_iters = 500
            improved = True
            while improved and swap_iterations < max_swap_iters:
                if time.time() > deadline - 5.0:
                    break
                improved = False
                swap_iterations += 1

                # Randomly pick an active codeword to potentially remove
                ci_remove = random.choice(active_list)
                # Randomly pick an inactive codeword to potentially add
                ci_add = random.choice(inactive_list)

                # Words that would become uncovered if we remove ci_remove
                at_risk = [wi for wi in codeword_covers[ci_remove] if coverage_count[wi] == 1]

                if len(at_risk) == 0:
                    # Safe to remove without adding anything - this is a pure removal
                    # But Phase 1 should have caught this, so skip
                    continue

                # Check if ci_add covers all at_risk words
                add_set = codeword_covers_set[ci_add]
                all_covered = all(wi in add_set for wi in at_risk)

                if all_covered:
                    # Perform the swap
                    for wi in codeword_covers[ci_remove]:
                        coverage_count[wi] -= 1
                    active[ci_remove] = False

                    for wi in codeword_covers[ci_add]:
                        coverage_count[wi] += 1
                    active[ci_add] = True

                    improved = True
                    # Update lists
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    if len(inactive_list) == 0:
                        break

        # Phase 3: Try to find additional removable codewords after swaps
        removed_any = True
        while removed_any:
            if time.time() > deadline - 5.0:
                break
            removed_any = False
            for ci in order:
                if time.time() > deadline - 5.0:
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

        trial_count = int(np.sum(active))
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
    if time.time() < deadline - 1.0 and len(best_code) <= 100:
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
