import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    total_words = q ** n

    # Precompute all words as a 2D numpy array (total_words x n)
    # Each row is a word in {0..q-1}^n
    all_words = np.zeros((total_words, n), dtype=np.int32)
    for idx in range(total_words):
        val = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = val % q
            val //= q

    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def covers_indices(codeword):
        """Return array of word indices covered by a codeword (itself + all 1-distance neighbors)."""
        c = np.array(codeword, dtype=np.int32)
        indices = [word_to_index(tuple(c))]
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                c[pos] = val
                indices.append(word_to_index(tuple(c)))
        c[pos] = orig
        return np.array(indices, dtype=np.int32)

    def build_coverage_matrices(code_list):
        """Build coverage matrix (m x total_words) uint8 and count array."""
        m = len(code_list)
        cov = np.zeros((m, total_words), dtype=np.uint8)
        for ci, c in enumerate(code_list):
            idxs = covers_indices(c)
            cov[ci, idxs] = 1
        return cov

    # Strategy: start with a diverse set of initial codewords, then greedily remove redundant ones.
    # Use multiple random initial sets and keep the best result.

    best_code = None
    best_count = total_words  # worst case: all words

    # Precompute: for the greedy removal, we need to quickly check if removing codeword ci
    # leaves all its covered words still covered by others.
    # We use a count matrix: count[wi] = number of active codewords covering word wi.
    # Codeword ci is redundant iff for all wi in covers(ci), count[wi] > 1.

    max_restarts = 200
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break

        # Generate initial code: a mix of structured and random codewords
        # Start with all-zero word plus random words to ensure good coverage
        initial_size = 200  # start with plenty of codewords
        init_set = set()

        # Always include all-zero
        init_set.add(tuple([0] * n))

        # Add structured codewords: variations on the tetracode idea
        for a in range(q):
            for b in range(q):
                if len(init_set) >= initial_size:
                    break
                head = [a, b, (a + b) % q, (a + 2 * b) % q]
                tail = [rng.randint(0, q - 1) for _ in range(n - 4)]
                w = tuple(head + tail)
                init_set.add(w)

        # Fill rest with random words
        while len(init_set) < initial_size:
            w = tuple(rng.randint(0, q - 1) for _ in range(n))
            init_set.add(w)

        code_list = list(init_set)
        m = len(code_list)

        # Build coverage matrix
        cov = build_coverage_matrices(code_list)

        # count[wi] = number of active codewords covering wi
        count = cov.sum(axis=0).astype(np.int32)

        # Precompute for each codeword, the list of word indices it covers
        covers_list = []
        for ci in range(m):
            idxs = np.where(cov[ci] == 1)[0]
            covers_list.append(idxs)

        active = np.ones(m, dtype=np.bool_)
        active_count = m

        # Greedy removal: repeatedly find and remove a redundant codeword
        # A codeword ci is redundant if for all wi in covers_list[ci], count[wi] > 1
        # i.e., min(count[covers_list[ci]]) > 1

        # Use a priority: prefer removing codewords whose minimum coverage count is largest
        # (most redundant first)

        for _pass in range(30):
            if time.time() > deadline - 15.0:
                break

            removed_any = False

            # Try each active codeword in random order
            active_indices = np.where(active)[0]
            rng.shuffle(active_indices.tolist())

            for ci in active_indices:
                if not active[ci]:
                    continue

                # Check redundancy: all words covered by ci must have count > 1
                ci_covers = covers_list[ci]
                if len(ci_covers) == 0:
                    # Degenerate case, remove it
                    active[ci] = False
                    count[cov[ci] == 1] -= 1
                    active_count -= 1
                    removed_any = True
                    continue

                min_count = count[ci_covers].min()
                if min_count > 1:
                    # Remove codeword ci
                    active[ci] = False
                    count[cov[ci] == 1] -= 1
                    active_count -= 1
                    removed_any = True

            if not removed_any:
                break

        if active_count < best_count:
            best_count = active_count
            best_code = [code_list[i] for i in range(m) if active[i]]

        # If we've hit a very good solution, do a final polish
        if best_count <= 80:
            # Try more restarts from the current best as a seed
            if time.time() > deadline - 20.0:
                break

    # Final verification and return
    if best_code is None:
        # Fallback: return all words (should not happen)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Verify coverage (safety check)
    final_cov = build_coverage_matrices(best_code)
    if not final_cov.any(axis=0).all():
        # Coverage failed, add missing words
        uncovered = np.where(~final_cov.any(axis=0))[0]
        for wi in uncovered:
            best_code.append(list(all_words[wi]))

    return [list(c) for c in best_code]
