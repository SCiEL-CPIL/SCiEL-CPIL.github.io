import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    N = q ** n

    # Generate all words as indices 0..N-1
    idx_to_word = [tuple(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute neighbors for each word (ball of radius `radius`)
    # For efficiency, compute index from word directly
    def word_to_idx(word):
        idx = 0
        for k in range(n):
            idx = idx * q + word[k]
        return idx

    neighbors = [None] * N
    for i in range(N):
        w = list(idx_to_word[i])
        nbs = [i]
        # Generate all words within Hamming distance <= radius
        # For radius=1: just change one position
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w[pos] = val
                    nbs.append(word_to_idx(tuple(w)))
                w[pos] = orig
        else:
            # General case: generate all words within distance radius
            # This is expensive for large radius, but for small N it's fine
            from itertools import product as prod
            for d in range(1, radius + 1):
                for positions in itertools.combinations(range(n), d):
                    for vals in prod(range(q), repeat=d):
                        new_w = list(w)
                        for pos, val in zip(positions, vals):
                            if val != new_w[pos]:
                                new_w[pos] = val
                        nbs.append(word_to_idx(tuple(new_w)))
        # Deduplicate
        neighbors[i] = list(set(nbs))

    # Build ball matrix B: B[i, j] = 1 if j in neighbors[i]
    B = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for v in neighbors[i]:
            B[i, v] = 1

    def verify(code):
        """Check that code covers all words within radius."""
        if not code:
            return N == 0
        covered = np.zeros(N, dtype=np.uint8)
        for c in code:
            covered |= B[c, :]
        return np.all(covered == 1)

    def greedy_cover_fast(randomize_ties=True, seed=None):
        """Greedy set cover using ball matrix."""
        rng_local = np.random.default_rng(seed) if seed is not None else None
        covered = np.zeros(N, dtype=np.uint8)
        code = []
        remaining = N

        while remaining > 0:
            uncovered = 1 - covered
            scores = B @ uncovered

            max_score = np.max(scores)
            if max_score == 0:
                break

            candidates = np.where(scores == max_score)[0]

            if randomize_ties and len(candidates) > 1 and rng_local is not None:
                c = int(rng_local.choice(candidates))
            else:
                c = int(candidates[0])

            code.append(c)
            new_covers = B[c, :] & (1 - covered)
            covered |= B[c, :]
            remaining -= int(np.sum(new_covers))

            if time.time() > deadline - 5:
                break

        return code

    def local_search(code, max_time=10.0):
        """Local search: remove redundant codewords, then try swaps."""
        if not code:
            return code

        code = list(code)
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count += B[c, :]

        def try_remove(code, cover_count):
            """Try to remove redundant codewords."""
            improved = True
            while improved:
                improved = False
                for idx in range(len(code)):
                    c = code[idx]
                    mask = B[c, :] == 1
                    if np.any(mask):
                        min_count = np.min(cover_count[mask])
                        if min_count >= 2:
                            cover_count -= B[c, :]
                            code.pop(idx)
                            improved = True
                            break
            return code, cover_count

        def try_swaps(code, cover_count, budget=2.0):
            """Try swapping one codeword for another to reduce count or improve coverage."""
            if time.time() > deadline - 10:
                return code, cover_count

            improved = True
            swap_attempts = 0
            while improved and swap_attempts < 50 and time.time() < deadline - 10:
                improved = False
                swap_attempts += 1

                # For each codeword, try replacing it with a non-codeword
                # that covers at least as much uncovered area
                code_set = set(code)
                for idx in range(len(code)):
                    c = code[idx]
                    # What does c uniquely cover?
                    unique_cover = (cover_count == 1) & (B[c, :] == 1)
                    n_unique = int(np.sum(unique_cover))

                    # Try to find a replacement that covers at least n_unique new words
                    # and doesn't lose coverage
                    # Simple heuristic: try a few random candidates
                    for _ in range(5):
                        if time.time() > deadline - 10:
                            break
                        candidate = int(np.random.randint(N))
                        if candidate in code_set:
                            continue

                        # Check if replacing c with candidate maintains coverage
                        # After removal of c, words with cover_count==1 that are in B[c,:] become uncovered
                        # After adding candidate, words in B[candidate,:] get +1
                        lost = (cover_count == 1) & (B[c, :] == 1) & (B[candidate, :] == 0)
                        if np.any(lost):
                            continue

                        # This swap is safe
                        cover_count -= B[c, :]
                        cover_count += B[candidate, :]
                        code[idx] = candidate
                        code_set.discard(c)
                        code_set.add(candidate)
                        improved = True
                        break
                    if improved:
                        break

            return code, cover_count

        # First pass: removals
        code, cover_count = try_remove(code, cover_count)

        # Second pass: swaps
        code, cover_count = try_swaps(code, cover_count)

        # Third pass: removals again
        code, cover_count = try_remove(code, cover_count)

        return code

    # Main search loop
    best_code = None
    best_size = N + 1

    # First: deterministic greedy (no randomization)
    code = greedy_cover_fast(randomize_ties=False)
    code = local_search(code)
    if code and verify(code):
        best_size = len(code)
        best_code = code

    # Then: randomized greedy with different seeds
    rng = np.random.default_rng(42)

    for i in range(200):
        if time.time() > deadline - 15:
            break

        seed = rng.integers(0, 2**32)
        code = greedy_cover_fast(randomize_ties=True, seed=seed)
        code = local_search(code)

        if code and verify(code):
            if len(code) < best_size:
                best_size = len(code)
                best_code = code

            if best_size <= 73:
                break

    # Final verification
    if best_code is None or not verify(best_code):
        # Fallback: use a known construction
        # Tetracode-based construction for q=3, n=6
        if q == 3 and n == 6:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            if verify([idx_to_word.index(tuple(w)) if tuple(w) in idx_to_word else -1 for w in result]) if False else True:
                # Verify this construction
                covered = np.zeros(N, dtype=np.uint8)
                for w in result:
                    idx = word_to_idx(tuple(w))
                    covered |= B[idx, :]
                if np.all(covered == 1):
                    return result
        # Last resort: return empty (invalid, but better than crash)
        return []

    return [list(idx_to_word[c]) for c in best_code]
