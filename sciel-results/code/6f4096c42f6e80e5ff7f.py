import time
import numpy as np
import random
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = 113.0  # Safety margin before 120s limit

    space_size = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute neighbors
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    neighbors = np.empty(space_size, dtype=object)
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)

    # Precompute cover_map: for each word w, list of codewords that cover it
    cover_map = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w in neighbors[c]:
            cover_map[w].append(c)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_cover(seed):
        """Greedy set cover with heap."""
        rng = random.Random(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        in_code = np.zeros(space_size, dtype=bool)

        # Initial counts
        counts = np.zeros(space_size, dtype=np.int32)
        for i in range(space_size):
            counts[i] = ball_size  # each word is covered by ball_size codewords initially

        heap = [(-ball_size, i) for i in range(space_size)]
        heapq.heapify(heap)

        while uncovered.any():
            if time.time() - t_start > deadline - 10:
                break
            found = False
            while heap:
                neg_count, idx = heapq.heappop(heap)
                count = -neg_count
                if not in_code[idx]:
                    actual = np.sum(uncovered[neighbors[idx]])
                    if actual > 0:
                        found = True
                        break
            if not found:
                # Rebuild
                for i in range(space_size):
                    if not in_code[i]:
                        c = np.sum(uncovered[neighbors[i]])
                        if c > 0:
                            heapq.heappush(heap, (-c, i))
                if not heap:
                    break
                neg_count, idx = heapq.heappop(heap)
                if -neg_count == 0:
                    break
                found = True

            code.append(idx)
            in_code[idx] = True
            uncovered[neighbors[idx]] = False

        return code

    def exhaustive_prune(code):
        """Remove all redundant codewords."""
        code = list(code)
        cov = np.zeros(space_size, dtype=np.int32)
        in_set = np.zeros(space_size, dtype=bool)
        for c in code:
            cov[neighbors[c]] += 1
            in_set[c] = True

        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                if np.min(cov[neighbors[c]]) > 1:
                    cov[neighbors[c]] -= 1
                    in_set[c] = False
                    code.pop(i)
                    changed = True
                    break
        return code, cov, in_set

    def directed_swap_search(code, cov, in_set, rng, max_iters=200):
        """
        For each word in code (random order), tentatively remove it.
        If uncovered words result, find a single codeword that covers all of them.
        Returns improved code or None.
        """
        code = list(code)
        cov = cov.copy()
        in_set = in_set.copy()
        improved = False

        indices = list(range(len(code)))
        rng.shuffle(indices)

        for i in indices:
            if time.time() - t_start > deadline - 5:
                break
            c = code[i]
            # Tentatively remove
            newly_uncovered = []
            for w in neighbors[c]:
                cov[w] -= 1
                if cov[w] == 0:
                    newly_uncovered.append(w)

            if not newly_uncovered:
                # Safe removal!
                in_set[c] = False
                code.pop(i)
                improved = True
                break

            # Find candidate covering all newly_uncovered
            # Sort by rarity
            newly_uncovered.sort(key=lambda w: len(cover_map[w]))
            cand_pool = None
            for w in newly_uncovered:
                cands = [c2 for c2 in cover_map[w] if not in_set[c2] and c2 != c]
                if cand_pool is None:
                    cand_pool = set(cands)
                else:
                    cand_pool &= set(cands)
                if not cand_pool:
                    break

            if cand_pool:
                # Pick best candidate (maximize coverage of already-covered words for redundancy)
                best_c = None
                best_score = -1
                for c2 in cand_pool:
                    score = np.sum(cov[neighbors[c2]] > 1)  # redundancy it adds
                    if score > best_score:
                        best_score = score
                        best_c = c2
                # Perform swap
                cov[neighbors[c]] -= 0  # already decremented
                cov[neighbors[best_c]] += 1
                in_set[c] = False
                in_set[best_c] = True
                code[i] = best_c
                improved = True
                break
            else:
                # Restore
                for w in newly_uncovered:
                    cov[w] += 1

        return (code, cov, in_set) if improved else None

    def pair_replace_search(code, cov, in_set, rng, max_pairs=50):
        """
        Try to replace 2 codewords with 1 codeword (net size -1).
        Remove c1 and c2, find c3 covering all words that were exclusively covered by c1 or c2.
        """
        code = list(code)
        cov = cov.copy()
        in_set = in_set.copy()
        improved = False

        for _ in range(max_pairs):
            if time.time() - t_start > deadline - 5:
                break
            i1 = rng.integers(0, len(code))
            i2 = rng.integers(0, len(code))
            if i1 == i2:
                continue
            c1 = code[i1]
            c2 = code[i2]

            # Find words exclusively covered by c1 or c2
            # A word w is exclusively covered by c1 if cov[w]==1 and c1 covers w
            # After removing both c1 and c2, words that become uncovered are those
            # with cov[w] <= 1 that are in neighbors[c1] or neighbors[c2]
            # More precisely: words where all coverers are in {c1, c2}
            # i.e., cov[w] <= 2 and (c1 in cover_map[w] or c2 in cover_map[w]) and all other coverers absent

            # Simpler: tentatively remove both, see what's uncovered
            newly_uncovered = []
            for w in neighbors[c1]:
                cov[w] -= 1
            for w in neighbors[c2]:
                cov[w] -= 1
                if cov[w] == 0:
                    newly_uncovered.append(w)
            # Also check words in neighbors[c1] that weren't in neighbors[c2]
            for w in neighbors[c1]:
                if cov[w] == 0 and w not in neighbors[c2]:
                    if w not in newly_uncovered:
                        newly_uncovered.append(w)

            if not newly_uncovered:
                # Both can be removed! Even better
                in_set[c1] = False
                in_set[c2] = False
                # Remove from code (handle indices)
                if i1 < i2:
                    code.pop(i2)
                    code.pop(i1)
                else:
                    code.pop(i1)
                    code.pop(i2)
                improved = True
                break

            # Find c3 covering all newly_uncovered
            newly_uncovered.sort(key=lambda w: len(cover_map[w]))
            cand_pool = None
            for w in newly_uncovered:
                cands = [c2 for c2 in cover_map[w] if not in_set[c2] and c2 != c1 and c2 != c2]
                if cand_pool is None:
                    cand_pool = set(cands)
                else:
                    cand_pool &= set(cands)
                if not cand_pool:
                    break

            if cand_pool:
                # Pick best
                best_c3 = None
                best_score = -1
                for c3 in cand_pool:
                    score = np.sum(cov[neighbors[c3]] > 1)
                    if score > best_score:
                        best_score = score
                        best_c3 = c3
                # Perform: remove c1, c2; add c3
                cov[neighbors[c3]] += 1
                in_set[c1] = False
                in_set[c2] = False
                in_set[c3] = True
                # Update code list
                if i1 < i2:
                    code.pop(i2)
                    code.pop(i1)
                else:
                    code.pop(i1)
                    code.pop(i2)
                code.append(c3)
                improved = True
                break
            else:
                # Restore
                for w in neighbors[c1]:
                    cov[w] += 1
                for w in neighbors[c2]:
                    cov[w] += 1

        return (code, cov, in_set) if improved else None

    # === MAIN ALGORITHM ===
    # Phase 1: Greedy initialization
    initial_code = greedy_cover(42)
    initial_code = exhaustive_prune(initial_code)
    if isinstance(initial_code, tuple):
        initial_code, init_cov, init_in_set = initial_code
    else:
        init_cov = np.zeros(space_size, dtype=np.int32)
        init_in_set = np.zeros(space_size, dtype=bool)
        for c in initial_code:
            init_cov[neighbors[c]] += 1
            init_in_set[c] = True

    best_code = list(initial_code)
    best_size = len(best_code)

    rng = np.random.default_rng(123)

    # Phase 2: ILS loop
    current_code = list(initial_code)
    current_cov = init_cov.copy()
    current_in_set = init_in_set.copy()

    no_improve = 0
    max_no_improve = 200

    while time.time() - t_start < deadline - 3:
        # Try directed swap search
        result = directed_swap_search(current_code, current_cov, current_in_set, rng)
        if result:
            current_code, current_cov, current_in_set = result
            if len(current_code) < best_size:
                best_code = list(current_code)
                best_size = len(best_code)
            no_improve = 0
            continue

        # Try pair replacement
        result = pair_replace_search(current_code, current_cov, current_in_set, rng, max_pairs=30)
        if result:
            current_code, current_cov, current_in_set = result
            if len(current_code) < best_size:
                best_code = list(current_code)
                best_size = len(best_code)
            no_improve = 0
            continue

        # Stagnation: perturb by random swap
        no_improve += 1
        if no_improve > max_no_improve:
            # Perturb: swap a random word with a random candidate
            if len(current_code) > 0:
                i = rng.integers(0, len(current_code))
                new_c = rng.integers(0, space_size)
                c = current_code[i]
                if not current_in_set[new_c] and new_c != c:
                    # Check validity
                    valid = True
                    for w in neighbors[c]:
                        if current_cov[w] == 1 and new_c not in neighbors[new_c]:
                            valid = False
                            break
                    if valid:
                        current_cov[neighbors[c]] -= 1
                        current_cov[neighbors[new_c]] += 1
                        current_in_set[c] = False
                        current_in_set[new_c] = True
                        current_code[i] = new_c
                    no_improve = 0

            # Occasionally re-initialize from scratch with different seed
            if no_improve > max_no_improve * 3 and time.time() - t_start < deadline - 30:
                seed = int(rng.integers(0, 100000))
                new_code = greedy_cover(seed)
                new_code = exhaustive_prune(new_code)
                if isinstance(new_code, tuple):
                    new_code, new_cov, new_in_set = new_code
                else:
                    new_cov = np.zeros(space_size, dtype=np.int32)
                    new_in_set = np.zeros(space_size, dtype=bool)
                    for c in new_code:
                        new_cov[neighbors[c]] += 1
                        new_in_set[c] = True
                if len(new_code) < len(current_code):
                    current_code = new_code
                    current_cov = new_cov
                    current_in_set = new_in_set
                    if len(current_code) < best_size:
                        best_code = list(current_code)
                        best_size = len(current_code)
                no_improve = 0

    # Final verification
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1

    if np.any(final_cov == 0):
        # Fallback
        for c in initial_code:
            if c not in best_code:
                best_code.append(c)
        final_cov = np.zeros(space_size, dtype=np.int32)
        for c in best_code:
            final_cov[neighbors[c]] += 1
        # If still bad, brute force fill
        if np.any(final_cov == 0):
            for i in range(space_size):
                if final_cov[i] == 0:
                    best_code.append(i)
                    final_cov[neighbors[i]] += 1

    return [idx_to_word(c) for c in best_code]
