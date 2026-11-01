"""Optimized construction for K_3(7,1) using vectorized greedy + advanced local search."""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """Vectorized greedy + local search with 2-for-1, 1-for-2, removal, and swap moves."""
    N = 2187
    L = 7

    # Precompute: for each word index, its 15 neighbors (itself + 14 at distance 1)
    # Using base-3 representation
    words = np.zeros((N, L), dtype=np.int32)
    for i in range(N):
        tmp = i
        for pos in range(L):
            words[i, pos] = tmp % 3
            tmp //= 3

    # Precompute cover sets as numpy arrays of indices
    # cover_idx[i] = list of word indices within distance 1 of word i
    # Also precompute a boolean matrix (N x N) for membership
    # N=2187, so N*N = ~4.8M booleans, manageable
    cover_mask = np.zeros((N, N), dtype=bool)
    cover_indices = [None] * N

    for i in range(N):
        w = words[i]
        indices = [i]
        for pos in range(L):
            for val in range(3):
                if val != w[pos]:
                    new_i = i + (val - w[pos]) * (3 ** pos)
                    indices.append(new_i)
        cover_indices[i] = indices
        cover_mask[i, indices] = True

    # Also: for each word j, which codewords cover it (reverse)
    # reverse_cover[j] = list of i such that j is in cover_indices[i]
    # This is just the columns of cover_mask
    # Actually cover_mask[i,j] = True means codeword i covers word j
    # So reverse: for word j, the codewords that cover it are those i where cover_mask[i,j]=True

    def greedy_cover_vectorized(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = None

        uncovered = np.ones(N, dtype=bool)
        selected = []

        # Precompute: for each candidate i, which words it covers (as a boolean vector)
        # cover_mask[i] is already that

        while uncovered.any():
            # Count how many uncovered words each candidate covers
            # counts[i] = sum(cover_mask[i] & uncovered)
            counts = (cover_mask & uncovered[np.newaxis, :]).sum(axis=1)

            best_count = counts.max()
            if best_count == 0:
                break

            # Find all candidates with max count
            best_mask = (counts == best_count)
            best_indices = np.where(best_mask)[0]

            if randomize and len(best_indices) > 1 and rng is not None:
                idx = rng.choice(best_indices)
            else:
                idx = best_indices[0]

            selected.append(idx)
            uncovered[cover_mask[idx]] = False

            if time.time() > deadline - 5.0:
                break

        return selected, uncovered

    def local_search_advanced(code, deadline):
        """Local search with removal, swap, 2-for-1, and 1-for-2 moves."""
        if not code:
            return code

        code_set = set(code)
        best_code = list(code_set)
        best_size = len(best_code)

        # counts[j] = number of codewords in code_set that cover word j
        counts = np.zeros(N, dtype=np.int32)
        for c in code_set:
            counts[cover_indices[c]] += 1

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            current_list = list(code_set)
            random.shuffle(current_list)

            # Phase 1: Removals
            for c in current_list:
                if c not in code_set:
                    continue
                # Check if all words covered by c are also covered by another codeword
                if (counts[cover_indices[c]] > 1).all():
                    counts[cover_indices[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

            # Phase 2: Swaps (replace one codeword with another)
            if time.time() < deadline - 5.0:
                swap_attempts = 30
                for _ in range(swap_attempts):
                    if time.time() > deadline - 5.0:
                        break
                    if not code_set:
                        break
                    c_old = random.choice(list(code_set))
                    c_new = random.randint(0, N - 1)
                    if c_new in code_set:
                        continue
                    # Check: every word covered only by c_old must be covered by c_new
                    # i.e., for all j in cover_indices[c_old] with counts[j]==1, j must be in cover_indices[c_new]
                    critical = cover_indices[c_old]
                    ok = True
                    for j in critical:
                        if counts[j] == 1:
                            if not cover_mask[c_new, j]:
                                ok = False
                                break
                    if ok:
                        counts[cover_indices[c_old]] -= 1
                        counts[cover_indices[c_new]] += 1
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        improved = True

            # Phase 3: 2-for-1 (replace two codewords with one)
            if time.time() < deadline - 8.0:
                two_for_one_attempts = 15
                cl = list(code_set)
                random.shuffle(cl)
                for _ in range(two_for_one_attempts):
                    if time.time() > deadline - 8.0:
                        break
                    if len(code_set) < 2:
                        break
                    c1 = random.choice(cl)
                    if c1 not in code_set:
                        continue
                    remaining = [x for x in cl if x != c1 and x in code_set]
                    if not remaining:
                        break
                    c2 = random.choice(remaining)
                    if c2 not in code_set:
                        continue
                    # Try to find a single codeword c_new that covers all words
                    # that are currently covered only by c1 or c2 (i.e., counts[j] <= 2 and j in cover(c1) or cover(c2))
                    # Actually: words that would lose coverage: j in (cover(c1) ∪ cover(c2)) where counts[j] <= 2
                    # After removing c1 and c2, counts[j] decreases by (1 if j in cover(c1)) + (1 if j in cover(c2))
                    # We need: for all j, new_counts[j] >= 1, i.e., counts[j] - (c1 covers j) - (c2 covers j) + (c_new covers j) >= 1
                    # So: for all j in (cover(c1) ∪ cover(c2)): counts[j] - cov1[j] - cov2[j] + cov_new[j] >= 1
                    # Where cov1[j]=1 if j in cover(c1), etc.
                    # Equivalently: for all j in (cover(c1) ∪ cover(c2)) where counts[j] - cov1[j] - cov2[j] < 1:
                    #   we need cov_new[j] = 1, i.e., j in cover(c_new)
                    # Let's compute the "must cover" set
                    must_cover = []
                    for j in set(cover_indices[c1]) | set(cover_indices[c2]):
                        cov1 = 1 if cover_mask[c1, j] else 0
                        cov2 = 1 if cover_mask[c2, j] else 0
                        remaining_count = counts[j] - cov1 - cov2
                        if remaining_count < 1:
                            must_cover.append(j)
                    if not must_cover:
                        # Any codeword works, pick one not in code_set
                        c_new = random.randint(0, N - 1)
                        if c_new in code_set:
                            # Try a few more
                            found = False
                            for _ in range(20):
                                c_new = random.randint(0, N - 1)
                                if c_new not in code_set:
                                    found = True
                                    break
                            if not found:
                                continue
                    else:
                        # Find codewords that cover all must_cover words
                        # A codeword c_new covers word j iff cover_mask[c_new, j]
                        # We need cover_mask[c_new, must_cover].all()
                        # This is expensive; approximate by checking candidates
                        # Candidates: codewords that cover at least the first few must_cover words
                        if len(must_cover) > 10:
                            continue  # Too restrictive, skip
                        # Try random candidates
                        found = False
                        for _ in range(30):
                            c_new = random.randint(0, N - 1)
                            if c_new in code_set:
                                continue
                            if all(cover_mask[c_new, j] for j in must_cover):
                                found = True
                                break
                        if not found:
                            continue

                    # Execute 2-for-1
                    counts[cover_indices[c1]] -= 1
                    counts[cover_indices[c2]] -= 1
                    counts[cover_indices[c_new]] += 1
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(c_new)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

            # Phase 4: 1-for-2 (replace one codeword with two)
            if time.time() < deadline - 6.0:
                one_for_two_attempts = 10
                for _ in range(one_for_two_attempts):
                    if time.time() > deadline - 6.0:
                        break
                    if not code_set:
                        break
                    c_old = random.choice(list(code_set))
                    # Find two codewords c_a, c_b such that cover(c_a) ∪ cover(c_b) ⊇ cover(c_old)
                    # i.e., for all j in cover(c_old), j in cover(c_a) or j in cover(c_b)
                    # And c_a, c_b not in code_set (or one could be c_old, but that's pointless)
                    # Strategy: pick c_a randomly, then c_b must cover cover(c_old) \ cover(c_a)
                    found_pair = False
                    for _ in range(20):
                        c_a = random.randint(0, N - 1)
                        if c_a in code_set:
                            continue
                        # Words in cover(c_old) not covered by c_a
                        need = [j for j in cover_indices[c_old] if not cover_mask[c_a, j]]
                        if not need:
                            # c_a alone covers everything, so just use c_a (1-for-1, handled by swap)
                            continue
                        # Find c_b that covers all of need
                        if len(need) > 15:
                            continue
                        for _ in range(20):
                            c_b = random.randint(0, N - 1)
                            if c_b in code_set or c_b == c_a:
                                continue
                            if all(cover_mask[c_b, j] for j in need):
                                # Execute
                                counts[cover_indices[c_old]] -= 1
                                counts[cover_indices[c_a]] += 1
                                counts[cover_indices[c_b]] += 1
                                code_set.remove(c_old)
                                code_set.add(c_a)
                                code_set.add(c_b)
                                improved = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if found_pair:
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        return best_code

    best_code = None
    best_size = N

    # 1. Deterministic greedy + local search
    code, unc = greedy_cover_vectorized(randomize=False)
    if not unc.any():
        code = local_search_advanced(code, deadline)
        if code:
            best_code = code
            best_size = len(code)

    # 2. Randomized greedy + local search
    attempts = 0
    while time.time() < deadline - 12.0:
        attempts += 1
        code, unc = greedy_cover_vectorized(randomize=True, seed=attempts * 7919)
        if not unc.any():
            code = local_search_advanced(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
                # Early exit if we hit a very good solution
                if best_size <= 188:
                    break

    if best_code is None:
        code, unc = greedy_cover_vectorized(randomize=False)
        if unc.any():
            while unc.any():
                idx = np.where(unc)[0][0]
                code.append(idx)
                unc[cover_mask[idx]] = False
        best_code = code
        best_size = len(code)

    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * 7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)

    return result

def _construct_general(q, n, radius, deadline):
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)

    uncovered = set(range(total))
    selected = []

    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i

        if best_idx == -1:
            break

        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)

    if uncovered:
        for j in uncovered:
            selected.append(j)

    return [list(all_words[i]) for i in selected]
