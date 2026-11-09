"""Rewritten program: vectorized greedy + local search for A_2(18,6)."""
import time
import numpy as np

N = 18
MINIMUM_DISTANCE = 6
FULL_MASK = (1 << N) - 1


def _reed_muller_base():
    """First-order Reed-Muller code on 16 coords, extended by two zero bits.
    32 words, pairwise distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _bits_array(words):
    """Convert list of ints to (K, N) uint8 array of bits (bit i -> column i)."""
    arr = np.zeros((len(words), N), dtype=np.uint8)
    for k, w in enumerate(words):
        x = w
        for i in range(N):
            arr[k, i] = x & 1
            x >>= 1
    return arr


def _popcount_int(x):
    return x.bit_count()


def _batch_min_dist(cand_arr, code_arr):
    """cand_arr: (B, N) uint8 bits. code_arr: (K, N) uint8 bits.
    Returns (B,) uint8 min Hamming distance from each candidate to the code."""
    # XOR via addition mod 2 (uint8): (a+b) % 2
    # Compute per-candidate distance to each code word: sum over N of (cand != code)
    # cand_arr[:, None, :] shape (B,1,N); code_arr[None,:,:] shape (1,K,N)
    diff = (cand_arr[:, None, :] + code_arr[None, :, :]) % 2  # (B, K, N)
    dists = diff.sum(axis=2, dtype=np.uint16)  # (B, K)
    return dists.min(axis=1)


def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    best_words = _reed_muller_base()
    best_len = len(best_words)

    # Precompute a pool of candidate words to draw from.
    # Use all 2^18 words? That's 262144 — manageable as an int list.
    # We'll work with a numpy array of all words' bit representations for batch checks.
    # But storing (262144, 18) uint8 is ~4.7 MB — fine.
    all_words = np.arange(1 << N, dtype=np.uint32)
    all_bits = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_bits[:, i] = (all_words >> i) & 1

    # Track which indices are in the current code (for exclusion).
    # code_set for O(1) membership.
    code_set = set(best_words)
    code_arr = _bits_array(best_words)
    K = len(best_words)

    # Precompute distance of every word to the all-zero vector (weight) for ordering.
    # weight of word w = popcount(w). We'll use this to order candidates.
    weights = all_bits.sum(axis=1, dtype=np.uint8)  # (262144,)

    # We'll maintain a "available" boolean mask over all_words indices.
    available = np.ones(1 << N, dtype=bool)
    for w in best_words:
        available[w] = False

    # Greedy phase: repeatedly find the best available word to add.
    # To keep it fast, we process in batches: take a chunk of available indices,
    # compute min dist to code, pick the one with max min-dist (>=6) and add it.
    # But naive "best available" requires scanning all 262144 each step — too slow
    # for many steps. Instead, use a randomized + greedy hybrid:
    #   Each iteration, sample a batch of candidate indices, compute their min-dist
    #   to the current code, and add any that are compatible (dist>=6), preferring
    #   those that are "far" (high min-dist) to leave room.

    rng = np.random.default_rng(12345)
    BATCH = 20000

    def try_add_batch():
        nonlocal available, code_arr, K, code_set, best_words, best_len
        # Pick a random batch of available indices.
        avail_idx = np.nonzero(available)[0]
        if len(avail_idx) == 0:
            return False
        take = min(BATCH, len(avail_idx))
        chosen = rng.choice(avail_idx, size=take, replace=False)
        cand_arr = all_bits[chosen]
        min_d = _batch_min_dist(cand_arr, code_arr)
        # Compatible: min_d >= MINIMUM_DISTANCE
        compat = np.nonzero(min_d >= MINIMUM_DISTANCE)[0]
        if len(compat) == 0:
            return False
        # Among compatible, pick the one with the largest min_d (most "room"),
        # tie-break by lower weight (lighter words tend to be more compatible).
        scores = min_d[compat]
        # We want max score; secondary: we don't have weight here directly.
        # Use argmax of score.
        order = np.argsort(-scores, kind='stable')
        added = False
        for oi in order:
            ci = compat[oi]
            idx = int(chosen[ci])
            w = int(all_words[idx])
            if w in code_set:
                continue
            # Add it
            best_words.append(w)
            code_set.add(w)
            available[idx] = False
            # Update code_arr
            new_row = all_bits[idx]
            code_arr = np.vstack([code_arr, new_row])
            K += 1
            added = True
            # Add just one per batch to keep code_arr updates cheap;
            # but we could add more. Let's add all compatible in this batch
            # to be efficient (they're all compatible with the OLD code, but
            # may not be compatible with each other). So add one at a time and
            # re-check. Simpler: add the best one, then break and re-batch.
            break
        return added

    # Greedy loop
    while time.time() < deadline - 5.0:
        if not try_add_batch():
            # No more easy additions; break to local search.
            break

    # Local search: add + swap moves, vectorized.
    # Represent current code as list.
    def current_len():
        return len(best_words)

    # We'll do local search with a time budget.
    ls_deadline = deadline - 2.0
    improved = True
    while time.time() < ls_deadline and improved:
        improved = False
        # --- ADD move: try to add a word (same as greedy but more thorough) ---
        avail_idx = np.nonzero(available)[0]
        if len(avail_idx) > 0:
            # Process in larger batches to find any addable word.
            step = 50000
            for start in range(0, len(avail_idx), step):
                if time.time() > ls_deadline:
                    break
                chunk = avail_idx[start:start + step]
                cand_arr = all_bits[chunk]
                min_d = _batch_min_dist(cand_arr, code_arr)
                compat = np.nonzero(min_d >= MINIMUM_DISTANCE)[0]
                if len(compat) > 0:
                    ci = int(compat[0])
                    idx = int(chunk[ci])
                    w = int(all_words[idx])
                    if w not in code_set:
                        best_words.append(w)
                        code_set.add(w)
                        available[idx] = False
                        new_row = all_bits[idx]
                        code_arr = np.vstack([code_arr, new_row])
                        K += 1
                        improved = True
                        break
        # --- SWAP move: for each existing word, see if removing it allows
        #     adding a better word (net +1) — but net count only changes if we
        #     can add after removing. Simpler swap: replace word i with a word
        #     that is compatible with all OTHERS (not with i). This keeps count
        #     same but may enable future adds. Try a few.
        if time.time() < ls_deadline - 1.0:
            # For a sample of existing words, compute the "constraint" of the
            # rest of the code, then find a replacement.
            # This is expensive; do a limited version.
            n = len(best_words)
            if n >= 2:
                # Pick a few indices to try swapping.
                try_swaps = min(4, n)
                swap_indices = rng.choice(n, size=try_swaps, replace=False)
                for si in swap_indices:
                    if time.time() > ls_deadline:
                        break
                    removed = best_words[si]
                    # Build code without this word.
                    mask = np.ones(n, dtype=bool)
                    mask[si] = False
                    rest_arr = code_arr[mask]
                    # Find an available word compatible with rest_arr.
                    avail_idx2 = np.nonzero(available)[0]
                    if len(avail_idx2) == 0:
                        continue
                    # Sample a batch.
                    take2 = min(30000, len(avail_idx2))
                    chosen2 = rng.choice(avail_idx2, size=take2, replace=False)
                    cand2 = all_bits[chosen2]
                    min_d2 = _batch_min_dist(cand2, rest_arr)
                    # Must also not equal removed (it's available so != removed).
                    compat2 = np.nonzero(min_d2 >= MINIMUM_DISTANCE)[0]
                    if len(compat2) > 0:
                        # Prefer the replacement with high min_d2.
                        order2 = np.argsort(-min_d2[compat2], kind='stable')
                        ci2 = int(compat2[order2[0]])
                        idx2 = int(chosen2[ci2])
                        w2 = int(all_words[idx2])
                        if w2 not in code_set and w2 != removed:
                            # Perform swap
                            best_words[si] = w2
                            code_set.discard(removed)
                            code_set.add(w2)
                            available[idx2] = False
                            available[removed] = True
                            # Update code_arr row
                            code_arr[si] = all_bits[idx2]
                            improved = True
                            break

    # Final validity check and return.
    # Ensure distinct and in range.
    result = list(dict.fromkeys(best_words))  # dedupe preserving order
    # Verify pairwise distance (integer XOR + popcount) as required.
    m = len(result)
    ok = True
    for i in range(m):
        wi = result[i]
        for j in range(i + 1, m):
            if (wi ^ result[j]).bit_count() < MINIMUM_DISTANCE:
                ok = False
                break
        if not ok:
            break
    if not ok or m == 0:
        # Fallback to the base code.
        result = _reed_muller_base()
    return result
