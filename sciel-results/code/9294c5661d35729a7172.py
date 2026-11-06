import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0

def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _precompute_neighbor_indices():
    """
    Precompute for every word w in [0, 2^N) the list of all words within
    Hamming distance <= MIN_DISTANCE-1 (=5). Store as a flat int32 array
    with offset pointers.
    """
    # Number of words within distance d of a center in N dimensions:
    # sum_{r=0}^{d} C(N, r)
    count = sum(1 for _ in range(1))  # just 1 for r=0
    for r in range(1, MIN_DISTANCE):
        count += 1
    # Actually compute properly:
    total_neighbors = 0
    for r in range(MIN_DISTANCE):
        c = 1
        for i in range(r):
            c = c * (N - i) // (i + 1)
        total_neighbors += c
    # total_neighbors = sum_{r=0}^{5} C(18,r) = 1+18+153+816+3060+8568 = 12616
    # That's 12616 * 2^18 * 4 bytes = ~3.3 GB. Too large.
    # Alternative: don't precompute all. Use on-the-fly with vectorized XOR.
    return None

def _neighbors_of_word(w, radius):
    """Return numpy array of all words within Hamming distance <= radius of w."""
    bits = np.arange(N)
    result = [w]
    for r in range(1, radius + 1):
        for comb in combinations(range(N), r):
            v = w
            for b in comb:
                v ^= (1 << b)
            result.append(v)
    return np.array(result, dtype=np.int32)

def _build_forbidden_mask(words):
    """Build a boolean mask of size 2^N where True means 'forbidden'."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _neighbors_of_word(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden

def _generate_candidates_by_weight():
    """
    Generate all 2^18 words sorted by Hamming weight, then by value.
    Returns a numpy array of int32.
    """
    # Bucket by weight
    buckets = [[] for _ in range(N + 1)]
    for w in range(TOTAL):
        buckets[w.bit_count()].append(w)
    flat = []
    for weight in range(N + 1):
        flat.extend(buckets[weight])
    return np.array(flat, dtype=np.int32)

def _vectorized_batch_check(candidates, forbidden):
    """
    Given a numpy array of candidate word indices and a boolean forbidden array,
    return a boolean mask of which candidates are NOT forbidden.
    This is a single fancy-index operation.
    """
    return ~forbidden[candidates]

def _bulk_add(words_list, candidates, valid_mask, forbidden):
    """Add all valid candidates to words_list and update forbidden mask."""
    valid_candidates = candidates[valid_mask]
    for w in valid_candidates:
        words_list.append(int(w))
    # Update forbidden: mark neighbors of all newly added words
    for w in valid_candidates:
        neighbors = _neighbors_of_word(int(w), MIN_DISTANCE - 1)
        forbidden[neighbors] = True

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    try:
        # Start with RM seed
        words = _seed_words()

        # Build initial forbidden mask
        forbidden = _build_forbidden_mask(words)

        # Pre-generate all candidates sorted by weight
        candidates_all = _generate_candidates_by_weight()

        # Batch size for vectorized checking
        BATCH_SIZE = 65536  # 2^16

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Process candidates in batches
            for batch_start in range(0, TOTAL, BATCH_SIZE):
                if time.time() > deadline:
                    break

                batch_end = min(batch_start + BATCH_SIZE, TOTAL)
                batch = candidates_all[batch_start:batch_end]

                # Vectorized check: which in this batch are not forbidden?
                valid_mask = _vectorized_batch_check(batch, forbidden)

                if valid_mask.any():
                    _bulk_add(words, batch, valid_mask, forbidden)
                    improved = True

        # After greedy packing, try a simple swap-based local search
        # to try to improve further
        if time.time() < deadline:
            # Try removing random words and re-greedy-ing
            import random
            random.seed(42)
            max_swap_rounds = 3
            for _ in range(max_swap_rounds):
                if time.time() > deadline:
                    break
                # Pick a random word to remove
                idx = random.randint(0, len(words) - 1)
                w_remove = words[idx]
                # Temporarily remove
                words.pop(idx)
                # Rebuild forbidden (expensive but we only do a few times)
                forbidden = _build_forbidden_mask(words)
                # Quick greedy pass
                for batch_start in range(0, TOTAL, BATCH_SIZE):
                    if time.time() > deadline:
                        break
                    batch_end = min(batch_start + BATCH_SIZE, TOTAL)
                    batch = candidates_all[batch_start:batch_end]
                    valid_mask = _vectorized_batch_check(batch, forbidden)
                    if valid_mask.any():
                        _bulk_add(words, batch, valid_mask, forbidden)

        if len(words) > best_len:
            best_len = len(words)
            best_words = list(words)

    except Exception:
        pass

    if best_words is None:
        best_words = _seed_words()

    # Deduplicate and validate
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
