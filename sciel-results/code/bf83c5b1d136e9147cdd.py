"""Vectorized greedy + local search for A_2(17,6)."""
import numpy as np
import time

N = 17
MIN_DIST = 6

def _parity_table(nbits):
    """Return array P where P[x] = popcount(x) mod 2 for x in [0, 2^nbits)."""
    size = 1 << nbits
    p = np.zeros(size, dtype=np.uint8)
    for i in range(1, size):
        p[i] = p[i >> 1] ^ (i & 1)
    return p

def _precompute_parity():
    return _parity_table(N)

def _hamming(v1, v2, P):
    """Vectorized Hamming distance between v1 and array v2."""
    return (P[v1] ^ P[v2]).sum(axis=1)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    P = _precompute_parity()
    all_words = np.arange(1 << N, dtype=np.int32)
    bits = (all_words[:, None] >> np.arange(N)) & 1  # (131072, 17) uint8

    # Greedy construction with vectorized distance checks
    code = []
    used = np.zeros(1 << N, dtype=bool)

    # Start with random-ish ordering for diversity
    rng = np.random.default_rng(42)

    # Phase 1: Greedy add
    remaining = all_words.copy()
    while len(remaining) > 0:
        if time.time() > deadline - 15.0:
            break
        # Pick a random candidate from remaining
        idx = rng.integers(len(remaining))
        w = remaining[idx]
        # Check distance to all current codewords
        if len(code) == 0:
            code.append(int(w))
            used[w] = True
            remaining = np.delete(remaining, idx)
            continue
        code_arr = np.array(code, dtype=np.int32)
        dists = (P[w] ^ P[code_arr]).sum(axis=1)
        if dists.min() >= MIN_DIST:
            code.append(int(w))
            used[w] = True
            remaining = np.delete(remaining, idx)

    # Phase 2: Local search - try to swap out a codeword for a better one
    # Johnson-style: try to add a word, if it conflicts, remove a conflicting word
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        code_arr = np.array(code, dtype=np.int32)
        # Find all words not in code that are at distance >= MIN_DIST from most codewords
        # Try each unused word as a potential addition
        unused_mask = ~used
        # Sample unused words
        n_unused = int(unused_mask.sum())
        if n_unused == 0:
            break

        # Try a batch of random unused words
        batch_size = min(5000, n_unused)
        candidates = rng.choice(np.where(unused_mask)[0], size=batch_size, replace=False)

        for w in candidates:
            if time.time() > deadline - 5.0:
                break
            w = int(w)
            dists = (P[w] ^ P[code_arr]).sum(axis=1)
            conflicts = np.where(dists < MIN_DIST)[0]
            if len(conflicts) == 0:
                # Can add directly
                code.append(w)
                used[w] = True
                improved = True
                code_arr = np.array(code, dtype=np.int32)
            elif len(conflicts) == 1:
                # Swap: remove the conflicting word, add w
                # Check if removing it and adding w improves (it does if w is "better positioned")
                # Simple heuristic: just do the swap
                removed = int(code_arr[conflicts[0]])
                code[conflicts[0]] = w
                used[removed] = False
                used[w] = True
                improved = True
                code_arr = np.array(code, dtype=np.int32)

    # Phase 3: Final greedy pass over remaining words
    if time.time() < deadline - 2.0:
        code_arr = np.array(code, dtype=np.int32)
        unused_mask = ~used
        candidates = np.where(unused_mask)[0]
        # Sort by weight (lower weight words tend to be more "spread out")
        weights = P[candidates].sum(axis=1)
        order = np.argsort(weights)
        for i in order:
            if time.time() > deadline - 2.0:
                break
            w = int(candidates[i])
            dists = (P[w] ^ P[code_arr]).sum(axis=1)
            if dists.min() >= MIN_DIST:
                code.append(w)
                used[w] = True
                code_arr = np.array(code, dtype=np.int32)

    return [int(x) for x in code]
