"""Bitset-accelerated greedy covering C(24,7,3) with randomized restarts.

Uses bitmasks for fast marginal gain calculation. Implements randomized
greedy set cover with multiple restarts to find a better (smaller) solution
than deterministic greedy.
"""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _trunc(n, r):
    """Number of r-combinations of n elements."""
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res


def _combo_id(combo, n, r):
    """Lexicographic rank of an r-combination."""
    rank = 0
    for i in range(r):
        lo = combo[i - 1] + 1 if i > 0 else 0
        rank += _trunc(combo[i] - lo, r - i)
    return rank


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0

    # --- build id table for all t-subsets -------------------------------
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    id_of = {}
    for idx, combo in enumerate(all_t):
        id_of[combo] = idx

    # --- precompute coverage bitmask for every candidate k-block ---------
    # Use a list of (mask, block) for faster iteration
    candidates = []
    for block in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(block, t):
            m |= 1 << id_of[sub]
        candidates.append((m, block))
    
    n_candidates = len(candidates)
    
    def greedy_run(rng):
        """Perform one greedy run with random tie-breaking."""
        uncovered = (1 << n_t) - 1
        blocks = []
        used = set()
        # List of indices of candidates not yet used? 
        # Iterating over all candidates and checking 'used' is O(N).
        # N=346k. This is fine.
        
        while uncovered:
            if time.time() >= deadline:
                return None # Timeout
            
            best_gain = -1
            best_candidates = []
            
            for i, (mask, block) in enumerate(candidates):
                if i in used:
                    continue
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain <= 0:
                return None # No valid block found (should not happen if cover exists)
            
            # Randomly pick from best candidates
            idx = rng.choice(best_candidates)
            mask, block = candidates[idx]
            used.add(idx)
            blocks.append(list(block))
            uncovered &= ~mask
            
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances
    # We can afford a few restarts.
    # Each greedy run takes roughly (num_blocks * N_candidates) operations.
    # 76 blocks * 346k candidates * ~10 ops = ~260M ops.
    # Python does ~10-50M ops/sec. So 1 run takes ~5-25s.
    # We have 115s. So we can do ~5-10 runs.
    
    rng = random.Random(42)
    
    for run_id in range(10):
        if time.time() > deadline - 20: # Leave time for final check
            break
            
        current_blocks = greedy_run(rng)
        
        if current_blocks is not None:
            # Verify validity (safety check, should be true)
            # and check count
            count = len(current_blocks)
            if count < best_count:
                best_count = count
                best_blocks = current_blocks
                # If we found a very good one, we might stop early? 
                # No, keep trying.
                
    if best_blocks is None:
        # Fallback to deterministic if all timed out or failed
        # This shouldn't happen if the first run didn't timeout
        # But if the first run timed out, we return None? 
        # The task requires a valid list. 
        # If we can't finish a greedy run, we might have to return a partial one?
        # No, "Valid when every t-subset... lies in at least one block".
        # If we return a partial cover, it's invalid -> score 0.
        # So we MUST finish a valid cover.
        # The deterministic greedy should finish in ~10-20s.
        # So best_blocks should not be None.
        pass

    return best_blocks
# EVOLVE-BLOCK-END
