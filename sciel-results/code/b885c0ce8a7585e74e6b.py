"""Covering design C(v,k,t) via randomized greedy set cover with NumPy-optimized gains."""
import time
import itertools
import numpy as np

# We hardcode the specific instance v=24,k=7,t=3 for speed, but keep the signature.
# Precompute all C(24,3) = 2024 t-subsets and assign each an index.
_V, _K, _T = 24, 7, 3

def _precompute():
    """Precompute mapping from (a,b,c) with a<b<c to index 0..2023."""
    combos = list(itertools.combinations(range(_V), _T))
    idx_map = {}
    for i, c in enumerate(combos):
        idx_map[c] = i
    # Also store combos as a numpy array of shape (2024, 3) for fast operations
    combos_arr = np.array(combos, dtype=np.int32)
    return idx_map, combos_arr

_IDX_MAP, _COMBOS_ARR = _precompute()
_N_SUBSETS = len(_COMBOS_ARR)  # 2024

def _block_to_subset_indices(block):
    """Given a sorted list of 3 points, return the index into _COMBOS_ARR."""
    return _IDX_MAP[tuple(block)]

def _block_covers(block):
    """Return the set of t-subset indices covered by a given k-block."""
    covered = set()
    for sub in itertools.combinations(block, _T):
        covered.add(_IDX_MAP[sub])
    return covered

def construct(v, k, t):
    """
    Construct a covering design using randomized greedy set cover.
    Multiple random restarts with NumPy-accelerated gain computation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # For generality, if parameters differ from hardcoded, fall back to simple greedy
    if v != _V or k != _K or t != _T:
        return _simple_greedy(v, k, t)

    best_blocks = None
    best_count = float('inf')
    rng = np.random.default_rng(42)

    # Precompute for each point x and each pair (a,b), the t-subset index for {a,b,x}
    # This is a 3D array: gain_table[x, a, b] = subset_index for {a,b,x} if a<b<x or similar
    # Actually, let's precompute: for each x, a list of (pair_index, subset_index)
    # But simpler: for a given partial block of size s, and candidate x,
    # the new subsets covered are those formed by x and (t-1)-subsets of the partial block.

    # Let's precompute a lookup: for each x in 0..23, and each (a,b) with a<b,
    # the index of the 3-subset sorted(a,b,x)
    # We'll store this as a 2D array: lookup[x, pair_idx] = subset_idx
    # where pair_idx encodes (a,b)

    # Number of pairs: C(24,2) = 276
    pairs = list(itertools.combinations(range(_V), 2))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    # lookup[x, pair_idx] = index of 3-subset formed by x and pair
    lookup = np.zeros((_V, len(pairs)), dtype=np.int32)
    for x in range(_V):
        for pi, (a, b) in enumerate(pairs):
            sub = tuple(sorted((a, b, x)))
            lookup[x, pi] = _IDX_MAP[sub]

    # For each x, precompute which pairs it forms valid 3-subsets with
    # (all pairs work, just need the subset index)

    # Strategy: randomized greedy
    # 1. Start with all subsets uncovered
    # 2. Repeatedly:
    #    a. Pick a random uncovered subset to "anchor"
    #    b. Build a block of size k starting from that subset's points
    #    c. For each remaining slot, pick the point that maximizes newly covered subsets
    #    d. Add block, update uncovered
    # 3. After each block, optionally try to remove redundant blocks
    # 4. Repeat with different random seeds, keep best

    def greedy_once(seed):
        local_rng = np.random.default_rng(seed)
        uncovered = np.ones(_N_SUBSETS, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered subset to anchor
            unc_idx = np.where(uncovered)[0]
            anchor_idx = unc_idx[local_rng.integers(len(unc_idx))]
            anchor = _COMBOS_ARR[anchor_idx]  # (a, b, c)
            
            # Start block with the 3 anchor points
            block = list(anchor)
            block_set = set(block)
            
            # Compute which subsets are covered by this partial block
            # For a 3-element block, it covers exactly 1 subset (itself)
            # For larger blocks, we need to track all t-subsets
            
            # Let's track covered subsets as a boolean array
            covered = np.zeros(_N_SUBSETS, dtype=bool)
            # Mark the anchor subset
            covered[anchor_idx] = True
            
            # Now fill remaining k - t = 4 points
            for _ in range(_K - _T):
                # For each candidate x not in block, compute how many new subsets it covers
                # A new subset is formed by x and (t-1)=2 elements from the current block
                # The current block has size s, so there are C(s, 2) pairs
                # For each pair (a,b) in block, the subset {a,b,x} has index lookup[x, pair_to_idx[(a,b)]]
                
                block_arr = np.array(block, dtype=np.int32)
                s = len(block)
                
                # Get all pairs in the current block
                block_pairs = list(itertools.combinations(block, 2))
                pair_indices = np.array([pair_to_idx[p] for p in block_pairs], dtype=np.int32)
                
                # Candidates: all points not in block
                candidates = [x for x in range(_V) if x not in block_set]
                if not candidates:
                    break
                
                # For each candidate x, compute the subset indices it would cover
                # and check how many are currently uncovered
                best_x = -1
                best_gain = -1
                
                for x in candidates:
                    # subset indices for x with each pair in block
                    sub_indices = lookup[x, pair_indices]
                    # Count how many are uncovered
                    gain = uncovered[sub_indices].sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                    
                block.append(best_x)
                block_set.add(best_x)
                
                # Update covered: mark all new subsets
                block_arr = np.array(block, dtype=np.int32)
                new_pairs = list(itertools.combinations(block_arr, 2))
                new_pair_indices = np.array([pair_to_idx[p] for p in new_pairs], dtype=np.int32)
                # All subsets formed by best_x with pairs that include best_x
                # Actually, we need to mark all subsets in the new block that weren't before
                # The new subsets are those that include best_x
                for a, b in itertools.combinations(block[:-1], 2):
                    sub = tuple(sorted((a, b, best_x)))
                    idx = _IDX_MAP[sub]
                    covered[idx] = True
            
            # Convert block to sorted list
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, _T):
                idx = _IDX_MAP[sub]
                uncovered[idx] = False
            
            # Check time
            if time.time() > deadline - 2.0:
                break
        
        return blocks

    # Run multiple random restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_once(i)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Print progress (optional, but helps debugging)
            # print(f"Restart {i}: {best_count} blocks")

    if best_blocks is None:
        # Fallback: simple deterministic greedy
        return _simple_greedy(v, k, t)
    
    return best_blocks

def _simple_greedy(v, k, t):
    """Simple deterministic greedy as fallback."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
