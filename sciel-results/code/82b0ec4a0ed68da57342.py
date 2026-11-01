"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: numpy-accelerated greedy + structured local search with counter tracking.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets, k-subsets, and coverage relationships."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # For each k-subset, which t-subset indices it covers
    block_to_tsubs = []
    for ks in k_subsets:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_index[ts])
        block_to_tsubs.append(np.array(covered, dtype=np.int32))
    
    # For each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Coverage bitmask for each block (for fast union checks)
    coverage_bits = []
    for tsubs in block_to_tsubs:
        mask = 0
        for s in tsubs:
            mask |= (1 << s)
        coverage_bits.append(mask)
    
    return all_t, t_index, k_subsets, block_to_tsubs, tsub_to_blocks_np, coverage_bits, n_t, n_k


def _greedy_construct(v, k, t, block_to_tsubs, tsub_to_blocks_np, n_t, n_k, rng):
    """Greedy set cover with numpy gain tracking."""
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(n_k, k_choose_t, dtype=np.int32)
    uncovered = np.ones(n_t, dtype=bool)
    selected = []
    
    while uncovered.any():
        if _elapsed() > _DEADLINE - 25:
            break
        
        b = int(np.argmax(gains))
        if gains[b] == 0:
            break
        
        selected.append(b)
        
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains[tsub_to_blocks_np[s]] -= 1
    
    return selected


def _local_search(blocks, block_to_tsubs, tsub_to_blocks_np, coverage_bits, n_t, n_k, rng, time_budget):
    """Local search: removal + structured swaps with counter tracking."""
    blocks = list(blocks)
    n_blocks = len(blocks)
    
    # Build cover count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        for s in block_to_tsubs[b]:
            cover_count[s] += 1
    
    improved = True
    swap_attempts = 0
    max_swaps_per_iter = 20
    
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Check if any t-subset in this block has cover_count == 1
            removable = True
            for s in block_to_tsubs[b]:
                if cover_count[s] == 1:
                    removable = False
                    break
            
            if removable:
                for s in block_to_tsubs[b]:
                    cover_count[s] -= 1
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass ---
        for _ in range(max_swaps_per_iter):
            if _elapsed() >= time_budget:
                break
            
            if not blocks:
                break
            
            i = rng.randrange(len(blocks))
            b_out = blocks[i]
            
            # Find unique t-subsets covered by b_out
            unique_tsubs = []
            for s in block_to_tsubs[b_out]:
                if cover_count[s] == 1:
                    unique_tsubs.append(s)
            
            if not unique_tsubs:
                # Can just remove it
                for s in block_to_tsubs[b_out]:
                    cover_count[s] -= 1
                blocks.pop(i)
                improved = True
                break
            
            # Try to find a replacement block covering all unique_tsubs
            first_idx = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_idx]
            rng_candidates = list(candidates)
            rng.shuffle(rng_candidates)
            
            blocks_set = set(blocks)
            best_candidate = None
            best_score = -1
            
            for c in rng_candidates[:80]:
                if c in blocks_set:
                    continue
                
                # Check if c covers all unique_tsubs
                covers_all = True
                for idx in unique_tsubs:
                    if idx not in block_to_tsubs[c]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Score: prefer candidates that cover more under-covered t-subsets
                score = len(unique_tsubs)
                for s in block_to_tsubs[c]:
                    if cover_count[s] <= 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate is not None:
                # Perform swap
                for s in block_to_tsubs[b_out]:
                    cover_count[s] -= 1
                for s in block_to_tsubs[best_candidate]:
                    cover_count[s] += 1
                blocks[i] = best_candidate
                improved = True
        
        # --- Add pass: if not covered, add best block ---
        cov_mask = 0
        for b in blocks:
            cov_mask |= coverage_bits[b]
        
        full_mask = (1 << n_t) - 1
        if cov_mask != full_mask:
            missing = full_mask ^ cov_mask
            best_i = -1
            best_g = 0
            for idx, mask in enumerate(coverage_bits):
                g = (mask & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i >= 0:
                blocks.append(best_i)
                for s in block_to_tsubs[best_i]:
                    cover_count[s] += 1
    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, block_to_tsubs, tsub_to_blocks_np, coverage_bits, n_t, n_k = _precompute(v, k, t)

    best_blocks = None
    best_len = float('inf')
    
    # Multi-start with different random seeds
    max_restarts = 4
    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 30:
            break
        
        r = random.Random(12345 + restart * 7919)
        
        # Greedy construction
        greedy_blocks = _greedy_construct(v, k, t, block_to_tsubs, tsub_to_blocks_np, n_t, n_k, r)
        
        # Local search
        time_budget = min(_DEADLINE - 10, 20 + restart * 8)
        ls_blocks = _local_search(greedy_blocks, block_to_tsubs, tsub_to_blocks_np, coverage_bits, n_t, n_k, r, time_budget)
        
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    if best_blocks is None:
        greedy_blocks = _greedy_construct(v, k, t, block_to_tsubs, tsub_to_blocks_np, n_t, n_k, rng)
        best_blocks = _local_search(greedy_blocks, block_to_tsubs, tsub_to_blocks_np, coverage_bits, n_t, n_k, rng, _DEADLINE - 5)

    blocks = best_blocks

    # Safety: ensure valid; if not, do a quick repair
    cov = 0
    for b in blocks:
        cov |= coverage_bits[b]
        
    full_mask = (1 << n_t) - 1
    if cov != full_mask:
        while cov != full_mask and _elapsed() < _DEADLINE - 2:
            missing = full_mask ^ cov
            best_i = -1
            best_g = 0
            for idx, mask in enumerate(coverage_bits):
                g = (mask & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage_bits[best_i]

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
