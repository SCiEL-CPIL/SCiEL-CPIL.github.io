"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + aggressive local search (removal/swap/add).
"""
import time
import itertools
import random
import numpy as np
from collections import Counter

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets as ints, and for every k-subset its coverage bitmap."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    # For each k-subset, the set of t-subset indices it covers
    k_subsets = list(itertools.combinations(range(v), k))
    # coverage: list of ints (bitmasks) for each k-subset
    coverage_bits = []
    for ks in k_subsets:
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
        coverage_bits.append(mask)
    
    # Also precompute t-subset to block index mapping for efficient swap search
    tsub_to_blocks = [[] for _ in range(n_t)]
    for i, mask in enumerate(coverage_bits):
        for j in range(n_t):
            if mask & (1 << j):
                tsub_to_blocks[j].append(i)
    
    return all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks


def _greedy_construct(v, k, t, rng):
    """Greedy set cover: repeatedly add the k-subset covering the most uncovered t-subsets."""
    all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks = _precompute(v, k, t)
    n_t = len(all_t)
    # uncovered as a bitmask
    uncovered = (1 << n_t) - 1
    blocks = []  # list of k-subset index
    
    # Optimization: Precompute list of (index, mask)
    # We need to find max popcount(coverage[i] & uncovered)
    # Using Python's int.bit_count() is fast.
    
    while uncovered and _elapsed() < _DEADLINE - 20:
        best_i = -1
        best_gain = 0
        # Iterate through all k-subsets
        # 50388 iterations * bit_count cost. 
        # bit_count on 3876-bit int is very fast.
        for i, mask in enumerate(coverage_bits):
            g = (mask & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
                if best_gain == 0:
                    break
        if best_i < 0:
            break
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
    return blocks, all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks


def _local_search(blocks, coverage_bits, n_t, k_subsets, tsub_to_blocks, rng, time_budget):
    """Removal + swap + add local search to minimize |blocks| while staying a cover."""
    blocks = list(blocks)
    
    # Helper to check if removing block i keeps coverage
    # We need union of all blocks except i to be full mask.
    # full_mask = (1 << n_t) - 1
    
    full_mask = (1 << n_t) - 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            # Check if removing b still covers everything
            # Compute union of all blocks except i
            cov = 0
            for idx, bb in enumerate(blocks):
                if idx != i:
                    cov |= coverage_bits[bb]
            
            if cov == full_mask:
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass ---
        # Try swapping a block with another to enable removal
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(len(k_subsets))
            if j in blocks:
                continue
            
            old = blocks[i]
            blocks[i] = j
            
            # Check if we can remove any block now
            removed = False
            for idx in range(len(blocks)):
                cov = 0
                for idx2 in range(len(blocks)):
                    if idx2 != idx:
                        cov |= coverage_bits[blocks[idx2]]
                
                if cov == full_mask:
                    blocks.pop(idx)
                    removed = True
                    break
            
            if removed:
                improved = True
            else:
                blocks[i] = old  # revert
        
        # --- Add pass: if not covered (shouldn't happen), add best block ---
        if len(blocks) > 0:
            cov = 0
            for b in blocks:
                cov |= coverage_bits[b]
            
            if cov != full_mask:
                missing = full_mask ^ cov
                best_i = -1
                best_g = 0
                for idx, mask in enumerate(coverage_bits):
                    g = (mask & missing).bit_count()
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    blocks.append(best_i)
                    
    return blocks


def _mask_to_list(mask):
    """Convert a bitmask to a list of set bit indices."""
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        result.append(lsb.bit_length() - 1)
        m ^= lsb
    return result


def _build_cover_count(blocks, coverage_bits):
    """Build a Counter of cover counts for each t-subset index."""
    cover_count = Counter()
    for b in blocks:
        for idx in _mask_to_list(coverage_bits[b]):
            cover_count[idx] += 1
    return cover_count


def _update_cover_count(cover_count, mask, delta):
    """Update cover_count by adding or removing the bits in mask."""
    for idx in _mask_to_list(mask):
        cover_count[idx] += delta
        if cover_count[idx] == 0:
            del cover_count[idx]


def _local_search_v2(blocks, coverage_bits, n_t, k_subsets, tsub_to_blocks, rng, time_budget):
    """Improved local search with better candidate generation and Counter-based tracking."""
    blocks = list(blocks)
    full_mask = (1 << n_t) - 1
    n_k = len(k_subsets)
    
    cover_count = _build_cover_count(blocks, coverage_bits)
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            mask = coverage_bits[b]
            # Check if any t-subset in this block has cover_count == 1
            removable = True
            for idx in _mask_to_list(mask):
                if cover_count[idx] == 1:
                    removable = False
                    break
            
            if removable:
                _update_cover_count(cover_count, mask, -1)
                blocks.pop(i)
                improved = True
                continue
            i += 1
        
        # --- Swap pass with better candidate generation ---
        swaps_per_iter = 10
        for _ in range(swaps_per_iter):
            if _elapsed() >= time_budget:
                break
            
            i = rng.randrange(len(blocks))
            b_out = blocks[i]
            mask_out = coverage_bits[b_out]
            
            # Find unique t-subsets covered by b_out
            unique_tsubs = []
            for idx in _mask_to_list(mask_out):
                if cover_count[idx] == 1:
                    unique_tsubs.append(idx)
            
            if not unique_tsubs:
                # Can just remove it
                _update_cover_count(cover_count, mask_out, -1)
                blocks.pop(i)
                improved = True
                break
            
            # Score multiple candidates: prefer those covering many unique tsubs
            # and overlapping well with existing blocks
            first_idx = unique_tsubs[0]
            candidates = tsub_to_blocks[first_idx][:]
            rng.shuffle(candidates)
            
            best_candidate = None
            best_score = -1
            blocks_set = set(blocks)
            
            for c in candidates[:50]:  # Limit to 50 candidates for speed
                if c in blocks_set:
                    continue
                mask_c = coverage_bits[c]
                
                # Check if c covers all unique_tsubs
                covers_all = True
                for idx in unique_tsubs:
                    if not (mask_c & (1 << idx)):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Score: number of unique tsubs covered (should be all) + overlap with existing
                # Prefer candidates that add more coverage overall
                score = len(unique_tsubs)
                # Add bonus for covering tsubs that are currently under-covered
                for idx in _mask_to_list(mask_c):
                    if cover_count[idx] <= 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate is not None:
                # Perform swap
                _update_cover_count(cover_count, mask_out, -1)
                _update_cover_count(cover_count, coverage_bits[best_candidate], 1)
                blocks[i] = best_candidate
                improved = True
        
        # --- Add pass: if not covered (shouldn't happen), add best block ---
        cov = 0
        for b in blocks:
            cov |= coverage_bits[b]
        
        if cov != full_mask:
            missing = full_mask ^ cov
            best_i = -1
            best_g = 0
            for idx, mask in enumerate(coverage_bits):
                g = (mask & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i >= 0:
                blocks.append(best_i)
                _update_cover_count(cover_count, coverage_bits[best_i], 1)
                
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage_bits, tsub_to_blocks = _precompute(v, k, t)
    n_t = len(all_t)

    best_blocks = None
    best_len = float('inf')
    
    # Try multiple greedy restarts with different random seeds
    max_restarts = 5
    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 30:
            break
        
        r = random.Random(12345 + restart * 7919)
        
        # Greedy construction
        greedy_blocks, _, _, _, _, _ = _greedy_construct(v, k, t, r)
        
        # Local search
        time_budget = min(_DEADLINE - 10, 25 + restart * 5)
        ls_blocks = _local_search_v2(greedy_blocks, coverage_bits, n_t, k_subsets, tsub_to_blocks, r, time_budget)
        
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    if best_blocks is None:
        # Fallback: single greedy + local search
        greedy_blocks, _, _, _, _, _ = _greedy_construct(v, k, t, rng)
        best_blocks = _local_search_v2(greedy_blocks, coverage_bits, n_t, k_subsets, tsub_to_blocks, rng, _DEADLINE - 5)

    blocks = best_blocks

    # Safety: ensure valid; if not, do a quick repair
    cov = 0
    for b in blocks:
        cov |= coverage_bits[b]
        
    full_mask = (1 << n_t) - 1
    if cov != full_mask:
        # repair with greedy add
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
