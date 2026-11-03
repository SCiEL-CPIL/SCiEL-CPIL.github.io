import itertools
import random
import time
import sys
import math
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    # Deadline guard
    deadline = time.time() + 145
    
    # 1. Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute masks for all blocks
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Precompute t-subsets within each block for faster gain calculation
    block_tsubs = []
    for b in all_blocks:
        tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs)
        
    # Helper to iterate set bits
    def iter_set_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # 2. Greedy Initialization - optimized
    def greedy_cover(rng):
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        # Randomize t-subset order
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered_mask:
            if time.time() > deadline - 20:
                return None
            
            # Find first uncovered t-subset in order
            seed_idx = -1
            for idx in order:
                if (uncovered_mask >> idx) & 1:
                    seed_idx = idx
                    break
            if seed_idx == -1:
                break
                
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            # Greedily add points - use precomputed t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                        
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                        block.append(best_point)
                        block_set.add(best_point)
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            bi = block_to_idx[tuple(block)]
            block_mask = block_masks[bi]
            uncovered_mask &= ~block_mask
            
        if uncovered_mask:
            return None
        return blocks

    # 3. Remove Redundant Blocks
    def remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
        
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                redundant = True
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
        return blocks

    # 4. Local Search: add-remove cycles to reduce block count
    def local_search(blocks, deadline):
        if not blocks:
            return blocks
        
        # Precompute for each block: which t-subsets it covers
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
        
        # For each t-subset, list which block indices cover it
        tsub_coverers = defaultdict(list)
        for i, bi in enumerate(block_indices):
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                tsub_coverers[idx].append(i)
        
        improved = True
        max_rounds = 30
        round_num = 0
        
        while improved and time.time() < deadline - 15 and round_num < max_rounds:
            improved = False
            round_num += 1
            
            # Try to find a block that can be removed and its gaps filled by one new block
            # Strategy: for each non-essential block, check if there exists a single block
            # that covers all t-subsets that would become uncovered
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                
                bi = block_indices[i]
                mask = block_masks[bi]
                
                # Find t-subsets that would become uncovered if we remove block i
                gaps = []
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        gaps.append(idx)
                
                if not gaps:
                    # Block is redundant - remove it
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                        tsub_coverers[idx].remove(i)
                    blocks.pop(i)
                    block_indices.pop(i)
                    # Re-index tsub_coverers for blocks after i
                    for idx in range(n_tsubs):
                        if idx in tsub_coverers:
                            tsub_coverers[idx] = [j - 1 if j > i else j for j in tsub_coverers[idx] if j != i]
                    improved = True
                    break
                
                # Check if any single block covers all the gaps
                # The candidate block must cover all t-subsets in gaps
                # And it's OK if it also covers other already-covered t-subsets
                found_replacement = False
                for gap_idx in gaps:
                    # Candidate blocks must be in tsub_coverers[gap_idx]
                    for cand_bi in tsub_coverers[gap_idx]:
                        if cand_bi == i:
                            continue
                        # Check if this candidate covers all gaps
                        cand_mask = block_masks[cand_bi]
                        all_covered = True
                        for g in gaps:
                            if not (cand_mask >> g) & 1:
                                all_covered = False
                                break
                        if all_covered:
                            # Great! We can swap: remove block i, add block cand_bi
                            # But wait - if cand_bi is already in our set, we can't add it twice
                            if cand_bi in block_indices:
                                continue
                            # Perform the swap
                            # Remove block i
                            for idx in iter_set_bits(mask):
                                cover_count[idx] -= 1
                                tsub_coverers[idx].remove(i)
                            blocks.pop(i)
                            block_indices.pop(i)
                            # Re-index
                            for idx in range(n_tsubs):
                                if idx in tsub_coverers:
                                    tsub_coverers[idx] = [j - 1 if j > i else j for j in tsub_coverers[idx] if j != i]
                            
                            # Add candidate block
                            new_i = len(blocks)
                            blocks.append(list(all_blocks[cand_bi]))
                            block_indices.append(cand_bi)
                            cand_mask = block_masks[cand_bi]
                            for idx in iter_set_bits(cand_mask):
                                cover_count[idx] += 1
                                tsub_coverers[idx].append(new_i)
                            
                            improved = True
                            found_replacement = True
                            break
                    if found_replacement:
                        break
                
                if found_replacement:
                    break
        
        # Final redundancy removal pass
        return aggressive_redundancy(blocks, deadline)

    # 5. More aggressive redundancy removal with multiple passes
    def aggressive_redundancy(blocks, deadline):
        if not blocks:
            return blocks
        
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
        
        changed = True
        passes = 0
        max_passes = 20
        
        while changed and time.time() < deadline - 5 and passes < max_passes:
            changed = False
            passes += 1
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                redundant = True
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
        
        return blocks

    # Main
    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')
    
    # Try several greedy starts with different random seeds
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        rng_seed = 42 + restart * 137
        local_rng = random.Random(rng_seed)
        init_blocks = greedy_cover(local_rng)
        if init_blocks:
            # Apply local search (add-remove cycles)
            searched = local_search(init_blocks, deadline)
            if len(searched) < best_count:
                best_count = len(searched)
                best_blocks = [list(b) for b in searched]
    
    if best_blocks is None:
        # Fallback: Simple greedy without masks to ensure validity
        covered = set()
        blocks = []
        while len(covered) < n_tsubs:
            best_block = None
            best_gain = -1
            for b in all_blocks:
                gain = 0
                for ts in itertools.combinations(b, t):
                    if ts not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None:
                break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                covered.add(ts)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Final validation and cleanup
    if best_blocks:
        # Verify all t-subsets are covered
        all_covered = set()
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                all_covered.add(ts)
        if len(all_covered) < n_tsubs:
            # Incomplete - do a final greedy fill
            remaining = set(all_tsubs) - all_covered
            for ts in list(remaining):
                # Find any block containing this t-subset
                for b in all_blocks:
                    if set(ts).issubset(set(b)):
                        best_blocks.append(list(b))
                        for ts2 in itertools.combinations(b, t):
                            all_covered.add(ts2)
                        break
            # Final redundancy removal
            best_blocks = aggressive_redundancy(best_blocks, deadline)
            best_count = len(best_blocks)
    
    return best_blocks
