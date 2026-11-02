"""Improved covering algorithm using vectorized greedy with local search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks and their coverage as numpy arrays
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build coverage: for each block, list of t-subset indices
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_idx[ts])
        block_tsubs.append(np.array(tsubs, dtype=np.int32))
    
    # Build inverse: for each t-subset, list of blocks covering it (as numpy array)
    tsub_blocks = [np.array([], dtype=np.int32) for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx] = np.append(tsub_blocks[ts_idx], bi)
    
    # Precompute flat arrays for fast gain computation
    # block_tsub_flat: concatenated array of all t-subset indices across all blocks
    # block_tsub_offsets: offset into block_tsub_flat for each block
    block_tsub_flat_parts = []
    block_tsub_offsets = np.zeros(n_blocks + 1, dtype=np.int64)
    offset = 0
    for bi, tsubs in enumerate(block_tsubs):
        block_tsub_flat_parts.append(tsubs)
        offset += len(tsubs)
        block_tsub_offsets[bi + 1] = offset
    block_tsub_flat = np.concatenate(block_tsub_flat_parts) if block_tsub_flat_parts else np.array([], dtype=np.int32)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Compute initial gain using vectorized approach
        # gain[i] = number of uncovered t-subsets in block i
        gain = np.zeros(n_blocks, dtype=np.int32)
        if len(block_tsub_flat) > 0:
            # For each block, count how many of its t-subsets are uncovered
            # Use scatter-add: for each t-subset index in flat array, add 1 to its block's gain
            block_ids = np.repeat(np.arange(n_blocks, dtype=np.int32), np.diff(block_tsub_offsets))
            # Only add for uncovered t-subsets
            mask = uncovered[block_tsub_flat]
            if np.any(mask):
                np.add.at(gain, block_ids[mask], 1)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Find block with maximum gain
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            # Pick randomly from blocks with max gain for diversity
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(all_blocks[chosen])
            
            # Update uncovered and gain
            chosen_tsubs = block_tsubs[chosen]
            newly_covered = chosen_tsubs[uncovered[chosen_tsubs]]
            
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                # Decrement gain for all blocks covering each newly covered t-subset
                # For each newly covered t-subset, get all blocks that cover it
                # and decrement their gains
                all_affected_blocks = []
                for ts_idx in newly_covered:
                    all_affected_blocks.extend(tsub_blocks[ts_idx])
                if all_affected_blocks:
                    all_affected_blocks = np.array(all_affected_blocks, dtype=np.int32)
                    np.add.at(gain, all_affected_blocks, -1)
            
            # Mark chosen block as unusable
            gain[chosen] = -1
        
        return blocks_used
    
    def local_search(blocks, seed=0):
        """Try to improve by removing redundant blocks and swapping."""
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        
        # Build cover count and per-block t-subset lists
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_idx[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        # Remove redundant blocks
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        reduced = [blocks[i] for i in range(len(blocks)) if kept[i]]
        
        # Swap-based local search: try replacing a block with another
        # that covers all t-subsets uniquely covered by the original
        n_reduced = len(reduced)
        if n_reduced > 1 and time.time() < deadline - 5:
            # Rebuild cover_count for reduced set
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_indices = []
            for b in reduced:
                tsubs = []
                for ts in itertools.combinations(b, t):
                    idx = tsub_idx[ts]
                    tsubs.append(idx)
                    cover_count[idx] += 1
                block_tsub_indices.append(tsubs)
            
            # Try swaps
            max_swaps = 100
            for trial in range(max_swaps):
                if time.time() > deadline:
                    break
                
                i = rng.randint(0, n_reduced - 1)
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_indices[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block i is redundant, remove it
                    reduced.pop(i)
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    block_tsub_indices.pop(i)
                    n_reduced -= 1
                    continue
                
                # Try to find a block that covers all unique_tsubs
                # and doesn't create new uncovered t-subsets
                found = False
                # Sample some candidate blocks
                n_candidates = min(200, n_blocks)
                candidates = rng.sample(range(n_blocks), n_candidates)
                for ci in candidates:
                    if time.time() > deadline:
                        break
                    cand_tsubs = set(block_tsubs[ci].tolist())
                    if all(ts in cand_tsubs for ts in unique_tsubs):
                        # Check: removing block i and adding block ci
                        # doesn't leave any t-subset uncovered
                        ok = True
                        for idx in block_tsub_indices[i]:
                            if cover_count[idx] == 1 and idx not in cand_tsubs:
                                ok = False
                                break
                        if ok:
                            # Perform swap
                            # Remove block i
                            for idx in block_tsub_indices[i]:
                                cover_count[idx] -= 1
                            # Add block ci
                            for ts_idx in block_tsubs[ci]:
                                cover_count[ts_idx] += 1
                            reduced[i] = all_blocks[ci]
                            block_tsub_indices[i] = block_tsubs[ci].tolist()
                            found = True
                            break
                if found:
                    continue
                # No improvement found for this block, try next
        
        return reduced
    
    # Run greedy with multiple seeds
    num_trials = 50
    for seed in range(num_trials):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(seed)
        if not blocks:
            continue
        
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Early termination if we hit a very good result
        if best_len <= 135:
            break
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0)
    
    return best_blocks
