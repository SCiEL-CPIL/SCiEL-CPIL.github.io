"""Covering design C(19,7,4) via vectorized greedy + local search with redundancy removal."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute all t-subsets and map to indices for vectorized operations
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    ts_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each k-subset (block), which t-subsets it covers
    # But we don't want to store all C(v,k) blocks upfront - too many
    # Instead, we'll compute coverage on the fly using bitmasks or numpy arrays

    # For v=19, k=7, t=4: C(19,4) = 3876 t-subsets
    # We'll use numpy boolean arrays for coverage tracking

    def get_block_tsubsets(block):
        """Get indices of all t-subsets covered by a block."""
        return [ts_to_idx[ts] for ts in itertools.combinations(sorted(block), t)]

    def greedy_construct():
        """Greedy: repeatedly pick the block that covers the most uncovered t-subsets."""
        uncovered_mask = np.ones(n_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            best_block = None
            best_gain = -1
            
            # Try multiple seed t-subsets
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Sample up to 100 seeds from uncovered
            n_seeds = min(100, len(uncovered_indices))
            seed_indices = rng.sample(list(uncovered_indices), n_seeds)
            
            for seed_idx in seed_indices:
                seed_subset = list(all_tsubsets[seed_idx])
                block = list(seed_subset)
                
                while len(block) < k:
                    best_x = -1
                    best_g = -1
                    block_set = set(block)
                    
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Count how many uncovered t-subsets would be covered by adding x
                        g = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            if ts in ts_to_idx:
                                idx = ts_to_idx[ts]
                                if uncovered_mask[idx]:
                                    g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x == -1:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                
                if len(block) < k:
                    continue
                
                block.sort()
                # Count coverage
                covered_indices = get_block_tsubsets(block)
                gain = sum(1 for idx in covered_indices if uncovered_mask[idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Fallback: random block
                best_block = sorted(rng.sample(range(v), k))
            
            blocks.append(best_block)
            # Update uncovered mask
            for idx in get_block_tsubsets(best_block):
                uncovered_mask[idx] = False
        
        return blocks, len(blocks)

    # Initial greedy construction
    blocks, n_blocks = greedy_construct()
    best_blocks = list(blocks)
    best_n = n_blocks

    def compute_coverage_counts(blocks_list):
        """Count how many blocks cover each t-subset."""
        counts = np.zeros(n_tsubsets, dtype=np.int32)
        for b in blocks_list:
            for idx in get_block_tsubsets(b):
                counts[idx] += 1
        return counts

    def try_remove_redundant(blocks_list):
        """Iteratively remove blocks that have no unique t-subsets."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            counts = compute_coverage_counts(blocks_list)
            
            for i in range(len(blocks_list)):
                block = blocks_list[i]
                covered_indices = get_block_tsubsets(block)
                # Check if any t-subset is uniquely covered by this block
                if not np.any(counts[covered_indices] == 1):
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    def is_valid(blocks_list):
        """Check if all t-subsets are covered."""
        covered_mask = np.zeros(n_tsubsets, dtype=bool)
        for b in blocks_list:
            for idx in get_block_tsubsets(b):
                covered_mask[idx] = True
        return np.all(covered_mask)

    def local_search(blocks_list):
        """Try to improve by removing redundant blocks and swapping."""
        current = list(blocks_list)
        current = try_remove_redundant(current)
        best = list(current)
        best_n = len(current)
        
        start_time = time.time()
        while time.time() - start_time < 35 and time.time() < deadline:
            if len(current) <= 10:
                break
            
            # Try random removal + repair
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)
            
            # Check if still valid
            if is_valid(current):
                if len(current) < best_n:
                    best = list(current)
                    best_n = len(current)
                current = try_remove_redundant(current)
                if len(current) < best_n:
                    best = list(current)
                    best_n = len(current)
            else:
                # Find uncovered t-subsets
                covered_mask = np.zeros(n_tsubsets, dtype=bool)
                for b in current:
                    for idx2 in get_block_tsubsets(b):
                        covered_mask[idx2] = True
                uncovered_indices = np.where(~covered_mask)[0]
                
                if len(uncovered_indices) > 0:
                    # Add greedy block for uncovered
                    best_block = None
                    best_gain = -1
                    
                    n_seeds = min(20, len(uncovered_indices))
                    seed_indices = rng.sample(list(uncovered_indices), n_seeds)
                    
                    for seed_idx in seed_indices:
                        seed_subset = list(all_tsubsets[seed_idx])
                        block = list(seed_subset)
                        
                        while len(block) < k:
                            best_x = -1
                            best_g = -1
                            block_set = set(block)
                            
                            for x in range(v):
                                if x in block_set:
                                    continue
                                g = 0
                                for rest in itertools.combinations(block, t - 1):
                                    ts = tuple(sorted(rest + (x,)))
                                    if ts in ts_to_idx:
                                        idx3 = ts_to_idx[ts]
                                        if not covered_mask[idx3]:
                                            g += 1
                                if g > best_g:
                                    best_g = g
                                    best_x = x
                            if best_x == -1:
                                break
                            block.append(best_x)
                            block_set.add(best_x)
                        
                        if len(block) == k:
                            block.sort()
                            covered_indices = get_block_tsubsets(block)
                            gain = sum(1 for idx4 in covered_indices if not covered_mask[idx4])
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                    
                    if best_block:
                        current.append(best_block)
                    else:
                        current.append(removed)
                        continue
                
                # Try to remove more
                current = try_remove_redundant(current)
                if len(current) < best_n:
                    best = list(current)
                    best_n = len(current)
        
        return best, best_n

    # Run local search
    improved_blocks, improved_n = local_search(best_blocks)
    if improved_n < best_n:
        best_blocks = improved_blocks
        best_n = improved_n

    # Final check and cleanup
    best_blocks = try_remove_redundant(best_blocks)
    
    # Ensure validity
    if not is_valid(best_blocks):
        # Fallback to original greedy
        best_blocks = list(blocks)
    
    return [sorted(b) for b in best_blocks]
