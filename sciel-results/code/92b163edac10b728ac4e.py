import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Bitset for each block
    block_bitsets = [0] * num_blocks
    # List of indices for each block (for counts update)
    block_t_indices = []
    # t_subset_to_blocks for kick
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        bitmask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            bitmask |= (1 << idx)
            indices.append(idx)
            t_subset_to_blocks[idx].append(b_idx)
        block_bitsets[b_idx] = bitmask
        block_t_indices.append(indices)

    # 2. Greedy Initialization with Bitsets
    def greedy_init():
        counts = [0] * num_t_subsets
        # uncovered as bitset
        uncovered_mask = (1 << num_t_subsets) - 1
        blocks = []
        
        while uncovered_mask:
            best_block = -1
            max_cov = -1
            # Iterate blocks
            # Optimization: skip blocks already in? No, greedy adds unique.
            # But if we add a block, we don't add it again.
            used_blocks = set()
            
            for b_idx in range(num_blocks):
                if b_idx in used_blocks:
                    continue
                # Calculate coverage
                cov_mask = block_bitsets[b_idx] & uncovered_mask
                cov_count = cov_mask.bit_count()
                
                if cov_count > max_cov:
                    max_cov = cov_count
                    best_block = b_idx
                    if max_cov == len(block_t_indices[best_block]):
                        break
            
            if best_block == -1:
                break
                
            used_blocks.add(best_block)
            blocks.append(best_block)
            
            # Update counts and uncovered_mask
            for idx in block_t_indices[best_block]:
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered_mask &= ~(1 << idx)
                    
        return blocks, counts

    # 3. Local Search
    def local_search(blocks, counts):
        while time.time() < deadline:
            removed_any = False
            new_blocks = []
            for b_idx in blocks:
                is_redundant = True
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    removed_any = True
                else:
                    new_blocks.append(b_idx)
            
            blocks = new_blocks
            
            if not removed_any:
                # Kick
                if not blocks:
                    break
                
                b_out = random.choice(blocks)
                lost_indices = [idx for idx in block_t_indices[b_out] if counts[idx] == 1]
                
                if not lost_indices:
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    blocks.remove(b_out)
                    continue
                
                candidate_blocks = None
                for idx in lost_indices:
                    if candidate_blocks is None:
                        candidate_blocks = set(t_subset_to_blocks[idx])
                    else:
                        candidate_blocks.intersection_update(t_subset_to_blocks[idx])
                
                if not candidate_blocks:
                    continue
                
                current_block_set = set(blocks)
                candidates = [b for b in candidate_blocks if b not in current_block_set]
                
                if not candidates:
                    continue
                
                b_in = random.choice(candidates)
                
                # Swap
                for idx in block_t_indices[b_out]:
                    counts[idx] -= 1
                blocks.remove(b_out)
                
                for idx in block_t_indices[b_in]:
                    counts[idx] += 1
                blocks.append(b_in)
                
                removed_any = True
            
            if time.time() > deadline - 5:
                break
                
        return blocks

    blocks, counts = greedy_init()
    blocks = local_search(blocks, counts)
    
    result = [list(all_blocks[b_idx]) for b_idx in blocks]
    return result
