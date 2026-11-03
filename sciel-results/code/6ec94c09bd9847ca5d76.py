import itertools
import time
import random
import sys
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for finalization
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is a bitset of t-subsets covered by block i
    block_masks = [0] * num_blocks
    # block_subset_indices[i] is a list of t-subset indices covered by block i
    block_subset_indices = [[] for _ in range(num_blocks)]
    
    # block_cover_masks[j] is a bitset of blocks covering t-subset j
    # This is used for candidate generation. 
    # Note: num_blocks ~ 50k, so this bitset is large (~6KB per subset). 
    # 3876 subsets * 6KB = ~23MB. Acceptable.
    block_cover_masks = [0] * num_t_subsets
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
            block_cover_masks[idx] |= (1 << i)
        block_masks[i] = mask
        block_subset_indices[i] = subs_idx

    full_mask = (1 << num_t_subsets) - 1
    
    # Helper: Greedy Set Cover using heap for efficiency
    def greedy_cover():
        uncovered = full_mask
        selected = []
        
        # Heap entries: (-gain, block_index)
        # We use a lazy heap: we push new gains, and when popping, we check if the gain is still valid.
        # Initial gains: number of subsets in block
        heap = []
        for i in range(num_blocks):
            gain = block_masks[i].bit_count()
            heapq.heappush(heap, (-gain, i))
        
        # To avoid re-scanning all blocks, we track the current gain of each block relative to 'uncovered'
        # Actually, recalculating gain for all blocks after each step is O(N_blocks * bit_count).
        # With N_blocks=50k and bit_count on ~3876 bits, this might be slow in Python.
        
        # Alternative: Standard greedy scan is O(N_blocks) per step.
        # Number of steps ~ 136. Total ops ~ 6.8M. This is fast enough in Python if optimized.
        # Let's stick to the scan but optimize the inner loop.
        
        while uncovered:
            best_gain = 0
            best_idx = -1
            
            # Scan all blocks
            # Optimization: skip blocks already selected? 
            # We can maintain a set of selected indices, but checking membership is O(1).
            # However, we just need to find the max gain.
            
            for i in range(num_blocks):
                # If i is already selected, gain is 0 (or we can just skip if we track it)
                # But we don't have a fast "selected" check for all i in the loop without a set.
                # Let's just calculate gain. If it's 0, it won't win.
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1 or best_gain == 0:
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # Helper: Reduce redundant blocks
    def reduce_blocks(block_indices):
        if not block_indices:
            return []
        
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
                    
            if not is_redundant:
                new_blocks.append(idx)
                
        return new_blocks

    # Helper: Find candidates for a set of t-subsets
    def find_candidates(E):
        if not E:
            return list(range(num_blocks))
        
        current_mask = block_cover_masks[E[0]]
        
        for e in E[1:]:
            current_mask &= block_cover_masks[e]
            if current_mask == 0:
                return []
                
        indices = []
        m = current_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        return indices

    # Initial Solution
    current_blocks = greedy_cover()
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Maintain cover_count dynamically for local search
    cover_count = [0] * num_t_subsets
    for idx in current_blocks:
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += 1

    def update_cover_count(idx, delta):
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += delta

    def is_redundant(idx):
        for s_idx in block_subset_indices[idx]:
            if cover_count[s_idx] == 1:
                return False
        return True

    def try_reduce():
        new_blocks = []
        for idx in current_blocks:
            if is_redundant(idx):
                update_cover_count(idx, -1)
            else:
                new_blocks.append(idx)
        return new_blocks

    # Local Search Loop
    no_improvement_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Delete redundant blocks
        new_blocks = try_reduce()
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            improved = True
            if len(current_blocks) < best_size:
                best_blocks = list(current_blocks)
                best_size = len(best_blocks)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            continue # Restart loop to re-evaluate
        
        # 2. 1-Swap: Replace a block with another that covers its unique elements
        indices_to_try = list(range(len(current_blocks)))
        random.shuffle(indices_to_try)
        
        for pos in indices_to_try:
            if time.time() > deadline:
                break
                
            b_idx = current_blocks[pos]
            
            # Find unique elements of b_idx
            unique_e = []
            for s_idx in block_subset_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    unique_e.append(s_idx)
                    
            if not unique_e:
                # Redundant, remove it
                update_cover_count(b_idx, -1)
                current_blocks.pop(pos)
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop as indices changed
            
            # Find candidates that cover unique_e
            candidates = find_candidates(unique_e)
            
            # Filter out b_idx itself
            if b_idx in candidates:
                candidates.remove(b_idx)
                
            if candidates:
                # Pick a random candidate to explore different paths
                new_idx = random.choice(candidates)
                
                # Perform swap
                update_cover_count(b_idx, -1)
                update_cover_count(new_idx, 1)
                current_blocks[pos] = new_idx
                
                improved = True
                no_improvement_count += 1
                
        # 3. 2-Exchange: Remove 2, Add 1
        max_2ex_attempts = 50
        attempts = 0
        while attempts < max_2ex_attempts and time.time() < deadline:
            attempts += 1
            if len(current_blocks) < 2:
                break
                
            i1, i2 = random.sample(range(len(current_blocks)), 2)
            b1 = current_blocks[i1]
            b2 = current_blocks[i2]
            
            # Find unique elements of b1 U b2 (subsets covered by b1 or b2, but count <= 2)
            # We need to cover all subsets that would become uncovered if we removed b1 and b2.
            # These are subsets where cover_count <= 2 and they are in b1 or b2.
            e_set = set()
            for s_idx in block_subset_indices[b1]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
            for s_idx in block_subset_indices[b2]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
                    
            if not e_set:
                continue
                
            unique_e_12 = list(e_set)
            
            # Find candidates that cover unique_e_12
            candidates = find_candidates(unique_e_12)
            
            # Filter out b1 and b2
            candidates = [c for c in candidates if c != b1 and c != b2]
            
            if candidates:
                new_idx = random.choice(candidates)
                
                # Perform 2-Exchange
                update_cover_count(b1, -1)
                update_cover_count(b2, -1)
                update_cover_count(new_idx, 1)
                
                if i1 > i2:
                    i1, i2 = i2, i1
                current_blocks.pop(i2)
                current_blocks.pop(i1)
                current_blocks.append(new_idx)
                
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop after a successful 2-exchange
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 50:
                # Perturbation: pick a random block, remove it, and repair greedily
                if len(current_blocks) > 0:
                    rem_pos = random.randint(0, len(current_blocks) - 1)
                    rem_idx = current_blocks[rem_pos]
                    
                    # Remove it
                    update_cover_count(rem_idx, -1)
                    current_blocks.pop(rem_pos)
                    
                    # Repair: find uncovered subsets
                    uncovered = []
                    for s_idx in range(num_t_subsets):
                        if cover_count[s_idx] == 0:
                            uncovered.append(s_idx)
                    
                    # Greedily add blocks to cover uncovered
                    # We only add blocks that cover at least one uncovered subset
                    # and prefer those that cover many.
                    uncovered_mask = 0
                    for s_idx in uncovered:
                        uncovered_mask |= (1 << s_idx)
                        
                    while uncovered_mask and time.time() < deadline:
                        best_gain = 0
                        best_idx = -1
                        
                        # Scan all blocks
                        for i in range(num_blocks):
                            if i in current_blocks:
                                continue
                            gain = (uncovered_mask & block_masks[i]).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                                
                        if best_idx == -1 or best_gain == 0:
                            break
                            
                        current_blocks.append(best_idx)
                        update_cover_count(best_idx, 1)
                        uncovered_mask &= ~block_masks[best_idx]
                        
                    new_blocks = try_reduce()
                    if len(new_blocks) < best_size:
                        best_blocks = list(new_blocks)
                        best_size = len(best_blocks)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    current_blocks = new_blocks
                else:
                    no_improvement_count += 1

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    
    # Final validation
    covered = [0] * num_t_subsets
    for block in final_blocks:
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            covered[idx] += 1
            
    if any(c == 0 for c in covered):
        # Fallback to greedy if invalid
        fallback = greedy_cover()
        fallback = reduce_blocks(fallback)
        final_blocks = [list(all_blocks[idx]) for idx in fallback]
        
    return final_blocks
