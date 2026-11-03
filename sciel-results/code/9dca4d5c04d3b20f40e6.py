import itertools
import time
import random
import sys
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
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
                    
            if is_redundant:
                pass
            else:
                new_blocks.append(idx)
                
        return new_blocks

    # Primal-dual approximation algorithm
    def primal_dual_cover():
        uncovered = full_mask
        selected = []
        dual_weights = [0.0] * num_t_subsets  # y_j for each t-subset
        
        # We'll use a priority queue to find blocks with best "value"
        # Value = sum of dual weights of newly covered subsets / cost (cost=1)
        # But we need to recompute as dual weights change
        
        while uncovered:
            # Increase dual weights for uncovered subsets
            # We'll do this in batches to avoid infinite loop
            # Standard primal-dual: increase y_j for all uncovered j until some block becomes "tight"
            
            # Find the block that would become tight first
            # A block i becomes tight when sum_{j in block_i, j uncovered} y_j >= 1
            # Initially all y_j = 0, so we increase them uniformly
            
            # For simplicity, we'll use a randomized approach:
            # Pick a random uncovered subset, add its block to candidates
            # Then greedily pick the block with max marginal gain weighted by duals
            
            # Actually, let's use a simpler approach:
            # 1. Find all uncovered subsets
            # 2. For each block, compute gain = number of newly covered subsets
            # 3. Weight the gain by a random factor to break ties and escape local minima
            # 4. Pick the block with max weighted gain
            
            best_gain = 0
            best_idx = -1
            
            # Sample some blocks to avoid O(N) scan every time
            # If N is small enough, scan all; otherwise sample
            if num_blocks <= 5000:
                candidates = list(range(num_blocks))
            else:
                candidates = random.sample(range(num_blocks), min(5000, num_blocks))
            
            for i in candidates:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # Initial Solution using primal-dual
    current_blocks = primal_dual_cover()
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    # Local Search
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

    # Reset cover_count for current_blocks
    cover_count = [0] * num_t_subsets
    for idx in current_blocks:
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += 1

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
            continue
        
        # 2. 1-Swap: Replace a block with another that covers its unique elements
        indices_to_try = list(range(len(current_blocks)))
        random.shuffle(indices_to_try)
        
        for pos in indices_to_try:
            if time.time() > deadline:
                break
                
            b_idx = current_blocks[pos]
            
            unique_e = []
            for s_idx in block_subset_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    unique_e.append(s_idx)
                    
            if not unique_e:
                update_cover_count(b_idx, -1)
                current_blocks.pop(pos)
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break
            
            candidates = find_candidates(unique_e)
            
            if b_idx in candidates:
                candidates.remove(b_idx)
                
            if candidates:
                new_idx = random.choice(candidates)
                
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
            
            candidates = find_candidates(unique_e_12)
            candidates = [c for c in candidates if c != b1 and c != b2]
            
            if candidates:
                new_idx = random.choice(candidates)
                
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
                break
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 100:
                pass

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks
