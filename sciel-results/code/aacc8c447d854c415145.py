import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitsets for blocks
    block_bitsets = [0] * n_blocks
    # Precompute mapping from t-subset to blocks containing it
    t_sub_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(all_blocks):
        bits = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            bits |= (1 << idx)
            t_sub_to_blocks[idx].append(i)
        block_bitsets[i] = bits

    def greedy_construct(seed):
        random.seed(seed)
        # Initialize uncovered as a set of indices for easy random pick, 
        # but use bitset for fast intersection count.
        uncovered_bits = (1 << n_t_subs) - 1
        # We also need to pick a random uncovered t-subset.
        # We can maintain a list of uncovered indices and remove from it.
        # Or, just pick a random index and check if it's uncovered.
        # Given n_t_subs ~ 11k, picking random and checking is fine.
        
        used_blocks = []
        
        while uncovered_bits:
            # Pick a random uncovered t-subset
            # To ensure we pick an uncovered one, we can try a few times
            # or maintain a list.
            # Let's maintain a list of uncovered indices for O(1) random pick.
            # But updating the list is expensive.
            # Alternative: Just pick random index, if covered, try again.
            # Probability of being covered increases, but early on it's high.
            # Better: Maintain a list of uncovered indices.
            
            # Let's restart with a list of uncovered indices.
            # This is O(N) to initialize, O(N) to update.
            # N = 11628.
            
            pass # Placeholder
            
        # This part needs careful implementation of the uncovered set.
        # Using a list of uncovered indices:
        # 1. Start with list(range(n_t_subs))
        # 2. Pick random index from list.
        # 3. Remove it from list? No, we remove all covered ones.
        # 4. To remove covered ones, we need to know which ones are covered.
        # 5. We can just iterate through the list and filter? O(N) per step.
        # 6. 300 steps * 11628 = 3.5M operations. This is fine.
        
        pass

    # Refined Greedy
    def greedy_construct(seed):
        random.seed(seed)
        uncovered_list = list(range(n_t_subs))
        uncovered_bits = (1 << n_t_subs) - 1
        used_blocks = []
        
        while uncovered_list:
            # Pick random uncovered t-subset
            rand_idx = random.randint(0, len(uncovered_list) - 1)
            target_t_idx = uncovered_list[rand_idx]
            
            # Get candidate blocks covering this t-subset
            candidates = t_sub_to_blocks[target_t_idx]
            
            best_block = -1
            max_cov = -1
            
            for b_idx in candidates:
                # Count how many uncovered t-subsets this block covers
                # Intersection of block_bitsets[b_idx] and uncovered_bits
                inter = block_bitsets[b_idx] & uncovered_bits
                # Count bits
                cov = bin(inter).count('1')
                if cov > max_cov:
                    max_cov = cov
                    best_block = b_idx
            
            if best_block == -1:
                # Should not happen
                break
                
            # Add block
            used_blocks.append(best_block)
            
            # Update uncovered
            # New uncovered bits
            new_uncovered_bits = uncovered_bits & ~block_bitsets[best_block]
            
            # Update uncovered_list
            # We need to remove all indices that are now covered.
            # This is expensive if we do it by checking each.
            # But we can just rebuild the list from new_uncovered_bits?
            # Iterating through all n_t_subs (11k) to check bit is O(N).
            # 300 * 11k = 3.3M ops. Fine.
            
            new_uncovered_list = []
            for i in range(n_t_subs):
                if (new_uncovered_bits >> i) & 1:
                    new_uncovered_list.append(i)
                    
            uncovered_list = new_uncovered_list
            uncovered_bits = new_uncovered_bits
            
            if time.time() > deadline:
                return None
                
        return used_blocks

    def local_search(blocks):
        # blocks: list of block indices
        # Try to remove redundant blocks
        coverage_count = [0] * n_t_subs
        for b_idx in blocks:
            # We need the t-subsets covered by this block
            # We didn't store block_t_indices in the previous snippet, 
            # but we can re-derive or store it.
            # Let's store block_t_indices in the precomputation phase.
            pass
            
    # Re-do precomputation to store block_t_indices
    block_t_indices = [[] for _ in range(n_blocks)]
    for i, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            block_t_indices[i].append(idx)

    def local_search(blocks):
        coverage_count = [0] * n_t_subs
        for b_idx in blocks:
            for t_idx in block_t_indices[b_idx]:
                coverage_count[t_idx] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_t_indices[b_idx]:
                        coverage_count[t_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    if time.time() > deadline:
                        return current_blocks
                    break
        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        raw_blocks = greedy_construct(i)
        if raw_blocks is None:
            continue
            
        optimized_blocks = local_search(raw_blocks)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_blocks = optimized_blocks
            
    if best_blocks is None:
        raw_blocks = greedy_construct(0)
        if raw_blocks:
            best_blocks = raw_blocks
        else:
            return []

    result = [list(all_blocks[i]) for i in best_blocks]
    return result
