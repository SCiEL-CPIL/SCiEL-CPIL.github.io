import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    total_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(total_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(total_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_k_blocks)
    
    # Precompute block -> list of subset indices
    block_subsets = []
    for block in all_k_blocks:
        subs = []
        for t_sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[t_sub])
        block_subsets.append(subs)
        
    # Precompute subset -> list of block indices
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)

    def greedy_initialize(rng):
        covered_count = [0] * n_t_subsets
        blocks = []
        # Use a list for uncovered for O(1) removal
        uncovered = list(range(n_t_subsets))
        
        while uncovered:
            # Pick random index from uncovered
            pos = rng.randint(0, len(uncovered) - 1)
            s_idx = uncovered[pos]
            
            # Pick random block covering s_idx
            candidates = subset_to_blocks[s_idx]
            if not candidates:
                # Fallback, shouldn't happen
                break
            b_idx = rng.choice(candidates)
            
            blocks.append(b_idx)
            
            # Update counts and uncovered
            for sub_idx in block_subsets[b_idx]:
                covered_count[sub_idx] += 1
                if covered_count[sub_idx] == 1:
                    # Find and remove from uncovered
                    # This is O(N) search. 
                    # To optimize, we can maintain a mapping from s_idx to position in uncovered?
                    # Or just let it be O(N) for now, N is 11k.
                    # 11k * 11k = 121M ops worst case. Might be slow in Python.
                    
                    # Better: use a set for uncovered? 
                    # Set removal is O(1). Iteration is O(N).
                    # We need to pick random from set. 
                    # Python set doesn't have random choice.
                    # We can keep a list and a set, or just use list and accept O(N) removal 
                    # if we are careful.
                    
                    # Actually, let's just use a set for uncovered and 
                    # convert to list for choice? 
                    # No, let's stick to list and do a linear scan for removal 
                    # but only for the ones we just covered.
                    
                    # Optimization: 
                    # We can just mark it as covered and filter the list later?
                    # No, we need to pick from uncovered.
                    
                    # Let's use a different strategy for Greedy:
                    # 1. Pick random s_idx from uncovered (list).
                    # 2. Pick random b_idx.
                    # 3. Add b_idx.
                    # 4. Mark all its subsets as covered.
                    # 5. Rebuild uncovered list? No, too slow.
                    
                    # Let's assume the O(N) removal is acceptable for ~1000-2000 blocks.
                    # 2000 blocks * 56 subsets * 11k avg scan = 1.2B ops. Too slow.
                    
                    # Alternative: Use a set for uncovered.
                    # To pick random from set, we can use `random.sample(uncovered_set, 1)`.
                    # This is O(1) or O(k).
                    
        return blocks, covered_count

    # Revised Greedy with Set
    def greedy_initialize_v2(rng):
        covered_count = [0] * n_t_subsets
        blocks = []
        uncovered = set(range(n_t_subsets))
        
        while uncovered:
            # Pick random from set
            s_idx = rng.choice(tuple(uncovered)) # O(N) conversion!
            # Better: random.sample(uncovered, 1)[0]
            s_idx = random.sample(uncovered, 1)[0]
            
            candidates = subset_to_blocks[s_idx]
            b_idx = rng.choice(candidates)
            
            blocks.append(b_idx)
            
            for sub_idx in block_subsets[b_idx]:
                covered_count[sub_idx] += 1
                if covered_count[sub_idx] == 1:
                    uncovered.discard(sub_idx)
                    
        return blocks, covered_count

    def remove_redundant(blocks, covered_count):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(blocks):
                b_idx = blocks[i]
                is_redundant = True
                for sub_idx in block_subsets[b_idx]:
                    if covered_count[sub_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub_idx in block_subsets[b_idx]:
                        covered_count[sub_idx] -= 1
                    blocks.pop(i)
                    changed = True
                else:
                    i += 1
        return blocks, covered_count

    def try_replace(blocks, covered_count, rng):
        if not blocks:
            return blocks, covered_count, False
            
        pos = rng.randint(0, len(blocks) - 1)
        old_b_idx = blocks[pos]
        
        criticals = []
        for sub_idx in block_subsets[old_b_idx]:
            if covered_count[sub_idx] == 1:
                criticals.append(sub_idx)
                
        if not criticals:
            for sub_idx in block_subsets[old_b_idx]:
                covered_count[sub_idx] -= 1
            blocks.pop(pos)
            return blocks, covered_count, True
            
        # Find intersection of blocks covering all criticals
        # Start with list for first critical
        candidates = list(subset_to_blocks[criticals[0]])
        
        for c in criticals[1:]:
            # Filter candidates to those also in subset_to_blocks[c]
            # Use set intersection for speed
            c_set = set(subset_to_blocks[c])
            candidates = [b for b in candidates if b in c_set]
            
            if not candidates:
                return blocks, covered_count, False
        
        # Remove old block from candidates
        candidates = [b for b in candidates if b != old_b_idx]
        
        if not candidates:
            return blocks, covered_count, False
            
        new_b_idx = rng.choice(candidates)
        
        # Apply
        for sub_idx in block_subsets[old_b_idx]:
            covered_count[sub_idx] -= 1
        for sub_idx in block_subsets[new_b_idx]:
            covered_count[sub_idx] += 1
        blocks[pos] = new_b_idx
        
        return blocks, covered_count, True

    start_time = time.time()
    time_limit = 115.0
    rng = random.Random(42)
    
    # Initialize
    blocks, covered_count = greedy_initialize_v2(rng)
    blocks, covered_count = remove_redundant(blocks, covered_count)
    
    best_len = len(blocks)
    best_blocks = list(blocks)
    
    # Local Search
    while time.time() - start_time < time_limit:
        for _ in range(50): # Inner loop iterations
            if time.time() - start_time > time_limit:
                break
                
            # Try replace
            blocks, covered_count, changed = try_replace(blocks, covered_count, rng)
            if changed:
                blocks, covered_count = remove_redundant(blocks, covered_count)
                
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
                
    return [list(all_k_blocks[idx]) for idx in best_blocks]
