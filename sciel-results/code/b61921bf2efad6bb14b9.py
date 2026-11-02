import time
import random
import itertools
import sys

def construct(v, k, t):
    deadline = time.time() + 130.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Helper to check if a set of blocks covers all t-subsets
    def is_valid(blocks):
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        return len(covered) == n_t_subsets

    # Helper to count how many unique t-subsets a block covers given current coverage
    def unique_coverage(block, covered_set):
        count = 0
        for sub in itertools.combinations(block, t):
            if sub not in covered_set:
                count += 1
        return count

    # Strategy: 
    # 1. Generate a good initial covering using a structured approach (e.g., greedy with strong heuristics)
    # 2. Apply local search (block removal and block exchange) to minimize count
    
    def generate_initial_covering(seed):
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy construction with randomization
        while uncovered and time.time() < deadline - 10.0:
            # Pick a random uncovered triple to start
            start_triple = rng.choice(list(uncovered))
            block = list(start_triple)
            block_set = set(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                # Optimization: Precompute (t-1) subsets of current block
                block_subsets = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    count = 0
                    for sub in block_subsets:
                        new_t_sub = tuple(sorted(sub + (x,)))
                        if new_t_sub in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    # If no gain, pick random to fill
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                else:
                    best_point = best_point if best_point != -1 else rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        
        return blocks

    def local_search(blocks, seed):
        rng = random.Random(seed)
        current_blocks = [tuple(b) for b in blocks]
        current_count = len(current_blocks)
        
        # Precompute coverage map: triple -> set of block indices covering it
        # This is expensive to update, so we'll maintain it carefully or recompute if necessary
        # For v=24, t=3, 2024 triples. Blocks ~70. 2024*70 is manageable.
        
        def build_coverage_map(blocks_list):
            cov_map = {}
            for i, b in enumerate(blocks_list):
                for sub in itertools.combinations(b, t):
                    if sub not in cov_map:
                        cov_map[sub] = set()
                    cov_map[sub].add(i)
            return cov_map

        cov_map = build_coverage_map(current_blocks)
        best_blocks = current_blocks
        best_count = current_count
        
        # Local Search Loop
        # Moves:
        # 1. Remove a block if it covers no unique triples (i.e., every triple in it is covered by >=1 other block)
        # 2. Swap a block with another block? (Complex, maybe skip for now)
        # 3. Add a block to cover missing triples? (Only if we broke validity, but we start valid and only remove valid blocks)
        
        # We will iterate:
        # - Check for removable blocks. If found, remove one and update map.
        # - If no removable blocks, try a random "swap" or "reconstruction" step? 
        # Actually, simple removal is the main driver for reduction. 
        # To escape local minima where no single block is removable, we might need to swap.
        # But swapping is hard to evaluate quickly without full re-coverage.
        
        # Let's stick to removal first, and if stuck, try a "random perturbation":
        # Remove 2 blocks and add 1 block that covers the union of their unique triples?
        # Or simply: restart with a different seed if stuck.
        
        stuck_count = 0
        while time.time() < deadline - 5.0:
            # Check for removable blocks
            removable = []
            for i, b in enumerate(current_blocks):
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if sub in cov_map and len(cov_map[sub]) == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    removable.append(i)
            
            if removable:
                # Remove a random removable block
                idx_to_remove = rng.choice(removable)
                # Update cov_map
                for sub in itertools.combinations(current_blocks[idx_to_remove], t):
                    if sub in cov_map:
                        cov_map[sub].remove(idx_to_remove)
                        if not cov_map[sub]:
                            del cov_map[sub]
                
                current_blocks.pop(idx_to_remove)
                current_count -= 1
                stuck_count = 0
            else:
                # No single block is removable.
                # Try a "2-for-1" move? 
                # Pick 2 blocks, remove them, see if we can cover their unique triples with 1 new block.
                # This is computationally heavy. 
                # Alternative: Random restart with a slightly modified initial seed?
                # Or: Try to swap a block with a "better" block?
                
                # Let's try a simple "kick": 
                # Pick a random block, remove it temporarily.
                # Identify uncovered triples.
                # Try to cover them with a new block.
                # If the new block covers all uncovered triples AND we don't increase the count (we removed 1, added 1, net 0), 
                # but maybe the new block is "better" for future removals?
                # This is complex. 
                
                # Instead, let's just break and try a new initial covering from a different seed.
                # This is a "portfolio" of local searches.
                break
        
        # Return the best found in this local search
        return current_blocks, current_count

    best_blocks_global = None
    best_count_global = float('inf')
    
    # Run multiple independent instances of (Greedy + Local Search)
    num_attempts = 500 # Upper bound, will break on time
    
    for i in range(num_attempts):
        if time.time() >= deadline - 10.0:
            break
            
        # 1. Generate initial covering
        init_blocks = generate_initial_covering(i)
        
        if not init_blocks:
            continue
            
        # 2. Validate initial covering
        if not is_valid(init_blocks):
            continue
            
        # 3. Local search
        optimized_blocks, optimized_count = local_search(init_blocks, i)
        
        if optimized_count < best_count_global:
            best_count_global = optimized_count
            best_blocks_global = optimized_blocks
            
        # Early termination if we hit a very good bound? 
        # Lower bound is 69. If we get 70, that's excellent.
        if best_count_global <= 70:
            break

    # Fallback if no valid covering found (should not happen)
    if best_blocks_global is None:
        # Simple greedy fallback
        uncovered = set(all_t_subsets)
        best_blocks_global = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    count = 0
                    for sub in itertools.combinations(block, t - 1):
                        if tuple(sorted(sub + (x,))) in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            block.sort()
            best_blocks_global.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
            
    return best_blocks_global
