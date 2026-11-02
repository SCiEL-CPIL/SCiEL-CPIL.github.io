import itertools
import random
import time
import sys
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    
    Strategy:
    1. Build a diverse initial solution using multiple greedy variants with random tie-breaking
       and different point-ordering perturbations.
    2. Run a local search phase that attempts single-block swaps and deletions to reduce
       the number of blocks while maintaining coverage.
    
    The key insight is to use a more principled greedy that scores blocks by the number
    of uncovered t-subsets they cover, with careful tie-breaking to encourage diversity.
    """
    start_time = time.time()
    timeout = 135  # Safety margin before 150s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute the set of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsub_to_idx[ts])
        block_tsubs.append(covered)
    
    # Inverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, covered in enumerate(block_tsubs):
        for t_idx in covered:
            tsub_to_blocks[t_idx].append(b_idx)
    
    # Helper: Check if a block is redundant given current block set and tsub counts
    def is_redundant(b_idx, tsub_count):
        for t_idx in block_tsubs[b_idx]:
            if tsub_count[t_idx] == 1:
                return False
        return True
    
    # Helper: Try to remove redundant blocks from a solution
    def remove_redundant(current_blocks):
        if not current_blocks:
            return current_blocks
        
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                if time.time() - start_time > timeout - 5:
                    return current_blocks
                b_idx = current_blocks[i]
                if is_redundant(b_idx, tsub_count):
                    current_blocks.pop(i)
                    for t_idx in block_tsubs[b_idx]:
                        tsub_count[t_idx] -= 1
                    changed = True
                    break
        
        return current_blocks
    
    # Helper: Greedy construction with randomization
    def greedy_construction(seed_offset, use_random_order):
        if time.time() - start_time > timeout - 5:
            return None
        
        rng = random.Random(seed_offset)
        
        uncovered = list(range(n_tsubs))
        uncovered_set = set(uncovered)
        current_blocks = []
        
        while uncovered_set:
            if time.time() - start_time > timeout - 5:
                return current_blocks
            
            # Pick a random uncovered t-subset as target
            target_tsub_idx = rng.choice(uncovered)
            candidate_blocks = tsub_to_blocks[target_tsub_idx]
            
            if not candidate_blocks:
                break
            
            # Score each candidate by how many uncovered tsubs it covers
            best_b_idx = -1
            best_gain = -1
            best_rand = rng.random()
            
            for b_idx in candidate_blocks:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                
                rand_val = rng.random()
                if gain > best_gain or (gain == best_gain and rand_val > best_rand):
                    best_gain = gain
                    best_b_idx = b_idx
                    best_rand = rand_val
            
            if best_b_idx == -1:
                break
            
            current_blocks.append(best_b_idx)
            
            # Update uncovered
            for t_idx in block_tsubs[best_b_idx]:
                uncovered_set.discard(t_idx)
        
        # Remove redundant blocks
        current_blocks = remove_redundant(current_blocks)
        
        return current_blocks
    
    # Helper: Local search with single-block swaps
    def local_search(current_blocks, max_iterations, seed_offset):
        if not current_blocks or time.time() - start_time > timeout - 10:
            return current_blocks
        
        rng = random.Random(seed_offset)
        
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        current_size = len(current_blocks)
        best_size = current_size
        best_blocks = current_blocks.copy()
        
        no_improve_count = 0
        max_no_improve = 20
        
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout - 10:
                break
            
            if no_improve_count > max_no_improve:
                break
            
            # Try to find a swap that improves the solution
            # Pick a random block to potentially remove
            if not current_blocks:
                break
                
            remove_idx = rng.randrange(len(current_blocks))
            b_remove = current_blocks[remove_idx]
            
            # Find all tsubs that would become uncovered if we remove b_remove
            # These are tsubs with count == 1
            need_cover = []
            for t_idx in block_tsubs[b_remove]:
                if tsub_count[t_idx] == 1:
                    need_cover.append(t_idx)
            
            if not need_cover:
                # This block is redundant, just remove it
                current_blocks.pop(remove_idx)
                for t_idx in block_tsubs[b_remove]:
                    tsub_count[t_idx] -= 1
                best_size = len(current_blocks)
                best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue
            
            # Find candidate blocks that cover all of need_cover
            # Start with blocks covering the first need_cover tsub
            candidates = set(tsub_to_blocks[need_cover[0]])
            
            # Intersect with blocks covering other need_cover tsubs
            for t_idx in need_cover[1:]:
                candidates &= set(tsub_to_blocks[t_idx])
            
            if not candidates:
                no_improve_count += 1
                continue
            
            # Score candidates by how many uncovered tsubs they cover
            best_add = -1
            best_gain = -1
            
            for b_add in candidates:
                if b_add == b_remove:
                    continue
                # Count how many uncovered tsubs this block covers
                # (uncovered means not covered by any other block in current set)
                # Actually, we need to count tsubs that are covered ONLY by b_remove
                # plus any additional tsubs
                gain = 0
                for t_idx in block_tsubs[b_add]:
                    # If this tsub is covered by b_remove and no one else, 
                    # then adding b_add covers it
                    if tsub_count[t_idx] == 1 and t_idx in block_tsubs[b_remove]:
                        gain += 1
                    # Also count tsubs that are already uncovered (shouldn't happen in valid solution)
                
                if gain > best_gain:
                    best_gain = gain
                    best_add = b_add
            
            if best_add == -1:
                no_improve_count += 1
                continue
            
            # Perform the swap
            current_blocks[remove_idx] = best_add
            # Update tsub counts
            for t_idx in block_tsubs[b_remove]:
                tsub_count[t_idx] -= 1
            for t_idx in block_tsubs[best_add]:
                tsub_count[t_idx] += 1
            
            # The size doesn't change, so this is not an improvement in size
            # But it might enable future improvements
            no_improve_count += 1
        
        return best_blocks
    
    # Run multiple greedy constructions with different seeds
    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 15:
            break
        
        seed_offset = restart * 1000 + 42
        current = greedy_construction(seed_offset, True)
        
        if current is None:
            continue
        
        # Local search
        current = local_search(current, 100, seed_offset + 1000)
        
        if len(current) < best_size:
            best_size = len(current)
            best_blocks = current.copy()
    
    # Final cleanup
    if best_blocks is None:
        return []
    
    best_blocks = remove_redundant(best_blocks)
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return final_blocks
