import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Set Cover for initialization
    followed by Simulated Annealing to minimize the number of blocks.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # 1. Pre-compute all possible k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = {}
    for b in all_blocks:
        subs = tuple(sorted(itertools.combinations(b, t)))
        block_to_subsets[b] = subs
        
    # Map from subset to list of blocks covering it (for fast greedy init)
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_blocks = {s: [] for s in all_subsets}
    for b, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(b)

    # 2. Greedy Set Cover for Initialization
    uncovered = set(all_subsets)
    blocks = []
    used_blocks = set()
    
    while uncovered:
        # Pick a random uncovered subset to focus on (randomized greedy)
        s = random.choice(list(uncovered))
        candidates = subset_to_blocks[s]
        
        best_block = None
        best_gain = -1
        
        # Sample a few candidates to speed up? No, C(21,4)=5985 is small enough.
        # But iterating all 5985 every step * 76 steps is ~450k checks. Fast.
        for b in candidates:
            if b in used_blocks:
                continue
            gain = 0
            for sub in block_to_subsets[b]:
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        used_blocks.add(best_block)
        
        for sub in block_to_subsets[best_block]:
            uncovered.discard(sub)
            
    # 3. Simulated Annealing
    # State: list of blocks
    # Coverage counts for fast updates
    coverage_counts = {}
    for b in blocks:
        for sub in block_to_subsets[b]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    # Helper to get number of uncovered subsets
    def get_uncovered_count(coverage_counts):
        count = 0
        for s in all_subsets:
            if coverage_counts.get(s, 0) == 0:
                count += 1
        return count

    current_uncovered = get_uncovered_count(coverage_counts)
    current_cost = len(blocks) + 10 * current_uncovered  # Lambda = 10
    
    best_blocks = [list(b) for b in blocks]
    best_cost = current_cost
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random block
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            b_to_remove = blocks[idx]
            
            # Calculate delta cost
            delta_uncovered = 0
            for sub in block_to_subsets[b_to_remove]:
                if coverage_counts[sub] == 1:
                    delta_uncovered += 1
            
            delta_cost = 1 + 10 * delta_uncovered
            
            if random.random() < math.exp(-delta_cost / T):
                # Apply move
                for sub in block_to_subsets[b_to_remove]:
                    coverage_counts[sub] -= 1
                del blocks[idx]
                current_cost += delta_cost
                current_uncovered += delta_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
                    
        elif move_type < 0.6:
            # Add a random block
            b_to_add = random.choice(all_blocks)
            if b_to_add in blocks:
                continue
            
            delta_uncovered = 0
            for sub in block_to_subsets[b_to_add]:
                if coverage_counts.get(sub, 0) == 0:
                    delta_uncovered += 1
            
            delta_cost = 1 - 10 * delta_uncovered
            
            if random.random() < math.exp(-delta_cost / T):
                # Apply move
                blocks.append(b_to_add)
                for sub in block_to_subsets[b_to_add]:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                current_cost += delta_cost
                current_uncovered -= delta_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
                    
        else:
            # Swap a point in a random block
            if not blocks:
                continue
            idx = random.randint(0, len(blocks) - 1)
            block = blocks[idx]
            x_idx = random.randint(0, k - 1)
            x = block[x_idx]
            
            # Choose a new point y not in block
            candidates = [y for y in range(v) if y not in block]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_block_list = [p for p in block if p != x]
            new_block_list.append(y)
            new_block_tuple = tuple(sorted(new_block_list))
            
            # Calculate delta
            # Subsets in old block containing x are lost
            # Subsets in new block containing y are gained
            
            # We need to check coverage changes for all subsets in old and new block
            # This is complex to do in O(1). We will simulate.
            
            # Save old state
            old_subs = block_to_subsets[block]
            new_subs = block_to_subsets[new_block_tuple]
            
            # Calculate delta uncovered
            delta_uncovered = 0
            # Subsets only in old block (lost)
            for sub in old_subs:
                if sub not in new_subs:
                    if coverage_counts[sub] == 1:
                        delta_uncovered += 1
            # Subsets only in new block (gained)
            for sub in new_subs:
                if sub not in old_subs:
                    if coverage_counts.get(sub, 0) == 0:
                        delta_uncovered -= 1
                        
            delta_cost = 10 * delta_uncovered # Size doesn't change
            
            if random.random() < math.exp(-delta_cost / T):
                # Apply move
                # Remove old
                for sub in old_subs:
                    coverage_counts[sub] -= 1
                # Add new
                for sub in new_subs:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                blocks[idx] = list(new_block_tuple)
                
                current_cost += delta_cost
                current_uncovered += delta_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [list(b) for b in blocks]
                    
        T *= cooling_rate
        
    # Final cleanup: remove any redundant blocks from best solution
    # Re-calculate coverage for best_blocks
    final_blocks = best_blocks
    coverage_counts = {}
    for b in final_blocks:
        for sub in block_to_subsets[tuple(b)]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    # Pass to remove redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks) - 1, -1, -1):
            b = tuple(final_blocks[i])
            redundant = True
            for sub in block_to_subsets[b]:
                if coverage_counts[sub] <= 1:
                    redundant = False
                    break
            if redundant:
                for sub in block_to_subsets[b]:
                    coverage_counts[sub] -= 1
                del final_blocks[i]
                changed = True
                break
                
    return final_blocks
