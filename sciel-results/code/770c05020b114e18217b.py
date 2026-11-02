import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    def greedy_cover():
        """Fast greedy to get a valid initial solution."""
        uncovered = set(range(num_t_subsets))
        selected = set()
        while uncovered:
            # Pick the block that covers the most uncovered subsets
            # Optimization: Pick a random uncovered subset, then find the best block covering it
            s_id = random.choice(list(uncovered))
            candidates = t_to_blocks[s_id]
            best_candidate = -1
            max_covered = -1
            for b_idx in candidates:
                if b_idx in selected:
                    continue
                covered = sum(1 for tid in block_t_ids[b_idx] if tid in uncovered)
                if covered > max_covered:
                    max_covered = covered
                    best_candidate = b_idx
            
            if best_candidate != -1:
                selected.add(best_candidate)
                for tid in block_t_ids[best_candidate]:
                    if tid in uncovered:
                        uncovered.discard(tid)
            else:
                # Should not happen
                break
        return selected

    # Initial Solution
    current_solution = greedy_cover()
    best_solution = set(current_solution)
    best_size = len(current_solution)
    
    # Precompute sets for faster lookup? 
    # block_t_ids is a list of lists. Converting to sets for membership testing is faster.
    block_t_ids_sets = [set(ids) for ids in block_t_ids]

    def get_cover_counts(solution):
        counts = [0] * num_t_subsets
        for b_idx in solution:
            for tid in block_t_ids[b_idx]:
                counts[tid] += 1
        return counts

    current_cover_counts = get_cover_counts(current_solution)

    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.95
    min_T = 0.1
    max_iterations = 200000 # Large number, controlled by time
    
    iteration = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Try to perform a move
        # Move Type 1: Remove a block (if redundant)
        # Move Type 2: 2-for-1 Swap
        # Move Type 3: 1-for-1 Swap (for diversification)
        
        move_type = random.choice([1, 2, 3])
        
        new_solution = None
        new_cover_counts = None
        delta_cost = 0
        
        if move_type == 1:
            # Try to remove a random block
            if not current_solution:
                continue
            b_idx = random.choice(list(current_solution))
            # Check if removing b_idx leaves any subset uncovered
            # A subset is uncovered if its count becomes 0.
            # We can check if any tid in b_idx has count 1.
            is_redundant = True
            for tid in block_t_ids[b_idx]:
                if current_cover_counts[tid] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                new_solution = set(current_solution)
                new_solution.discard(b_idx)
                new_cover_counts = current_cover_counts[:]
                for tid in block_t_ids[b_idx]:
                    new_cover_counts[tid] -= 1
                delta_cost = -1 # Improvement
            else:
                # If not redundant, we can't just remove it.
                # We could try to repair, but that's complex. 
                # For SA, we usually only accept valid states or use a penalty.
                # Let's stick to valid states only for simplicity and correctness.
                continue

        elif move_type == 2:
            # 2-for-1 Swap
            if len(current_solution) < 2:
                continue
            # Pick two distinct blocks
            b1, b2 = random.sample(list(current_solution), 2)
            
            # Find subsets covered ONLY by b1 or b2
            # A subset is "critical" for the pair if it is covered by b1 or b2 
            # and not by any other block in the solution.
            # Note: If a subset is covered by b1 AND b2, and no other block, it's critical.
            # If it's covered by b1 and b3, it's not critical for the removal of b1,b2.
            
            critical_subsets = []
            for tid in block_t_ids[b1]:
                if current_cover_counts[tid] == 1:
                    critical_subsets.append(tid)
            for tid in block_t_ids[b2]:
                if current_cover_counts[tid] == 1:
                    critical_subsets.append(tid)
            
            # Remove duplicates
            critical_subsets = set(critical_subsets)
            
            if not critical_subsets:
                # This case is handled by Move 1 (redundancy)
                continue
                
            # Find a block that covers all critical_subsets
            # Heuristic: Pick the first critical subset, look at blocks covering it
            # Check if any of those blocks covers ALL critical subsets.
            target_tid = next(iter(critical_subsets))
            candidates = t_to_blocks[target_tid]
            
            replacement_found = False
            best_replacement = -1
            
            for new_b_idx in candidates:
                if new_b_idx == b1 or new_b_idx == b2:
                    continue
                if new_b_idx in current_solution:
                    continue
                
                # Check if new_b_idx covers all critical subsets
                new_b_set = block_t_ids_sets[new_b_idx]
                if critical_subsets.issubset(new_b_set):
                    best_replacement = new_b_idx
                    replacement_found = True
                    break
            
            if replacement_found:
                new_solution = set(current_solution)
                new_solution.discard(b1)
                new_solution.discard(b2)
                new_solution.add(best_replacement)
                
                # Update counts
                new_cover_counts = current_cover_counts[:]
                for tid in block_t_ids[b1]:
                    new_cover_counts[tid] -= 1
                for tid in block_t_ids[b2]:
                    new_cover_counts[tid] -= 1
                for tid in block_t_ids[best_replacement]:
                    new_cover_counts[tid] += 1
                    
                delta_cost = -1 # Improvement (2 blocks -> 1 block)

        elif move_type == 3:
            # 1-for-1 Swap (for diversification, might not improve immediately but helps SA)
            if not current_solution:
                continue
            b_idx = random.choice(list(current_solution))
            
            # Find critical subsets for b_idx
            critical_subsets = []
            for tid in block_t_ids[b_idx]:
                if current_cover_counts[tid] == 1:
                    critical_subsets.append(tid)
            
            if not critical_subsets:
                continue # Redundant, should be handled by Move 1
                
            target_tid = next(iter(critical_subsets))
            candidates = t_to_blocks[target_tid]
            
            replacement_found = False
            best_replacement = -1
            
            for new_b_idx in candidates:
                if new_b_idx == b_idx:
                    continue
                if new_b_idx in current_solution:
                    continue
                
                new_b_set = block_t_ids_sets[new_b_idx]
                if set(critical_subsets).issubset(new_b_set):
                    best_replacement = new_b_idx
                    replacement_found = True
                    break
            
            if replacement_found:
                new_solution = set(current_solution)
                new_solution.discard(b_idx)
                new_solution.add(best_replacement)
                
                new_cover_counts = current_cover_counts[:]
                for tid in block_t_ids[b_idx]:
                    new_cover_counts[tid] -= 1
                for tid in block_t_ids[best_replacement]:
                    new_cover_counts[tid] += 1
                    
                delta_cost = 0 # Neutral move

        # Acceptance Criteria
        if new_solution is not None:
            if delta_cost < 0:
                # Improvement, always accept
                current_solution = new_solution
                current_cover_counts = new_cover_counts
                if len(current_solution) < best_size:
                    best_size = len(current_solution)
                    best_solution = set(current_solution)
            else:
                # Neutral or worse (though we only generate neutral or better here)
                # Accept with probability exp(-delta/T)
                # Since delta is 0, prob is 1.
                # If we allowed worse moves, we'd use the formula.
                # For now, we only generate valid states.
                current_solution = new_solution
                current_cover_counts = new_cover_counts
                
        T *= cooling_rate
        iteration += 1

    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
