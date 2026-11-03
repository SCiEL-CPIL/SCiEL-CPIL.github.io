import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Generate all candidate blocks and their masks
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                return None # Timeout
            
            best_gain = -1
            candidates = []
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                return selected_indices # Should not happen if valid input
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    def refine(selected_indices):
        if selected_indices is None:
            return []
            
        solution_set = set(selected_indices)
        
        # Calculate coverage counts
        coverage_count = [0] * num_triples
        for idx in solution_set:
            block = blocks[idx]
            for triple in itertools.combinations(block, t):
                coverage_count[triple_to_idx[triple]] += 1
                
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate over a copy of the list to allow modification
            for idx in list(solution_set):
                block = blocks[idx]
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    t_idx = triple_to_idx[triple]
                    if coverage_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    solution_set.remove(idx)
                    for triple in itertools.combinations(block, t):
                        coverage_count[triple_to_idx[triple]] -= 1
                    changed = True
                    
        return [blocks[i] for i in solution_set]

    def local_search(initial_solution, time_budget):
        """Simulated annealing to minimize block count from a valid covering."""
        if initial_solution is None:
            return None
            
        # Work with indices for speed
        solution = list(initial_solution)
        if not solution:
            return solution
            
        # Build coverage counts
        coverage_count = [0] * num_triples
        for idx in solution:
            for triple in itertools.combinations(blocks[idx], t):
                coverage_count[triple_to_idx[triple]] += 1
        
        best_solution = list(solution)
        best_len = len(solution)
        
        # Temperature for simulated annealing
        temp = 10.0
        cooling = 0.995
        min_temp = 0.1
        
        rng = random.Random(42)
        
        while time.time() < time_budget and temp > min_temp:
            # Try random removals
            for _ in range(50):
                if time.time() > time_budget:
                    break
                    
                if len(solution) <= 1:
                    break
                    
                # Pick a random block to try removing
                rem_idx = rng.randrange(len(solution))
                rem_block_idx = solution[rem_idx]
                
                # Check if removing this block leaves any triple uncovered
                can_remove = True
                for triple in itertools.combinations(blocks[rem_block_idx], t):
                    t_idx = triple_to_idx[triple]
                    if coverage_count[t_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    solution.pop(rem_idx)
                    for triple in itertools.combinations(blocks[rem_block_idx], t):
                        coverage_count[triple_to_idx[triple]] -= 1
                    
                    if len(solution) < best_len:
                        best_len = len(solution)
                        best_solution = list(solution)
                else:
                    # Try to repair: find a block that covers the uncovered triples
                    # First identify which triples would become uncovered
                    uncovered_triples = []
                    for triple in itertools.combinations(blocks[rem_block_idx], t):
                        t_idx = triple_to_idx[triple]
                        if coverage_count[t_idx] <= 1:
                            uncovered_triples.append(t_idx)
                    
                    if not uncovered_triples:
                        continue
                    
                    # Find blocks that cover all uncovered triples
                    # Use bit masks for efficiency
                    needed_mask = 0
                    for t_idx in uncovered_triples:
                        needed_mask |= (1 << t_idx)
                    
                    # Find candidate blocks that cover needed_mask
                    candidates = []
                    for i in range(num_blocks):
                        if i in solution:
                            continue
                        if (block_masks[i] & needed_mask) == needed_mask:
                            candidates.append(i)
                    
                    if candidates:
                        # Pick a random candidate
                        new_block_idx = rng.choice(candidates)
                        
                        # Remove the old block
                        solution.pop(rem_idx)
                        for triple in itertools.combinations(blocks[rem_block_idx], t):
                            coverage_count[triple_to_idx[triple]] -= 1
                        
                        # Add the new block
                        solution.append(new_block_idx)
                        for triple in itertools.combinations(blocks[new_block_idx], t):
                            coverage_count[triple_to_idx[triple]] += 1
            
            # Try direct swaps to potentially enable future removals
            for _ in range(20):
                if time.time() > time_budget:
                    break
                    
                if len(solution) <= 1:
                    break
                    
                # Pick a random block to swap
                swap_idx = rng.randrange(len(solution))
                old_block_idx = solution[swap_idx]
                
                # Find a random other block
                other_block_idx = rng.randrange(num_blocks)
                if other_block_idx == old_block_idx or other_block_idx in solution:
                    continue
                
                # Check if the swap maintains validity
                would_uncover = []
                for triple in itertools.combinations(blocks[old_block_idx], t):
                    t_idx = triple_to_idx[triple]
                    if coverage_count[t_idx] <= 1:
                        would_uncover.append(t_idx)
                
                if not would_uncover:
                    continue
                
                # Check if new block covers all would_uncover triples
                new_mask = block_masks[other_block_idx]
                valid_swap = True
                for t_idx in would_uncover:
                    if not (new_mask & (1 << t_idx)):
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Perform the swap
                    solution[swap_idx] = other_block_idx
                    for triple in itertools.combinations(blocks[old_block_idx], t):
                        coverage_count[triple_to_idx[triple]] -= 1
                    for triple in itertools.combinations(blocks[other_block_idx], t):
                        coverage_count[triple_to_idx[triple]] += 1
            
            temp *= cooling
        
        # Final refinement
        final = refine(best_solution)
        return final

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts with local search
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s buffer
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        # First do basic refinement
        refined = refine(greedy_result)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined
        
        # Then try local search with remaining time
        remaining = deadline - 25 - time.time()
        if remaining > 5:
            ls_result = local_search(refined, time.time() + remaining)
            if ls_result is not None and len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result
    
    if best_solution is None:
        # Fallback: try one more greedy
        greedy_result = run_greedy(0)
        if greedy_result is None:
            greedy_result = run_greedy(1)
        if greedy_result is not None:
            best_solution = refine(greedy_result)
    
    if best_solution is None:
        # Absolute fallback
        greedy_result = run_greedy(0)
        best_solution = refine(greedy_result)
    
    # Sort for consistency
    best_solution.sort()
    return [list(b) for b in best_solution]
