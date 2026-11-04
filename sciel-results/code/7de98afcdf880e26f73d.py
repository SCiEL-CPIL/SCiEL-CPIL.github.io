import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy + Deletion with randomization.
    """
    # Parameters for this specific task are fixed by the evaluator, 
    # but we handle general v, k, t for robustness.
    # v=19, k=7, t=4
    
    deadline = time.time() + 115  # Leave 5s margin
    
    # 1. Precomputation
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map points to indices for faster subset generation? 
    # itertools.combinations on small tuples is fine.
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each block, list of t-subset indices it covers
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_index[s])
        block_subsets.append(subs)
        
    # For each t-subset, list of block indices that cover it
    # This is needed for the efficient greedy update
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
            
    best_solution = None
    best_len = float('inf')
    
    # Run multiple iterations of Randomized Greedy + Deletion
    while time.time() < deadline:
        # --- Greedy Phase ---
        # block_scores[i] = number of uncovered subsets covered by block i
        block_scores = [0] * num_blocks
        # Initially, all subsets are uncovered
        # So score of block i is len(block_subsets[i])
        for i in range(num_blocks):
            block_scores[i] = len(block_subsets[i])
            
        uncovered_count = num_t_subsets
        # We need to know which subsets are uncovered to update scores correctly
        # But in the "efficient" update, we decrement scores of blocks covering a subset 
        # when that subset becomes covered.
        # We don't strictly need a boolean array for uncovered if we track scores, 
        # but we need to know *when* a subset becomes covered to trigger the decrement.
        # A subset becomes covered when the block covering it is selected.
        # Wait, the logic is:
        # 1. Pick block B with max score.
        # 2. Add B to solution.
        # 3. For each subset S in B:
        #    If S was uncovered:
        #       Mark S as covered.
        #       For each block B' covering S:
        #           block_scores[B'] -= 1
        #
        # We need a `is_uncovered` array to know if S was uncovered.
        is_uncovered = [True] * num_t_subsets
        
        solution_blocks = []
        
        while uncovered_count > 0:
            # Find max score block
            # Optimization: We can use a max-heap or just max(). 
            # max() is O(N). N=50k. 150 steps -> 7.5M ops. OK.
            # But we can be smarter.
            max_score = -1
            max_block = -1
            # To allow randomization, we can pick from top K
            # For now, standard greedy
            for i in range(num_blocks):
                if block_scores[i] > max_score:
                    max_score = block_scores[i]
                    max_block = i
            
            if max_score == 0:
                # Should not happen if uncovered_count > 0
                break
                
            solution_blocks.append(max_block)
            
            # Update scores
            for s_idx in block_subsets[max_block]:
                if is_uncovered[s_idx]:
                    is_uncovered[s_idx] = False
                    uncovered_count -= 1
                    # Decrement score of all blocks covering s_idx
                    for b_prime in subset_to_blocks[s_idx]:
                        block_scores[b_prime] -= 1
        
        # --- Deletion Phase ---
        # Convert solution_blocks to a set for faster membership? 
        # We need to check redundancy.
        # We maintain a count of how many times each subset is covered by the *current* solution.
        cover_count = [0] * num_t_subsets
        for b_idx in solution_blocks:
            for s_idx in block_subsets[b_idx]:
                cover_count[s_idx] += 1
        
        # Try to remove each block
        # Iterate in reverse order or random order
        # Random order is better for finding different local minima
        indices_to_check = list(range(len(solution_blocks)))
        random.shuffle(indices_to_check)
        
        for pos in indices_to_check:
            if pos >= len(solution_blocks):
                continue
            b_idx = solution_blocks[pos]
            
            # Check if removing b_idx leaves any subset uncovered
            # i.e., if for any s in b_idx, cover_count[s] == 1
            is_redundant = True
            for s_idx in block_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for s_idx in block_subsets[b_idx]:
                    cover_count[s_idx] -= 1
                # Mark as removed
                solution_blocks[pos] = -1
                
        # Clean up solution_blocks
        current_solution = [b for b in solution_blocks if b != -1]
        
        if len(current_solution) < best_len:
            best_len = len(current_solution)
            best_solution = current_solution
            # If we hit a very good score, we might want to continue to polish, 
            # but the loop continues.
            
        # Optional: Simple Local Search (1-for-1 exchange)
        # This is expensive. We might skip it if time is tight.
        # Given the efficiency of Greedy+Deletion, it's a strong baseline.
        
    # Convert indices back to blocks
    if best_solution is None:
        # Fallback: return empty or a trivial covering if something went wrong
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
