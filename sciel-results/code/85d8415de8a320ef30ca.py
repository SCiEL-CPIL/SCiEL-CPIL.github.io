import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    
    # Map t-subset to index
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks and their coverage as bitsets (Python ints)
    # Python integers have arbitrary precision and support fast bitwise ops
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    block_bitsets = []
    for b in k_blocks:
        bits = 0
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            bits |= (1 << idx)
        block_bitsets.append(bits)
        
    # Helper to convert index list to actual blocks
    def to_blocks(indices):
        return [list(k_blocks[i]) for i in indices]
    
    def is_valid(solution_indices):
        """Check if solution covers all t-subsets."""
        if not solution_indices:
            return num_t_subsets == 0
        covered = 0
        for idx in solution_indices:
            covered |= block_bitsets[idx]
        # Check if all bits up to num_t_subsets are set
        # (1 << num_t_subsets) - 1 is a mask of num_t_subsets ones
        return (covered & ((1 << num_t_subsets) - 1)) == ((1 << num_t_subsets) - 1)
        
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << num_t_subsets) - 1
        solution = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            # To avoid scanning all 50k blocks every time if we can't beat best_gain quickly,
            # we could randomize order or use heuristics. 
            # For now, scan all. 50k * (popcount ops) is fast in Python big ints?
            # popcount is O(1) in C for machine words, but for big ints it's O(n_words).
            # num_t_subsets is ~3875 for C(19,4). That's ~61 64-bit words.
            # So popcount is very fast.
            
            # Optimization: Randomize iteration order to break ties and allow different restarts
            indices = list(range(num_k_blocks))
            random.shuffle(indices)
            
            for b_idx in indices:
                bits = block_bitsets[b_idx]
                # Gain is number of newly covered t-subsets
                # newly_covered = bits & uncovered
                # gain = popcount(newly_covered)
                # Python 3.8+ has int.bit_count()
                gain = (bits & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Early termination if we found a block that covers all uncovered?
                    # Unlikely, but if gain == uncovered.bit_count(), we are done with this block
                    # and might be done with the whole problem if uncovered becomes 0.
                    # But we still need to check other blocks? No, if this block covers ALL currently uncovered,
                    # adding it makes uncovered 0. We can stop immediately.
                    if best_gain == uncovered.bit_count():
                        break
            
            if best_block_idx == -1:
                # Should not happen if uncovered != 0 and we have all blocks
                break
                
            solution.append(best_block_idx)
            uncovered &= ~block_bitsets[best_block_idx]
            
            # Time check
            if time.time() > deadline:
                return None # Signal timeout/incomplete
        
        return solution

    def reduce_solution(solution):
        """Remove redundant blocks. Returns new list of indices."""
        if not solution:
            return []
        
        # Count coverage for each t-subset
        # Using a list of counts is faster than bitsets for checking "is count 1?"
        counts = [0] * num_t_subsets
        for b_idx in solution:
            bits = block_bitsets[b_idx]
            # Iterate set bits to increment counts
            # This is slower than bitset OR, but we need counts for removal logic.
            # Alternative: Just re-check validity after removal?
            # "Is block X redundant?" -> Remove X, check if still valid.
            # Validity check is O(|S| * popcount).
            # Removal check for each of |S| blocks is O(|S|^2 * popcount).
            # |S| ~ 150. 150^2 * 60 ops ~ 1.3M ops. Fast.
            
            # Let's use the re-verify method as it's simpler and less error-prone with big ints.
            pass 
            
        # Actually, let's use the counts method but optimize the increment.
        # We can precompute the list of t-subset indices for each block?
        # That uses memory but speeds up counting.
        # Let's stick to a hybrid: 
        # 1. Compute counts.
        # 2. Try to remove.
        
        # Precompute list of t-subset indices for each block for faster counting?
        # C(7,4) = 35. 50k blocks * 35 = 1.75M ints. ~14MB. Acceptable.
        block_t_indices = []
        for b in k_blocks:
            indices = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_t_indices.append(indices)
            
        # Now reduce
        counts = [0] * num_t_subsets
        for b_idx in solution:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
                
        new_solution = []
        for b_idx in solution:
            removable = True
            for t_idx in block_t_indices[b_idx]:
                if counts[t_idx] == 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_t_indices[b_idx]:
                    counts[t_idx] -= 1
            else:
                new_solution.append(b_idx)
                
        return new_solution

    def local_search_improve(solution, max_iters=100):
        """Try to swap blocks to enable further reductions."""
        if not solution:
            return solution
        
        # Precompute t-indices for speed in local search
        # (Already done in reduce_solution, but let's make it local or global scope accessible)
        # We'll just pass block_t_indices from outer scope if we define it there.
        # Let's define block_t_indices in construct() scope.
        
        current_size = len(solution)
        sol_set = set(solution)
        
        # Heuristic: Pick a block to remove, then find a replacement that covers its unique parts?
        # Or: Pick a random block to add, then reduce.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Strategy: Add a random block, then reduce.
            # If size decreases, keep.
            candidate = random.randint(0, num_k_blocks - 1)
            if candidate in sol_set:
                continue
                
            new_sol = solution + [candidate]
            reduced = reduce_solution(new_sol)
            
            if len(reduced) < current_size:
                solution = reduced
                sol_set = set(solution)
                current_size = len(solution)
            # If size is same, we might still keep it if it allows future improvements?
            # Usually, we only keep if strictly better to avoid drift.
            
        return solution

    # Precompute block_t_indices once
    block_t_indices = []
    for b in k_blocks:
        indices = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_t_indices.append(indices)
        
    # Re-define reduce_solution to use precomputed indices
    def reduce_solution_v2(solution):
        if not solution:
            return []
        counts = [0] * num_t_subsets
        for b_idx in solution:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
                
        new_solution = []
        for b_idx in solution:
            removable = True
            for t_idx in block_t_indices[b_idx]:
                if counts[t_idx] == 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_t_indices[b_idx]:
                    counts[t_idx] -= 1
            else:
                new_solution.append(b_idx)
        return new_solution

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Greedy is fast. Reduce is fast. Improve is moderate.
    
    for seed in range(200):
        if time.time() > deadline - 10: # Stop 10s early to ensure return
            break
            
        sol = greedy_construct(seed)
        if sol is None:
            continue # Timed out during greedy, skip
            
        # Ensure validity (should be true by construction of greedy)
        if not is_valid(sol):
            continue
            
        # Reduce
        sol = reduce_solution_v2(sol)
        
        # Improve
        if time.time() < deadline - 5:
            sol = local_search_improve(sol, max_iters=20)
            # Final reduce to ensure no redundancy introduced by swaps
            sol = reduce_solution_v2(sol)
            
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
        # Early stop if we hit a very good score
        if best_size <= 140:
            break

    if best_solution is None:
        # Fallback: Return a valid solution.
        # If we couldn't finish even one greedy, something is wrong or machine is very slow.
        # Return all blocks? No, too many.
        # Return a single block? Invalid.
        # We must return a valid covering.
        # The safest valid covering is the set of ALL blocks.
        # But that's 50k blocks.
        # Let's try one last desperate greedy without shuffle overhead?
        # Or just return the first valid one we found in a previous iteration?
        # If best_solution is None, it means NO valid solution was found.
        
        # Emergency: Construct a trivial valid covering.
        # Pick blocks that cover disjoint sets?
        # Actually, just return all k-blocks. It's valid. Score will be bad, but valid.
        return [list(b) for b in k_blocks]
        
    return to_blocks(best_solution)
