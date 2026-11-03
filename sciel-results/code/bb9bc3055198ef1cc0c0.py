import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the list of t-subset indices it covers
    # Using numpy arrays for fast operations
    block_to_covered = []
    for block in all_blocks:
        covered = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_covered.append(covered)
        
    # Convert to numpy array for potential vectorization if needed, 
    # but lists of lists are fine for set operations in SA if we use sets for current coverage.
    
    # Build reverse index: for each t-subset, which blocks cover it?
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)

    def greedy_cover(seed=0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        indices = []
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            
            # Pick a random uncovered t-subset to guide the search
            start_idx = rng.choice(list(uncovered))
            
            # Find all blocks that cover this t-subset
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            indices.append(best_block)
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        
        return indices

    def get_initial_solution():
        """Get a valid initial solution using greedy."""
        indices = greedy_cover(0)
        # Ensure it's valid
        covered = set()
        for idx in indices:
            covered.update(block_to_covered[idx])
        if len(covered) != n_t:
            # Fallback: brute force add missing
            missing = set(range(n_t)) - covered
            while missing:
                if time.time() > deadline - 5.0:
                    break
                idx = random.choice(list(missing))
                candidates = list(t_subset_to_blocks[idx])
                best_b = candidates[0]
                for b in candidates:
                    if len(block_to_covered[b] & missing) > len(block_to_covered[best_b] & missing):
                        best_b = b
                indices.append(best_b)
                missing -= set(block_to_covered[best_b])
        return indices

    def simulated_annealing(initial_indices, time_budget=120.0):
        """
        SA to minimize the number of blocks.
        Moves:
        1. Remove a block if coverage remains valid.
        2. Swap a block with another block if it improves coverage overlap or allows removal?
           Actually, simple removal is the primary driver for reduction.
           To escape local minima where no single removal is valid, we can use:
           - Swap: Replace block A with block B. This keeps size constant but might enable a future removal.
           - Or: Accept a temporary increase in size? No, we want to minimize.
           
           Standard approach for set cover reduction:
           - Try to remove any block. If valid, accept.
           - If no removal is valid, try to swap a block with a "better" one (one that covers more unique elements or overlaps less with others) to create a state where a removal becomes possible.
           
           Let's implement:
           1. Try random removals.
           2. If stuck, try random swaps.
           
           Cost function: Number of blocks.
           Validity: All t-subsets covered.
           
           Since we start valid, we only accept moves that keep us valid OR (in SA) allow temporary invalidity?
           Allowing invalidity is risky for the final answer. 
           Better: Only accept valid moves. This is essentially a hill-climbing / local search.
           To escape local minima, we can use "2-opt" like moves: 
           - Remove block A, Add block B. Size same. 
           - This might not help reduce size directly, but changes the structure.
           
           However, the most effective move for size reduction is removal.
           If no removal is possible, we are in a local minimum w.r.t removals.
           We can try to swap a block to make another block redundant.
           
           Let's try:
           1. Pick a random block A.
           2. Try to remove it. If valid, accept (size - 1).
           3. If invalid, pick a random block B (not A).
           4. Try to swap A with B (remove A, add B). 
              - Check if new state is valid.
              - Accept if valid. (Size unchanged, but structure changed).
              - This allows us to move around the space of valid coverings of the same size, hoping to find a configuration where a removal is possible.
        """
        current_indices = list(initial_indices)
        current_size = len(current_indices)
        
        # Maintain coverage counts for fast validity check
        # cover_count[ts_idx] = number of blocks in current_indices covering ts_idx
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in current_indices:
            for ts_idx in block_to_covered[idx]:
                cover_count[ts_idx] += 1
        
        # Verify initial state
        if np.any(cover_count == 0):
            return initial_indices # Should not happen if initial is valid
            
        best_indices = list(current_indices)
        best_size = current_size
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        rng = random.Random(42)
        
        start_sa = time.time()
        
        while time.time() < deadline - 10.0:
            if T < T_min:
                break
                
            # 1. Try Removal
            if current_size > 1:
                idx_to_remove = rng.choice(current_indices)
                
                # Check if removal is valid
                # All ts_idx covered by idx_to_remove must have count > 1
                is_valid = True
                for ts_idx in block_to_covered[idx_to_remove]:
                    if cover_count[ts_idx] <= 1:
                        is_valid = False
                        break
                
                if is_valid:
                    # Accept removal
                    for ts_idx in block_to_covered[idx_to_remove]:
                        cover_count[ts_idx] -= 1
                    current_indices.remove(idx_to_remove)
                    current_size -= 1
                    best_indices = list(current_indices)
                    best_size = current_size
                    T = max(T_min, T * cooling)
                    continue
            else:
                # Cannot remove if size is 1? C(24,7,3) needs at least 69 blocks, so this won't happen.
                pass

            # 2. Try Swap (to escape local minima)
            # Pick a block to swap out
            idx_out = rng.choice(current_indices)
            
            # Pick a block to swap in. 
            # Heuristic: Pick a block that covers many elements currently covered by only 1 block?
            # Or just random from a small candidate set?
            # Random from all blocks is too slow (346k).
            # Sample a few random blocks.
            
            swap_improved = False
            for _ in range(10): # Try 10 random swaps per iteration
                if time.time() > deadline - 10.0:
                    break
                
                idx_in = rng.randint(0, n_blocks - 1)
                if idx_in == idx_out:
                    continue
                if idx_in in current_indices:
                    continue
                
                # Check validity of swap:
                # Remove idx_out, Add idx_in.
                # 1. Elements covered by idx_out must still be covered by others OR by idx_in.
                # 2. Elements covered by idx_in must be valid (they always are, just adding coverage).
                
                # Efficient check:
                # For each ts_idx in block_to_covered[idx_out]:
                #   if cover_count[ts_idx] == 1 and ts_idx not in block_to_covered[idx_in]:
                #       invalid
                #   else:
                #       valid for this ts_idx
                
                valid_swap = True
                for ts_idx in block_to_covered[idx_out]:
                    if cover_count[ts_idx] == 1:
                        # This element is only covered by idx_out.
                        # It must be covered by idx_in.
                        if ts_idx not in block_to_covered[idx_in]:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Accept swap
                    # Update cover counts
                    for ts_idx in block_to_covered[idx_out]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_covered[idx_in]:
                        cover_count[ts_idx] += 1
                    
                    # Update indices list
                    current_indices.remove(idx_out)
                    current_indices.append(idx_in)
                    
                    swap_improved = True
                    break # Restart search with new state
            
            if not swap_improved:
                # No valid swap found in this iteration
                # Cool down
                T *= cooling
            
            # Check time
            if time.time() > deadline - 10.0:
                break

        return best_indices

    # 1. Get initial solution
    initial_indices = get_initial_solution()
    
    if not initial_indices:
        return []
        
    # 2. Run SA
    sa_indices = simulated_annealing(initial_indices)
    
    # 3. Final Validation
    covered = set()
    for idx in sa_indices:
        covered.update(block_to_covered[idx])
        
    if len(covered) != n_t:
        # Fallback to initial
        return [all_blocks[i] for i in initial_indices]
        
    return [all_blocks[i] for i in sa_indices]

if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Found covering with {len(blocks)} blocks")
