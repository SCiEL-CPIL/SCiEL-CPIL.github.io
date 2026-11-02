import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy: Randomized Greedy Set Cover with Local Search Reduction.
    """
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # This helps in gain calculation, but for greedy set cover on t-subsets,
    # we usually work with the incidence of blocks.
    # However, constructing all possible blocks C(v,k) is too large for v=19, k=7.
    # C(19,7) = 50388. This is manageable.
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute the set of t-subsets it covers
    # We represent coverage as a bitset or a list of indices for speed
    # Since n_tsubs = C(19,4) = 3876, we can use a Python set of indices or a bitarray.
    # Let's use a list of indices for each block.
    
    block_covered_tsubs = []
    for block in all_blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsub_to_idx[ts])
        block_covered_tsubs.append(covered)
    
    # Inverse index: for each t-subset index, which blocks cover it?
    # tsub_to_blocks[i] = list of block indices covering tsub i
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, covered in enumerate(block_covered_tsubs):
        for t_idx in covered:
            tsub_to_blocks[t_idx].append(b_idx)
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple randomized greedy restarts
    # Greedy Set Cover:
    # 1. Start with empty set of blocks, all tsubs uncovered.
    # 2. While uncovered:
    #    a. Find block that covers max uncovered tsubs (with random tie-breaking).
    #    b. Add block, mark covered tsubs.
    # 3. Local search: try to remove blocks that don't compromise coverage.
    
    num_restarts = 20 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 10:
            break
            
        random.seed(restart * 12345 + 67890)
        
        uncovered = set(range(n_tsubs))
        current_blocks = []
        
        while uncovered:
            # Find best block
            max_covered = -1
            best_b_idx = -1
            candidates = []
            
            # Optimization: Only consider blocks that cover at least one uncovered tsub
            # We can iterate through uncovered tsubs and collect candidate blocks
            # But that might be slow if uncovered is large.
            # Better: Iterate through all blocks? 50k blocks.
            # Or maintain a count of uncovered tsubs per block?
            
            # Let's use a simpler approach for the first few steps, then optimize if needed.
            # Actually, 50k blocks * 3876 tsubs is ~200M ops per step if naive.
            # Too slow.
            
            # Efficient approach:
            # Maintain a list of uncovered tsubs.
            # For each block, we need the intersection size with uncovered.
            # We can precompute nothing more.
            
            # Let's try a different greedy:
            # Pick a random uncovered tsub.
            # Among blocks covering it, pick the one covering the most other uncovered tsubs.
            
            if not uncovered:
                break
                
            target_tsub_idx = random.choice(list(uncovered))
            candidate_blocks = tsub_to_blocks[target_tsub_idx]
            
            if not candidate_blocks:
                # Should not happen
                break
                
            best_gain = -1
            best_b_idx = -1
            best_rand = random.random()
            
            for b_idx in candidate_blocks:
                # Count how many uncovered tsubs this block covers
                gain = 0
                for t_idx in block_covered_tsubs[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat current best? No, we don't know max possible.
                            pass
                
                # Tie-breaking with randomness
                rand_val = random.random()
                if gain > best_gain or (gain == best_gain and rand_val > best_rand):
                    best_gain = gain
                    best_b_idx = b_idx
                    best_rand = rand_val
            
            if best_b_idx == -1:
                break
                
            current_blocks.append(best_b_idx)
            
            # Update uncovered
            for t_idx in block_covered_tsubs[best_b_idx]:
                if t_idx in uncovered:
                    uncovered.discard(t_idx)
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks.copy()
            
        # Local Search: Try to remove redundant blocks
        # A block is redundant if all tsubs it covers are covered by other blocks.
        # We can check this by:
        # 1. Remove block.
        # 2. Check if any tsub becomes uncovered.
        
        # This is expensive if done naively.
        # Let's try a simple pass:
        # For each block, check if it's redundant.
        # We need to know for each tsub, which blocks cover it.
        # We have tsub_to_blocks.
        
        # To check if block b is redundant:
        # For each tsub t in block_covered_tsubs[b]:
        #   If len(tsub_to_blocks[t] excluding b) == 0, then b is NOT redundant.
        #   Else, b might be redundant.
        
        # This requires knowing the current set of blocks.
        # Let's maintain a count of how many blocks cover each tsub.
        
        tsub_count = [0] * n_tsubs
        for b_idx in current_blocks:
            for t_idx in block_covered_tsubs[b_idx]:
                tsub_count[t_idx] += 1
        
        # Try to remove blocks
        improved = True
        while improved:
            improved = False
            for i in range(len(current_blocks)):
                if time.time() - start_time > timeout - 5:
                    break
                
                b_idx = current_blocks[i]
                # Check if removing b_idx leaves any tsub uncovered
                is_redundant = True
                for t_idx in block_covered_tsubs[b_idx]:
                    if tsub_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove b_idx
                    current_blocks.pop(i)
                    for t_idx in block_covered_tsubs[b_idx]:
                        tsub_count[t_idx] -= 1
                    improved = True
                    break # Restart loop as indices changed
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks.copy()
            
    # Convert best_blocks (indices) back to actual blocks
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        # This shouldn't happen
        return []
        
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Sort blocks for consistency? Not required, but good for determinism if seed fixed.
    # The task asks for a list of blocks.
    
    return final_blocks
