import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # Greedy Construction with Weighted Randomization
    def greedy_construct(randomize_ties=True, bias_factor=1.0):
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 10.0:
                return selected_blocks
            
            # Pick a random uncovered t-subset
            if uncovered_count < 500:
                # Build list for efficiency when few remain
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                # Random probing
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            best_tie_breaker = -1.0
            
            # Calculate gains
            # Optimization: If we have many candidates, we can sample or use a smarter heuristic.
            # For now, scan all candidates (avg 364) and compute gain.
            # Gain = number of newly covered t-subsets.
            
            # To speed up, we can pre-compute something? No, uncovered state changes.
            # Let's just do it.
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain > best_gain:
                            # Early exit for this block's gain calculation?
                            # We still need to compare with other blocks.
                            # But if gain reaches 56 (max possible), we can break inner loop.
                            if gain == 56:
                                break
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if randomize_ties:
                        # Reset tie breaker
                        best_tie_breaker = random.random()
                elif gain == best_gain and randomize_ties:
                    # Tie breaking: random choice among equal gains
                    # To avoid always picking the first one, we can pick randomly.
                    # A simple way: if we keep the first one, it's biased towards lower indices.
                    # Let's randomly replace with probability 0.5 or similar?
                    # Or just collect ties? That's expensive.
                    # Let's just randomly decide to keep the new one.
                    if random.random() < 0.5:
                        best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Local Search: Aggressive Reduction
    def local_search(block_indices):
        # Convert to list of unique block indices
        block_indices = list(set(block_indices))
        
        # Step 1: Iterative Removal of Redundant Blocks
        # A block is redundant if every t-subset it covers is covered by at least one other block.
        # i.e., for all j in block_tsubs[b], cover_count[j] > 1.
        
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 5.0:
                break
                
            # Count coverage
            cover_count = [0] * num_tsubs
            for b_idx in block_indices:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_block_indices = []
            for b_idx in block_indices:
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    changed = True
            
            block_indices = new_block_indices
            
        # Step 2: Block Replacement (Swap)
        # Try to replace a block B1 with B2 such that coverage is maintained.
        # This doesn't reduce count, but might allow further removals?
        # Actually, if we just want to reduce count, removal is key.
        # But sometimes a swap allows a different structure that permits removal.
        # Given time, let's focus on removal first.
        # We can also try "2-for-1" exchanges? Too complex.
        
        # Let's try a few rounds of "try to remove a random block, if it fails, maybe try to swap it?"
        # Simpler: Just return the reduced list.
        
        return block_indices

    # Run initial greedy
    initial_blocks = greedy_construct(randomize_ties=True)
    optimized_blocks = local_search(initial_blocks)
    
    best_len = len(optimized_blocks)
    best_block_list = [all_blocks[i] for i in optimized_blocks]
    
    # Random Restarts
    # We have ~140s left.
    # Each greedy+local_search takes ~1-2s?
    # Let's estimate:
    # Greedy: ~335 steps. Each step scans ~364 candidates * 56 tsubs = 20k ops.
    # 335 * 20k = 6.7M ops. In Python, ~0.5-1s.
    # Local Search: Removal loops. Might take a bit.
    # So ~2s per restart.
    # We can do ~70 restarts.
    
    for restart in range(100):
        if get_elapsed() > deadline - 10.0:
            break
            
        # Vary the random seed or just rely on internal randomness
        # We can also vary the bias or tie-breaking strategy
        
        current_blocks = greedy_construct(randomize_ties=True)
        if not current_blocks:
            continue
            
        optimized = local_search(current_blocks)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_block_list = [all_blocks[i] for i in optimized]
            
            # Early exit if we hit a very good number
            if best_len <= 325:
                break

    # Final validation check (optional but good for safety)
    # We trust the construction, but let's ensure no duplicates in final list just in case.
    # best_block_list is built from indices, so they are distinct if indices are distinct.
    # local_search uses set(), so indices are distinct.
    
    return [list(b) for b in best_block_list]
