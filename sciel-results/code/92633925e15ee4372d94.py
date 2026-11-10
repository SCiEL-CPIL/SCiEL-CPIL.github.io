"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all t-subsets and map to indices for fast counting
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    combo_to_idx = {combo: i for i, combo in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    max_restarts = 30  # Try more restarts, limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(range(n_t_subsets))  # Set of indices of uncovered t-subsets
        blocks = []
        cover_counts = [0] * n_t_subsets
        
        # Greedy construction
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick an uncovered t-subset index
            s_idx = next(iter(uncovered))
            s = all_t_subsets[s_idx]
            
            # Generate extensions of s to size k
            available = [x for x in range(v) if x not in s]
            num_to_add = k - t
            
            best_gain = -1
            best_block_idx = -1
            candidates = []
            
            # Iterate through possible extensions
            for ext in itertools.combinations(available, num_to_add):
                block_tuple = tuple(sorted(s + ext))
                
                # Calculate gain: how many t-subsets in this block are uncovered?
                gain = 0
                for sub in itertools.combinations(block_tuple, t):
                    idx = combo_to_idx[sub]
                    if idx in uncovered:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = block_tuple
                    candidates = [block_tuple]
                elif gain == best_gain:
                    candidates.append(block_tuple)
            
            if best_block_idx != -1:
                # Randomly choose from best candidates
                chosen_block = random.choice(candidates)
                blocks.append(chosen_block)
                
                # Update uncovered and cover_counts
                for sub in itertools.combinations(chosen_block, t):
                    idx = combo_to_idx[sub]
                    if idx in uncovered:
                        uncovered.discard(idx)
                    cover_counts[idx] += 1
        
        if not uncovered:  # Valid covering found
            # Redundancy removal using cover_counts
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b = blocks[i]
                    # Check if any t-subset in b is covered only by b
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        idx = combo_to_idx[sub]
                        if cover_counts[idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove block i
                        for sub in itertools.combinations(b, t):
                            idx = combo_to_idx[sub]
                            cover_counts[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break  # Restart loop as indices shifted
        
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # If we found a very good solution (e.g., close to known lower bound), we could stop early
        # But for now, let's use the time for more restarts.
        
    if best_blocks is None:
        # Fallback: Return a simple greedy result if time ran out before any valid solution
        # This should not happen as greedy is fast.
        # Construct one quickly without redundancy check
        uncovered = set(range(n_t_subsets))
        blocks = []
        while uncovered:
            s_idx = next(iter(uncovered))
            s = all_t_subsets[s_idx]
            available = [x for x in range(v) if x not in s]
            num_to_add = k - t
            best_gain = -1
            best_block = None
            for ext in itertools.combinations(available, num_to_add):
                block_tuple = tuple(sorted(s + ext))
                gain = sum(1 for sub in itertools.combinations(block_tuple, t) if combo_to_idx[sub] in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tuple
            if best_block:
                blocks.append(best_block)
                for sub in itertools.combinations(best_block, t):
                    uncovered.discard(combo_to_idx[sub])
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
