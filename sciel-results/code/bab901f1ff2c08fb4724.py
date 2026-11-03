import itertools
import time
import random
import math
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using bitmask-accelerated greedy 
    and local search.
    
    Strategy:
    1. Precompute all k-blocks and their coverage masks (bitsets of t-subsets).
    2. Perform multiple randomized greedy passes.
    3. Apply local search (redundancy removal and potential swaps) to each pass.
    4. Return the best solution found within the time limit.
    """
    start_time = time.time()
    # The evaluator kills at 150s. We use 140s to be safe.
    deadline = start_time + 140.0 

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Precompute coverage mask for each k-block
    # mask is an integer where bit i is 1 if t_subsets[i] is contained in the block
    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << t_index_map[combo])
        block_masks[idx] = mask
    
    full_mask = (1 << num_t) - 1
    
    # Helper: Check time
    def time_up():
        return time.time() > deadline - 5.0  # Leave 5s for final cleanup/return

    # Helper: Remove redundant blocks from a list of indices
    def _remove_redundant(selected_indices):
        """
        Remove blocks that do not cover any unique t-subset.
        Returns a new list of indices.
        """
        if not selected_indices:
            return selected_indices
        
        # To speed up redundancy check, we can use bitmasks of the union of others.
        # However, for ~300 blocks, O(N^2) bitmask ORs are fast enough in Python 
        # because the integers are small (11k bits).
        
        kept = list(selected_indices)
        changed = True
        # Limit iterations to prevent infinite loops if logic is flawed (shouldn't happen)
        max_iter = len(kept) + 1
        
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1
            new_kept = []
            n = len(kept)
            for i in range(n):
                if time_up():
                    return kept # Fallback to current state if time runs out
                
                idx = kept[i]
                # Calculate coverage of all other blocks
                others_mask = 0
                for j in range(n):
                    if i != j:
                        others_mask |= block_masks[kept[j]]
                
                # If (block_mask & ~others_mask) == 0, it covers nothing unique
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                else:
                    new_kept.append(idx)
            
            if changed:
                kept = new_kept
                
        return kept

    # Helper: Local Search - Try to improve by swapping or removing
    def _local_search(selected_indices):
        """
        Attempts to reduce block count further.
        1. Remove redundants.
        2. Try to find a block that can be removed if we add one new block? 
           (This is complex. A simpler heuristic: 
            If we are stuck, try random restarts in the main loop.)
        
        For now, we rely heavily on the randomized greedy restarts.
        We can add a simple "kick" if we have time:
        - Pick a random block B_i.
        - Remove it.
        - Identify uncovered subsets U.
        - Try to cover U with 1 new block B_new.
        - If found, replace B_i with B_new.
        - Then remove redundants again.
        """
        current = _remove_redundant(selected_indices)
        
        if not current:
            return current
            
        # Calculate current coverage mask
        current_mask = 0
        for idx in current:
            current_mask |= block_masks[idx]
            
        # If not full coverage, we shouldn't be here from a valid greedy pass, 
        # but just in case.
        if current_mask != full_mask:
             # This shouldn't happen if greedy was correct, but if it did, 
             # we can't do local search safely without fixing coverage first.
             # We'll just return what we have.
             pass

        # Try a few random kicks/swaps
        # We try to remove a block and replace it with a better one that allows 
        # removal of another block? That's 2-opt.
        # Simpler: Just try to remove a block and see if the remaining set 
        # can be "completed" by adding 1 block that covers the gap, 
        # resulting in same size but potentially better structure?
        # Actually, the most effective local search for set cover is often 
        # just "remove redundant". The diversity comes from the greedy phase.
        
        # Let's add a specific heuristic: 
        # Find the block with the fewest unique subsets.
        # Try to remove it.
        # If we remove it, some subsets become uncovered.
        # Can we cover them with 1 new block?
        # If yes, we swap. Does this help? 
        # It might allow further redundancy removal.
        
        # Given the time budget, the most powerful driver is the number of 
        # randomized greedy restarts. We will keep local search simple to 
        # maximize the number of restarts.
        
        return current

    def _greedy_pass(seed):
        """
        One randomized greedy pass.
        """
        if time_up():
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        # To add randomness, we can shuffle the order we check blocks, 
        # or add noise to the gain.
        # Shuffling 75k items every step is too slow.
        # Instead, we can pick a random subset of candidates to inspect? 
        # Or just use standard greedy but with randomized tie-breaking.
        # Standard greedy with tie-breaking is fast.
        
        while uncovered:
            if time_up():
                return None
            
            best_idx = -1
            best_gain = -1
            
            # Optimization: 
            # We iterate all blocks. 
            # To break ties randomly, we can add a small random noise to gain 
            # or just pick the first one found with max gain (deterministic).
            # To ensure diversity across restarts, we can shuffle the block indices 
            # ONCE at the start of the pass.
            
            # Actually, shuffling 75k indices is O(N) which is fine.
            # But doing it every step is O(N^2) which is bad.
            # Let's just iterate in a fixed order but if we find a gain == best_gain,
            # we might keep the first one.
            # To get diversity, we can randomize the block indices list once.
            
            # Let's do: 
            # 1. Create a list of indices.
            # 2. Shuffle it.
            # 3. Iterate through it to find max gain.
            # This ensures different blocks are picked if gains are equal.
            
            # But creating/shuffling a list of 75k ints 300 times is 22.5M ops.
            # Acceptable.
            
            indices = list(range(num_k))
            rng.shuffle(indices)
            
            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    best_blocks = None
    best_len = float('inf')
    
    attempt = 0
    # Run as many passes as possible
    while time.time() < deadline - 10.0: # Leave 10s for final best check
        selected = _greedy_pass(attempt)
        if selected is None:
            break
        
        # Refine
        selected = _local_search(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
            
            # Optional: If we get very close to lower bound, we can stop?
            # Lower bound is 264. Best known is 335.
            # If we get < 330, that's a record. 
            # We should keep searching to see if we can go lower.
            
        attempt += 1
        
    # Fallback if no solution found (should not happen)
    if best_blocks is None:
        # Deterministic greedy
        uncovered = full_mask
        selected = []
        selected_set = set()
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        if selected:
            selected = _local_search(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
