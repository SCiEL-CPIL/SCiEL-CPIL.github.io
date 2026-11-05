"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _check_valid_fast(seq):
    """Check if all consecutive block sums in seq are distinct.
    Returns True if valid, False otherwise.
    """
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _dfs_backtrack(seq, candidates, best_seq, time_limit, rng):
    """
    Depth-first search with backtracking to maximize sequence length.
    seq: current sequence (sorted)
    candidates: list of remaining values to try (shuffled)
    best_seq: reference to list holding the best sequence found so far
    time_limit: deadline timestamp
    rng: random generator
    """
    # Pruning: if current length + remaining candidates < best length, stop
    # This is a weak bound, but helps a bit.
    
    if time.time() > time_limit:
        return

    if len(seq) > len(best_seq):
        best_seq[:] = seq[:]

    if not candidates:
        return

    # Try each candidate
    # We use a copy of candidates to modify? No, we pass indices or slices.
    # To be efficient, we iterate and remove from a working list.
    
    # Optimization: If we can't fit any more, return.
    # Simple heuristic: try smaller numbers first? Or random?
    # Let's try the candidates in the order provided (which will be shuffled).
    
    # We need to try inserting each candidate into the correct sorted position?
    # No, the problem states "strictly increasing sequence".
    # If we maintain `seq` as sorted, we just need to check if `x` can be added.
    # But `x` must be placed in the correct sorted order.
    # Since we iterate `candidates` which are just a pool of values,
    # we should pick a value `x`, find its insertion point, and check validity.
    
    # However, standard DFS on "next element" usually implies we are building
    # the sequence in increasing order of values? No, the sequence is increasing,
    # but we can pick values in any order as long as the final sorted list is valid.
    # Actually, if we just pick a set of numbers, the "consecutive block" property
    # depends on their sorted order.
    
    # So, the state is a SET of numbers.
    # But DFS usually builds a list.
    # Let's stick to: We have a set of chosen numbers. We try to add a new number.
    # We don't care about the order of addition, only the final sorted set.
    
    # Revised DFS:
    # State: `chosen` (set/list of chosen numbers), `available` (set of remaining numbers)
    # Action: Pick x from available, add to chosen, check if sorted(chosen) is valid.
    # If valid, recurse.
    # If invalid, backtrack.
    
    # To make this fast, we check validity on the sorted list.
    # O(N^2) check.
    
    # Let's implement this specific DFS.
    
    # To avoid O(N) insertion sort every time, we can keep `chosen` sorted.
    # But adding a number to a sorted list is O(N).
    # Checking validity is O(N^2).
    # Total per node: O(N^2).
    
    # We'll use a list for `chosen` to keep it sorted.
    # And a set for `available` for O(1) removal.
    
    # This function is a bit complex to pass around state efficiently in Python recursion.
    # Let's use a class or non-local variables.
    
    # For simplicity in the edit block, I will implement a standalone recursive function
    # that takes the current sorted list and a set of available numbers.
    
    # Note: Python recursion limit might be an issue if depth > 100, but max length is likely < 50.
    
    # We will shuffle `available` to randomize the search path.
    
    pass # Placeholder, actual logic below

def construct():
    best_seq = []
    time_limit = time.time() + 55.0 # Leave 5s for safety
    
    # Multiple restarts with different random seeds
    for seed in range(1000):
        if time.time() > time_limit:
            break
            
        rng = random.Random(seed)
        
        # Start with an empty sequence
        # Available numbers 1..256
        available = set(range(1, 257))
        
        # Shuffle available to randomize order of consideration
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        # We use a stack-based or recursive DFS.
        # Given the branching factor is high (up to 256), deep DFS might be slow.
        # We can limit the depth of backtracking or use a heuristic.
        
        # Let's try a simpler heuristic search:
        # 1. Greedy fill with random order.
        # 2. If stuck, try to "swap" elements.
        
        # Actually, let's stick to the Backtracking idea but limit the "lookahead".
        # A full backtracking over 256 items is impossible.
        # We need a heuristic to prune.
        
        # Heuristic: Only consider the top K candidates at each step?
        # Or, just do a deep greedy with random restarts (which the seed already does).
        
        # The seed does 100 random greedy runs.
        # A truly new approach: **Local Search (Hill Climbing) on the Set**.
        
        # Let's implement Local Search.
        
        # Initialize with a random valid sequence (using greedy)
        current_seq = []
        sums = set()
        # Random Greedy Init
        init_candidates = list(range(1, 257))
        rng.shuffle(init_candidates)
        for x in init_candidates:
            # Check if adding x is valid
            # We need to insert x into current_seq (sorted)
            # Find insertion point
            left = 0
            right = len(current_seq)
            while left < right:
                mid = (left + right) // 2
                if current_seq[mid] < x:
                    left = mid + 1
                else:
                    right = mid
            
            # Insert x at `left`
            # Check validity of new sequence
            # To speed up, we can just check the full sequence.
            # N is small.
            new_seq = current_seq[:left] + [x] + current_seq[left:]
            
            # Check validity
            valid = True
            n = len(new_seq)
            new_sums = set()
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += new_seq[j]
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            
            if valid:
                current_seq = new_seq
                sums = new_sums # Not strictly needed if we re-check, but good for tracking
        
        best_seq_local = current_seq[:]
        
        # Local Search Loop
        # Try to improve by swapping or adding/removing
        no_improve_count = 0
        max_no_improve = 500
        
        while no_improve_count < max_no_improve and time.time() < time_limit:
            improved = False
            # Randomly pick a move
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Try to add a random element not in current_seq
                if len(current_seq) < 256:
                    # Pick random candidate
                    candidates_to_try = [x for x in range(1, 257) if x not in current_seq]
                    if candidates_to_try:
                        x = rng.choice(candidates_to_try)
                        # Find insertion point
                        left = 0
                        right = len(current_seq)
                        while left < right:
                            mid = (left + right) // 2
                            if current_seq[mid] < x:
                                left = mid + 1
                            else:
                                right = mid
                        
                        new_seq = current_seq[:left] + [x] + current_seq[left:]
                        
                        # Check validity
                        valid = True
                        n = len(new_seq)
                        new_sums = set()
                        for i in range(n):
                            s = 0
                            for j in range(i, n):
                                s += new_seq[j]
                                if s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                            if not valid:
                                break
                        
                        if valid:
                            current_seq = new_seq
                            improved = True
                            no_improve_count = 0
                            if len(current_seq) > len(best_seq_local):
                                best_seq_local = current_seq[:]
                                if len(current_seq) > len(best_seq):
                                    best_seq = current_seq[:]
            
            elif move_type == 'remove':
                # Remove a random element to allow future additions
                if len(current_seq) > 0:
                    idx = rng.randint(0, len(current_seq) - 1)
                    new_seq = current_seq[:idx] + current_seq[idx+1:]
                    # Removing an element always maintains validity?
                    # Yes, subset of a valid sequence is valid.
                    # But we want to maximize length, so removing is bad unless it enables a larger addition.
                    # We accept removal only if we can immediately add two elements?
                    # Or just accept it to explore, but penalize in objective?
                    # Since objective is length, we only accept if it leads to improvement eventually.
                    # In hill climbing, we usually only accept if objective improves.
                    # So we should NOT accept a pure removal unless we combine it with an addition.
                    # Let's implement "swap" as the primary improvement move.
                    pass # Skip pure removal in this simple hill climber
            
            elif move_type == 'swap':
                # Swap an element in current_seq with an element not in current_seq
                if len(current_seq) > 0:
                    # Pick element to remove
                    idx = rng.randint(0, len(current_seq) - 1)
                    old_val = current_seq[idx]
                    
                    # Pick element to add
                    candidates_to_try = [x for x in range(1, 257) if x not in current_seq]
                    if candidates_to_try:
                        new_val = rng.choice(candidates_to_try)
                        
                        # Construct new sequence
                        # Remove old_val, insert new_val
                        # Since current_seq is sorted, removing old_val keeps it sorted.
                        # Then insert new_val at correct position.
                        
                        # Remove old_val
                        temp_seq = current_seq[:idx] + current_seq[idx+1:]
                        
                        # Insert new_val
                        left = 0
                        right = len(temp_seq)
                        while left < right:
                            mid = (left + right) // 2
                            if temp_seq[mid] < new_val:
                                left = mid + 1
                            else:
                                right = mid
                        
                        new_seq = temp_seq[:left] + [new_val] + temp_seq[left:]
                        
                        # Check validity
                        valid = True
                        n = len(new_seq)
                        new_sums = set()
                        for i in range(n):
                            s = 0
                            for j in range(i, n):
                                s += new_seq[j]
                                if s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                            if not valid:
                                break
                        
                        if valid:
                            current_seq = new_seq
                            improved = True
                            no_improve_count = 0
                            if len(current_seq) > len(best_seq_local):
                                best_seq_local = current_seq[:]
                                if len(current_seq) > len(best_seq):
                                    best_seq = current_seq[:]
            
            if not improved:
                no_improve_count += 1
        
        # End Local Search for this seed
    
    return best_seq
# EVOLVE-BLOCK-END
