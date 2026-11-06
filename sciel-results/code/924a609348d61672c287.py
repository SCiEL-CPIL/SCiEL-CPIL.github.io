import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Prescribed elements
    initial = [1, 2, 5, 11]
    best_size = len(initial)
    best_set = initial[:]
    
    # Pre-calculate sums for initial set
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
    
    best_sums = get_sums(best_set)
    
    # Candidates are 12 to 284
    candidates = list(range(12, 285))
    
    def is_compatible(x, current_set, current_sums):
        # Check if x can be added
        # 1. 2x not in sums
        if 2 * x in current_sums:
            return False
        # 2. x + y not in sums for any y in current_set
        for y in current_set:
            if x + y in current_sums:
                return False
        return True
        
    def greedy_fill(current_set, current_sums, candidates_pool):
        """
        Greedily add elements from a shuffled pool until no more can be added.
        Returns the final set and sums.
        """
        s = current_set[:]
        sums = set(current_sums)
        # Shuffle candidates to introduce randomness
        pool = candidates_pool[:]
        random.shuffle(pool)
        
        for x in pool:
            if is_compatible(x, s, sums):
                s.append(x)
                # Update sums
                for y in s[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
                
        return s, sums

    # Local search / Hill Climbing
    def local_search(start_set, start_sums):
        s = start_set[:]
        sums = set(start_sums)
        current_size = len(s)
        
        # First, try to fill greedily
        s, sums = greedy_fill(s, sums, candidates)
        current_size = len(s)
        
        if current_size >= best_size:
            return s, sums, current_size
            
        # Try swaps and replacements
        # This part is complex to implement efficiently in a short time.
        # A simpler approach: Randomized restarts with greedy fill are often sufficient for N=284.
        return s, sums, current_size

    # Strategy: Randomized Restart with Greedy Fill
    # We can also add a "kick" phase where we remove a random element and try to refill.
    
    while time.time() < deadline:
        # Start with initial set
        s = initial[:]
        sums = get_sums(s)
        
        # Randomize the order of candidates for greedy fill
        # To increase diversity, we can also randomly select a subset of candidates to prioritize
        # Or just shuffle the whole list
        
        # Add a small amount of "noise" by sometimes skipping a valid candidate to allow a better one later?
        # No, greedy is usually fine for the first pass.
        
        s, sums = greedy_fill(s, sums, candidates)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
            
            # If we reached 20, we can return early as it's the theoretical max mentioned
            if best_size >= 20:
                return best_set
        
        # If we didn't reach 20, try a local improvement step
        # Simple kick: remove a random element from s (not in initial) and refill
        if best_size < 20 and len(s) > 4:
            # Pick a random element to remove (excluding the mandatory ones)
            removable_indices = [i for i, x in enumerate(s) if x not in initial]
            if removable_indices:
                idx = random.choice(removable_indices)
                removed_val = s.pop(idx)
                # Recalculate sums? Or just subtract?
                # Recalculating is safer and fast enough for k~15
                sums = get_sums(s)
                
                # Refill
                s, sums = greedy_fill(s, sums, candidates)
                
                if len(s) > best_size:
                    best_size = len(s)
                    best_set = s
                    best_sums = sums
                    if best_size >= 20:
                        return best_set

    return best_set
