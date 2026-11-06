import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    best_seq = []
    
    # State for backtracking
    # a: list of chosen values
    # S: list of prefix sums
    # diffs: set of all block sums (differences of prefix sums)
    
    def backtrack(a, S, diffs):
        nonlocal best_seq, start_time
        
        # Time check
        if time.time() - start_time > 55:
            return

        # Update best
        if len(a) > len(best_seq):
            best_seq = a.copy()
        
        # Pruning: Can we beat the best length?
        if not a:
            max_possible = 256 # Very loose
        else:
            last_val = a[-1]
            # Next values must be > last_val, strictly increasing.
            # Min next: last_val + 1
            # Min next+1: last_val + 2
            # ...
            # Min next+k: last_val + k + 1
            # We need last_val + k + 1 <= 256 => k <= 255 - last_val
            # So max additional elements is 255 - last_val
            max_additional = 255 - last_val
            
        if len(a) + max_additional <= len(best_seq):
            return

        # Try next candidates
        # Start from last_val + 1 to 256
        start_candidate = (a[-1] + 1) if a else 1
        
        for x in range(start_candidate, 257):
            # Check if adding x is valid
            new_S = S[-1] + x if S else x
            # New block sums are new_S - S_i for all S_i in S
            # Plus x itself if S is empty? 
            # If a is empty, S is [0]. new_S = x. Sum = x - 0 = x.
            # So we always have S containing 0 at index 0.
            
            valid = True
            for s_i in S:
                diff = new_S - s_i
                if diff in diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Forward check (optional but good): 
            # Does this choice preclude any immediate next step?
            # We can just let the recursion handle it.
            
            # Prepare new state
            new_a = a + [x]
            new_S = S + [new_S]
            new_diffs = set(diffs)
            for s_i in S:
                new_diffs.add(new_S - s_i)
                
            backtrack(new_a, new_S, new_diffs)
            
            # If we found a solution that uses the entire range or is very long, 
            # we might break early if we are sure we can't do better? 
            # Not easily, so we continue to explore other branches.

    # Initial state
    # a = []
    # S = [0]
    # diffs = set()
    
    backtrack([], [0], set())
    
    return best_seq

