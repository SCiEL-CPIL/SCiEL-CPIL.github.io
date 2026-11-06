import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # counts[s] stores the number of representations of s as a+b with a<=b from S
    # Max sum is 2000
    counts = [0] * (2 * N + 1)
    S = []
    in_set = [False] * (N + 1)
    
    # Helper to add x to S
    def add_to_set(x):
        S.append(x)
        in_set[x] = True
        # Update counts
        for y in S:
            if y >= x:
                counts[x + y] += 1
            else:
                counts[x + y] += 1 # Wait, a<=b. If y < x, then y is the smaller one.
                # The pair is (y, x). Sum is x+y.
                # We just need to increment counts[x+y] for all y in S.
                # But we must be careful not to double count if we iterate all y.
                # Actually, if we iterate all y in S, we count (x,y) and (y,x) if we're not careful?
                # No, the set of pairs is {(a,b) | a,b in S, a<=b}.
                # If we add x, the new pairs are (x, y) for all y in S_old such that x <= y, and (y, x) for all y in S_old such that y < x.
                # In both cases, the sum is x+y.
                # So we just increment counts[x+y] for ALL y in S_old.
                # And counts[2*x] for the pair (x,x).
                pass
        # Let's re-implement add/remove correctly.
        
    # Correct logic for update:
    # When adding x:
    #   For each y in S:
    #       counts[x+y] += 1
    #   counts[2*x] += 1
    #   S.append(x)
    #   in_set[x] = True
    
    # When removing x:
    #   For each y in S (excluding x):
    #       counts[x+y] -= 1
    #   counts[2*x] -= 1
    #   S.remove(x)
    #   in_set[x] = False

    def try_add(x):
        # Check if x can be added
        if in_set[x]:
            return False
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if counts[x + y] >= 2:
                return False
        return True

    def do_add(x):
        # Assume try_add(x) was True
        for y in S:
            counts[x + y] += 1
        counts[2 * x] += 1
        S.append(x)
        in_set[x] = True

    def do_remove(x):
        # Remove x from S and update counts
        for y in S:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        S.remove(x)
        in_set[x] = False

    # 1. Greedy Initialization
    for x in range(1, N + 1):
        if try_add(x):
            do_add(x)
            
    best_size = len(S)
    
    # 2. Local Search
    # We will try to swap elements to find a larger set.
    # Since the greedy set is maximal, we can't just add.
    # We must swap to change the structure.
    
    # To make swapping faster, we can pre-compute or just iterate.
    # Randomized swaps.
    
    attempts = 0
    while time.time() < deadline:
        # Try to add any element if possible (in case previous swaps opened space)
        added_any = False
        # Randomly shuffle candidates to avoid bias
        candidates = [i for i in range(1, N + 1) if not in_set[i]]
        random.shuffle(candidates)
        
        for z in candidates:
            if try_add(z):
                do_add(z)
                best_size = max(best_size, len(S))
                added_any = True
                break # Restart candidate check after adding one? Or continue?
                # Continuing is fine, but adding one changes the state significantly.
                # Let's just break and restart the outer loop logic to be safe and simple.
        
        if added_any:
            continue
            
        # If no addition possible, try a swap
        if len(S) == 0:
            break
            
        # Pick a random element to remove
        x_idx = random.randint(0, len(S) - 1)
        x = S[x_idx]
        
        # Pick a random element to add
        # To be efficient, we might want to pick y that is "compatible" with most of S \ {x}
        # But random is a good start.
        
        # Remove x
        do_remove(x)
        
        # Try to find a y that can be added
        found_swap = False
        # Check a subset of candidates or all?
        # Checking all is O(N*|S|). With |S|~100, N=1000, that's 10^5 ops per swap attempt.
        # 10^4 attempts is 10^9 ops. Too slow in Python.
        
        # Optimization: Only check a few random y's?
        # Or, check y's that are "close" to x?
        # Let's try checking a random sample of non-set elements.
        
        sample_size = min(50, N - len(S))
        if sample_size > 0:
            non_set_elements = [i for i in range(1, N + 1) if not in_set[i]]
            random.shuffle(non_set_elements)
            candidates_y = non_set_elements[:sample_size]
            
            for y in candidates_y:
                if try_add(y):
                    do_add(y)
                    found_swap = True
                    break
        
        if not found_swap:
            # Revert: Re-add x
            do_add(x)
            
        attempts += 1
        if attempts % 1000 == 0:
            # Periodic check
            pass

    return S
