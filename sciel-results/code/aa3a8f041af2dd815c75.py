import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # State
    S = set()
    counts = {}
    
    # Helper to add x to S and counts
    def add(x):
        S.add(x)
        # Update counts for sums involving x
        # Pairs (x, y) for y in S, y != x, and (x, x)
        # Since S is a set, we iterate over current S (before adding x? No, after adding x, x is in S).
        # It's safer to update based on the set BEFORE adding x, plus x itself.
        # Let's do it manually to avoid double counting or logic errors.
        pass # We will implement add/remove carefully in the main logic or helpers.

    # Let's use lists for S for faster iteration if needed, but set is fine for N=1000.
    # We'll maintain S as a set for O(1) lookup, and maybe a list for iteration if we need to shuffle.
    
    def is_valid_add(x):
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x+y for y in S
        for y in S:
            if counts.get(x+y, 0) >= 2:
                return False
        return True

    def do_add(x):
        S.add(x)
        # Update counts
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in S:
            if y != x:
                s = x + y
                counts[s] = counts.get(s, 0) + 1

    def do_remove(x):
        S.remove(x)
        # Update counts
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in S:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    # 1. Greedy Initialization
    # Standard greedy: add 1, 2, 3... if valid
    for x in range(1, 1001):
        if is_valid_add(x):
            do_add(x)
    
    best_size = len(S)
    
    # 2. Local Search
    # We will try to improve the set.
    # Strategy: 
    # - Try to add a random number not in S.
    # - If fail, try to swap an element in S with a number not in S.
    
    all_numbers = set(range(1, 1001))
    
    # To speed up "random not in S", we can maintain a list of candidates or just sample.
    # Sampling: pick random r in 1..1000, if r not in S, use it.
    
    no_improve_count = 0
    max_no_improve = 500 # Reset strategy if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a different random seed or perturb current state
            # Simple perturbation: remove 5 random elements and re-add greedily/randomly
            for _ in range(5):
                if S:
                    x = random.choice(list(S))
                    do_remove(x)
            no_improve_count = 0
            max_no_improve += 100 # Increase patience
            
        # Phase 1: Try Insertion
        # Try a few random insertions
        inserted = False
        for _ in range(20):
            z = random.randint(1, 1000)
            if z not in S and is_valid_add(z):
                do_add(z)
                best_size = max(best_size, len(S))
                no_improve_count = 0
                inserted = True
                break
        
        if inserted:
            continue
            
        # Phase 2: Try Swaps
        # Pick a random x in S and a random z not in S
        # To be smarter, pick x that is "restrictive"
        # Heuristic: x is restrictive if it participates in many sums with count 2.
        
        swapped = False
        for _ in range(50):
            if not S:
                break
            x = random.choice(list(S))
            z = random.randint(1, 1000)
            
            if z in S:
                continue
                
            # Check if swapping x for z is valid
            # This is equivalent to: remove x, then check if z is valid.
            # We can simulate this without actually modifying state, or modify and rollback.
            # Modify and rollback is safer for correctness if logic is complex, but here it's simple.
            
            # Simulate remove x
            # Counts decrease by 1 for x+y (y in S, y!=x) and 2x.
            # Then check if z can be added to (S \ {x}).
            
            # Let's just do it:
            do_remove(x)
            
            if is_valid_add(z):
                do_add(z)
                swapped = True
                no_improve_count = 0
                # Note: Size remains same. But structure changed.
                break
            else:
                # Rollback
                do_add(x)
                
        if not swapped and not inserted:
            no_improve_count += 1

    return list(S)

# EVOLVE-BLOCK-END
