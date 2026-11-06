import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit, or 40s for 150s? Prompt says 150s kill, 120s finish. Let's use 110s.
    
    def is_valid_add(x, S, counts):
        # Check if adding x keeps all sum counts <= 2
        # Sums involving x: x + y for y in S, and 2*x
        # We need to check if count[s] + 1 <= 2 for all new sums s
        # i.e., count[s] < 2 for all new sums s
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def add_element(x, S, counts):
        S.add(x)
        for y in S:
            if y != x: # x+y
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1

    def remove_element(x, S, counts):
        S.discard(x)
        for y in S:
            if y != x:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    # 1. Greedy Initialization
    S = set()
    counts = {}
    for x in range(1, 1001):
        if is_valid_add(x, S, counts):
            add_element(x, S, counts)
    
    best_size = len(S)
    
    # 2. Local Search
    # We will try to improve the size.
    # Strategy: 
    # - Try to add any element not in S.
    # - If no add possible, try swaps (remove x, add y) to change configuration.
    # - If stuck for a while, restart with random perturbation.
    
    # To make it efficient, we can maintain a list of candidates.
    # But checking all 1000 elements for add is O(N * |S|) which is ~1000 * 500 = 500k ops. 
    # Doing this in a loop might be slow if we do it many times.
    # However, we only need to do it until we find one valid add.
    
    while time.time() < deadline:
        improved = False
        
        # Try to add
        # Randomize order to avoid bias
        candidates = [x for x in range(1, 1001) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(x, S, counts):
                add_element(x, S, counts)
                best_size = len(S)
                improved = True
                break
        
        if improved:
            continue
            
        # If no add possible, try swaps
        # Try to swap an element x in S with y not in S
        # This is more expensive. 
        # Let's pick a random x in S and try to swap with random y not in S.
        # If successful, accept.
        
        if S:
            x = random.choice(list(S))
            # Try a few random y's
            for _ in range(50): # Limit attempts per step to avoid hanging
                if not S: break
                y = random.randint(1, 1000)
                if y in S:
                    continue
                # Check if swap (remove x, add y) is valid
                # 1. Temporarily remove x
                # We need to check if y can be added after x is removed.
                # is_valid_add(y, S \ {x}, counts_updated)
                
                # To avoid modifying state if invalid, we can simulate.
                # But simulation is O(|S|).
                # Let's just do it:
                
                # Save state? No, just try.
                # Remove x
                remove_element(x, S, counts)
                
                # Check if y can be added
                if is_valid_add(y, S, counts):
                    add_element(y, S, counts)
                    # If size increased? No, size is same.
                    # But configuration changed.
                    # We accept this move to explore.
                    # We don't update best_size here as it's same.
                    # But we break the "no add" loop.
                    # We should continue searching for adds.
                    # So we set improved = True? No, improved means size increased.
                    # Let's just break the swap loop and go back to try adds.
                    break
                else:
                    # Revert
                    add_element(x, S, counts)
            
            # If we did a swap, we go back to the top to try adds again.
            # If we didn't do a swap (all failed), we might be stuck.
            # We can try more swaps or restart.
            
        # If we are stuck (no adds, no successful swaps in this iteration), 
        # we can try a more aggressive perturbation or restart.
        # For now, the loop just continues trying random swaps.
        
    return list(S)
