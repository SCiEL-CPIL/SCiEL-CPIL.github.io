import random
import time

def construct():
    n = 128
    # Start with odds
    S = set(range(1, n + 1, 2))
    
    def is_valid(S):
        S_list = sorted(S)
        K = len(S_list)
        S_set = S
        # For each pair a, b in S with a < b
        for i in range(K):
            a = S_list[i]
            for j in range(i + 1, K):
                b = S_list[j]
                s_ab = a + b
                if s_ab not in S_set:
                    continue
                # Check for c
                # c must be in S, c != a, c != b
                # and a+c in S, b+c in S
                # Iterate c
                for k in range(K):
                    c = S_list[k]
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    def count_violations(S):
        # Returns number of violating triples
        S_list = sorted(S)
        K = len(S_list)
        S_set = S
        count = 0
        for i in range(K):
            a = S_list[i]
            for j in range(i + 1, K):
                b = S_list[j]
                s_ab = a + b
                if s_ab not in S_set:
                    continue
                for k in range(K):
                    c = S_list[k]
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        count += 1
        return count

    # Simulated Annealing
    # Start with valid set
    current_S = set(S)
    best_S = set(current_S)
    best_size = len(current_S)
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Initial temperature
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    # We want to maximize size.
    # Score = size - penalty * violations
    # But we want to stay in valid space mostly?
    # Or allow invalid states?
    # If we allow invalid states, we need a penalty.
    # Let's try to stay in valid space first (Greedy/Local Search).
    # If stuck, try swaps.
    
    # Let's do a simple Local Search:
    # 1. Try to add every missing element.
    # 2. If adding creates violation, skip.
    # 3. If no more additions possible, try swaps:
    #    Remove x, Add y.
    #    If size increases (impossible, swap keeps size same) or if it allows future additions?
    #    Actually, we want to maximize size.
    #    So:
    #    Phase 1: Greedy Add
    #    Phase 2: Greedy Swap (to enable more adds)
    
    # Let's implement a robust Local Search
    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            # 1. Try adding
            for x in range(1, n + 1):
                if x not in S:
                    S_new = S | {x}
                    if is_valid(S_new):
                        S = S_new
                        improved = True
                        # Restart adding loop as adding one might enable another
                        break 
            # 2. Try swapping if no additions
            if not improved:
                # Try to remove one element and add another to break deadlocks
                # This is expensive.
                # Let's just try random restarts of the greedy phase
                pass
        return S

    best_S = set(range(1, n + 1, 2))
    
    # Run multiple random restarts of Local Search
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize the order of addition or start with slightly different sets
        # Start with Odds + some random evens
        S_curr = set(range(1, n + 1, 2))
        
        # Randomly add some evens first to break symmetry
        evens = [x for x in range(2, n + 1, 2)]
        random.shuffle(evens)
        # Try adding up to 5 random evens
        for e in evens[:5]:
            S_test = S_curr | {e}
            if is_valid(S_test):
                S_curr = S_test
        
        # Now run local search (greedy add)
        S_curr = local_search(S_curr)
        
        if len(S_curr) > len(best_S):
            best_S = S_curr
            if len(best_S) == n: # Max possible? Unlikely
                break

    return sorted(best_S)
