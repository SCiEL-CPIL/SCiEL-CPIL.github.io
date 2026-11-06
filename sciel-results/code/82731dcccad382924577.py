import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    best_set = []
    best_size = 0
    
    max_n = 1000
    
    def run_greedy(order):
        chosen = []
        counts = {}
        
        for c in order:
            valid = True
            
            # Check 2*c
            if counts.get(2 * c, 0) >= 2:
                valid = False
            
            if valid:
                # Check c + x for x in chosen
                for x in chosen:
                    s = c + x
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                # Update counts
                counts[2 * c] = counts.get(2 * c, 0) + 1
                chosen.append(c)
                for x in chosen[:-1]:
                    s = c + x
                    counts[s] = counts.get(s, 0) + 1
                
        return chosen, counts

    def local_search(initial_set, initial_counts, max_iterations=2000):
        S = set(initial_set)
        counts = dict(initial_counts)
        C = set(range(1, max_n + 1)) - S
        
        # Helper to check if x can be added
        def can_add(x):
            if counts.get(2 * x, 0) >= 2:
                return False
            for y in S:
                if counts.get(x + y, 0) >= 2:
                    return False
            return True

        def add(x):
            S.add(x)
            C.remove(x)
            counts[2 * x] = counts.get(2 * x, 0) + 1
            for y in S:
                if y != x:
                    counts[x + y] = counts.get(x + y, 0) + 1

        def remove(x):
            S.remove(x)
            C.add(x)
            counts[2 * x] -= 1
            if counts[2 * x] == 0:
                del counts[2 * x]
            for y in S:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]

        best_local = len(S)
        best_local_set = set(S)
        
        for _ in range(max_iterations):
            if time.time() - start_time > deadline:
                break
                
            if not C:
                break
            
            # Try to add a random element from C
            added = False
            # Pick a few random candidates to try
            candidates = random.sample(list(C), min(10, len(C)))
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x):
                    add(x)
                    added = True
                    if len(S) > best_local:
                        best_local = len(S)
                        best_local_set = set(S)
                    break
            
            if not added:
                # Stuck. Try to remove a random element from S to break deadlock
                if S:
                    y = random.choice(list(S))
                    remove(y)
                    
        return list(best_local_set)

    # 1. Deterministic Ascending (Replicates seed behavior, ensures baseline score >= 1.0)
    asc_order = list(range(1, max_n + 1))
    current_set, current_counts = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)
    
    # Apply local search to the baseline
    if time.time() - start_time < deadline:
        improved = local_search(best_set, current_counts)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved

    # 2. Random Restarts with Local Search
    while time.time() - start_time < deadline:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set, current_counts = run_greedy(order)
        
        # If greedy result is already better, update
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Try to improve with local search
        # Only run local search if we have time and potential to improve
        if time.time() - start_time < deadline - 2.0: # Reserve 2s for next iteration
             improved = local_search(current_set, current_counts)
             if len(improved) > best_size:
                 best_size = len(improved)
                 best_set = improved

    return best_set
