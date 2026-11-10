import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    N = 256
    
    # Precompute conflicts for fast can_add
    # conflicts[x] is a list of tuples (a, b) such that {x, a, b} forms a 3-AP
    # We store the pair (a, b) to check if both are in S
    conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # Case 1: x is the middle term. (x-d, x, x+d)
        for d in range(1, N + 1):
            a = x - d
            b = x + d
            if a >= 1 and b <= N:
                conflicts[x].append((a, b))
            else:
                # Since d is increasing, if b > N, further d will also be > N
                # But a will become < 1. 
                # Actually, if b > N, we can break? No, a might still be valid for other d?
                # No, if b = x+d > N, then for d' > d, b' > N.
                # So we can break.
                break
        
        # Case 2: x is the first term. (x, x+d, x+2d)
        for d in range(1, N + 1):
            a = x + d
            b = x + 2 * d
            if b <= N:
                conflicts[x].append((a, b))
            else:
                break
                
        # Case 3: x is the last term. (x-2d, x-d, x)
        for d in range(1, N + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1:
                conflicts[x].append((a, b))
            else:
                break

    def can_add(x, S):
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    def get_score(S):
        # Use float for speed during search
        return sum(1.0 / x for x in S)

    # 1. Initial Greedy Construction
    S_greedy = set()
    for x in range(1, N + 1):
        if can_add(x, S_greedy):
            S_greedy.add(x)
    
    best_set = set(S_greedy)
    best_score = get_score(best_set)
    
    # 2. Local Search (Hill Climbing with Greedy Refill)
    def local_search(S_init):
        S = set(S_init)
        current_score = get_score(S)
        improved = True
        
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            # Try removing elements, prioritizing those with low weight (large x)
            # Sorting S is O(N log N), N is small (< 256)
            sorted_S = sorted(S, reverse=True)
            
            for x in sorted_S:
                if (time.time() - start_time) > deadline - 5:
                    return S, current_score
                
                # Backup state
                S_backup = set(S)
                
                # Remove x
                S.remove(x)
                
                # Greedy Refill: Add smallest possible y
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    if can_add(y, S):
                        S.add(y)
                
                new_score = get_score(S)
                
                if new_score > current_score + 1e-12:
                    current_score = new_score
                    improved = True
                    # Restart the loop with the new S to catch cascading improvements
                    break
                else:
                    # Restore backup
                    S = S_backup
        
        return S, current_score

    best_set, best_score = local_search(best_set)
    
    # 3. Randomized Perturbation and Restart
    # If time remains, try to escape local optima by perturbing the best set
    while (time.time() - start_time) < deadline - 10:
        # Create a perturbed version of the best set
        S_perturbed = set(best_set)
        
        # Remove a small number of random elements
        # The number of removals can vary
        num_removals = random.randint(1, 3)
        if len(S_perturbed) > num_removals:
            elements_to_remove = random.sample(list(S_perturbed), num_removals)
            for x in elements_to_remove:
                S_perturbed.remove(x)
        else:
            # If set is small, just clear it and start fresh? 
            # Or just skip this iteration
            continue
            
        # Greedy Refill
        for y in range(1, N + 1):
            if y in S_perturbed:
                continue
            if can_add(y, S_perturbed):
                S_perturbed.add(y)
                
        score_perturbed = get_score(S_perturbed)
        
        if score_perturbed > best_score + 1e-12:
            best_set = S_perturbed
            best_score = score_perturbed
            # Run local search on the improved set
            best_set, best_score = local_search(best_set)
            
        # Occasionally try a fresh greedy with a random skip
        if random.random() < 0.01:
            S_fresh = set()
            # Randomly skip a few small numbers to explore different regions
            # This is a heuristic
            pass

    return list(best_set)
