import math
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    moduli = list(range(20, 201))
    # Sort by weight descending (smallest m first)
    sorted_moduli = sorted(moduli, key=lambda m: 1.0/m, reverse=True)
    n_moduli = len(sorted_moduli)
    
    # Precompute gcds
    gcd_cache = {}
    for i, m1 in enumerate(sorted_moduli):
        for m2 in sorted_moduli[i:]:
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
            if m1 != m2:
                gcd_cache[(m2, m1)] = math.gcd(m1, m2)

    def get_score(sol):
        return sum(1.0 / m for m, _ in sol)

    def is_compatible(m, r, sol):
        for m2, r2 in sol:
            g = gcd_cache.get((m, m2), math.gcd(m, m2))
            if (r - r2) % g == 0:
                return False
        return True

    # 1. Greedy Initialization
    # Try to fill all moduli greedily
    greedy_sol = []
    for m in sorted_moduli:
        for r in range(m):
            if is_compatible(m, r, greedy_sol):
                greedy_sol.append((m, r))
                break
    
    best_sol = list(greedy_sol)
    best_score = get_score(best_sol)
    
    # 2. Branch and Bound on the first K moduli
    # We optimize the high-weight moduli, and complete the rest greedily.
    K = 30  # Number of moduli to branch on
    
    start_time = time.time()
    time_limit = 115.0
    
    # Global best
    global_best_score = best_score
    global_best_sol = best_sol
    
    def greedy_complete(selected_so_far, start_idx):
        """
        Complete the solution greedily from start_idx onwards,
        given the fixed selected_so_far.
        Returns the total score and the solution.
        """
        current_sol = list(selected_so_far)
        current_score = sum(1.0 / m for m, _ in current_sol)
        
        for i in range(start_idx, n_moduli):
            m = sorted_moduli[i]
            # Try to find a valid r
            found = False
            for r in range(m):
                if is_compatible(m, r, current_sol):
                    current_sol.append((m, r))
                    current_score += 1.0 / m
                    found = True
                    break
            # If not found, skip m (can't add it)
            
        return current_score, current_sol

    def upper_bound(current_score, start_idx):
        """
        Optimistic bound: assume we can add all remaining moduli.
        """
        bound = current_score
        for i in range(start_idx, n_moduli):
            bound += 1.0 / sorted_moduli[i]
        return bound

    def bnb(idx, selected_so_far, current_score):
        nonlocal global_best_score, global_best_sol
        
        if time.time() - start_time > time_limit:
            return
        
        # If we have decided on K moduli, complete greedily
        if idx == K:
            score, sol = greedy_complete(selected_so_far, K)
            if score > global_best_score:
                global_best_score = score
                global_best_sol = sol
            return
        
        # Prune if upper bound is not better
        if upper_bound(current_score, idx) <= global_best_score:
            return
        
        m = sorted_moduli[idx]
        
        # Option 1: Skip m
        # Note: We can't just "skip" in the sense of not picking a residue,
        # we just don't add m to the list.
        bnb(idx + 1, selected_so_far, current_score)
        
        # Option 2: Try each residue r for m
        # Heuristic: Try residues in a specific order?
        # Maybe try r=0 first?
        for r in range(m):
            if is_compatible(m, r, selected_so_far):
                new_selected = selected_so_far + [(m, r)]
                new_score = current_score + 1.0 / m
                bnb(idx + 1, new_selected, new_score)
                
                if time.time() - start_time > time_limit:
                    return

    # Run B&B
    bnb(0, [], 0.0)
    
    return [[m, r] for m, r in global_best_sol]
