import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute GCDs
    max_m = 128
    gcds = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(2, max_m + 1):
        for j in range(2, max_m + 1):
            gcds[i][j] = math.gcd(i, j)
            
    even_moduli = list(range(2, max_m + 1, 2))
    
    def get_valid_residues(m, assigned):
        valid_r = []
        for r in range(m):
            ok = True
            for r_i, m_i in assigned:
                g = gcds[m][m_i]
                if (r - r_i) % g == 0:
                    ok = False
                    break
            if ok:
                valid_r.append(r)
        return valid_r

    def backtrack(remaining, assigned, best_solution, time_limit):
        if time.time() > time_limit:
            return False
            
        if not remaining:
            if len(assigned) > len(best_solution):
                best_solution[:] = assigned[:]
            return True
            
        # MRV Heuristic + Forward Checking
        best_m = None
        best_valid = None
        min_count = float('inf')
        
        for m in remaining:
            valid_r = get_valid_residues(m, assigned)
            if len(valid_r) < min_count:
                min_count = len(valid_r)
                best_m = m
                best_valid = valid_r
            if min_count == 0:
                return False

        if best_m is None:
            return False
            
        # Update best solution if current partial is better
        if len(assigned) > len(best_solution):
            best_solution[:] = assigned[:]

        random.shuffle(best_valid)
        
        remaining.remove(best_m)
        
        for r in best_valid:
            assigned.append([r, best_m])
            if backtrack(remaining, assigned, best_solution, time_limit):
                remaining.add(best_m)
                assigned.pop()
                return True
            assigned.pop()
            if time.time() > time_limit:
                break
                
        remaining.add(best_m)
        return False

    best_pairs = []
    
    # Phase 1: Greedy
    for _ in range(200):
        if time.time() > deadline:
            break
            
        assigned = []
        moduli = list(even_moduli)
        random.shuffle(moduli)
        
        for m in moduli:
            valid_r = get_valid_residues(m, assigned)
            if valid_r:
                r = random.choice(valid_r)
                assigned.append([r, m])
                
        if len(assigned) > len(best_pairs):
            best_pairs = assigned
            
            if len(best_pairs) == 64:
                return best_pairs

    # Phase 2: Backtracking
    # We try to improve upon best_pairs
    # We run backtracking for the remaining time
    
    # We do a few restarts of backtracking
    # Each restart uses a fresh 'remaining' and 'assigned'
    # But we can seed it with the greedy solution? 
    # Backtracking from scratch is simpler and often faster if it finds a solution quickly.
    # If it doesn't, it explores the tree.
    
    # To utilize time effectively, we can just run backtrack in a loop
    # with a time limit per run, or one big run.
    # One big run is better as it can explore deeply.
    # But if it gets stuck in a bad branch, it wastes time.
    # Randomized restarts help.
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        remaining = list(even_moduli)
        assigned = []
        current_best = best_pairs[:]
        
        # Run backtrack
        # We pass deadline as the hard limit for this specific call?
        # No, we want to use all remaining time.
        # But if we do 5 restarts, we should split the time.
        # Let's just use the global deadline.
        # If it finishes early, we break.
        
        if backtrack(remaining, assigned, current_best, deadline):
            if len(current_best) > len(best_pairs):
                best_pairs = current_best
            if len(best_pairs) == 64:
                break
        else:
            if len(current_best) > len(best_pairs):
                best_pairs = current_best

    return best_pairs
