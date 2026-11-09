import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = Fraction(5, 31)
    
    # Known solution with max denominator 248
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    # Search for smaller max denominator M
    # Denominators must be >= 7 because 1/6 > 5/31
    min_den = 7
    
    # Iterate M downwards from 247 to min_den
    for M in range(247, min_den - 1, -1):
        if time.time() > deadline:
            break
            
        # DP: reachable sums using denominators from M down to min_den
        # State: Fraction sum -> list of denominators used
        # To save memory, we can just check feasibility first, then find path?
        # Or just keep the path.
        
        reachable = {Fraction(0): []}
        found = False
        
        # Iterate denominators from M down to min_den
        # We add 1/i to existing sums
        for i in range(M, min_den - 1, -1):
            new_reachable = {}
            inv_i = Fraction(1, i)
            
            for s, path in reachable.items():
                ns = s + inv_i
                if ns <= target:
                    # If we've seen this sum before, keep the one with fewer terms?
                    # Or just any. Fewer terms might be faster to verify?
                    # Actually, we just need one solution.
                    if ns not in new_reachable:
                        new_reachable[ns] = path + [i]
                    else:
                        # Optional: keep shorter path
                        if len(path) < len(new_reachable[ns]):
                            new_reachable[ns] = path + [i]
            
            # Merge
            for ns, path in new_reachable.items():
                if ns not in reachable:
                    reachable[ns] = path
                else:
                    if len(path) < len(reachable[ns]):
                        reachable[ns] = path
                        
            if target in reachable:
                best_solution = reachable[target]
                best_max_den = M
                found = True
                break
                
            # Pruning: if reachable gets too big, maybe skip?
            # But for M < 248, it should be small.
            # Let's add a safety check
            if len(reachable) > 50000:
                break
                
        if found:
            break
            
    return best_solution
