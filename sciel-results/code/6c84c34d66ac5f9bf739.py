import random
import time

def construct():
    n = 12
    # Start with powers of 2
    best_set = [1 << i for i in range(n)]
    best_max = max(best_set)
    
    # Simple local search
    # Try to reduce the largest element
    # This is a heuristic and might not find the global optimum
    # But it's better than nothing if powers of 2 is not optimal.
    
    # However, given the strict constraint and time, 
    # and the fact that powers of 2 is a known solution,
    # I will return powers of 2 to be safe, 
    # unless I can prove a better one quickly.
    
    # Actually, I'll try a few random restarts with a greedy construction 
    # that allows non-minimal steps to create holes.
    
    def is_valid(a):
        sums = {0}
        for x in a:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            if sums & new_sums:
                return False
            sums |= new_sums
        return True

    # Greedy with randomization
    best_max_found = best_max
    for _ in range(100): # 100 restarts
        a = []
        sums = {0}
        for i in range(n):
            # Try candidates from 1 to current_best_max
            # Shuffle candidates to explore different paths
            candidates = list(range(1, best_max_found + 1))
            random.shuffle(candidates)
            found = False
            for x in candidates:
                if x in a:
                    continue
                new_sums = set()
                for s in sums:
                    new_sums.add(s + x)
                if not (sums & new_sums):
                    a.append(x)
                    sums |= new_sums
                    found = True
                    break
            if not found:
                break
        if len(a) == n:
            current_max = max(a)
            if current_max < best_max_found:
                best_max_found = current_max
                best_set = a.copy()
                
    return best_set
