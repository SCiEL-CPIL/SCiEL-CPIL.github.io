"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    # Build conflict graph: two numbers conflict if they could be part of the same 4-AP
    # Actually, we need: a set is 4-AP-free if no 4 distinct elements form an AP
    # Two numbers alone don't form a 4-AP; we need to think differently.
    # 
    # Better approach: Use greedy + local search refinement
    # Start with randomized greedy, then try to add/remove elements to improve
    
    def is_4ap_free(selected):
        """Check if selected set contains no nonconstant 4-term AP."""
        s = set(selected)
        for a in s:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in s and (a + 2*d) in s and (a + 3*d) in s:
                    return False
        return True
    
    def can_add(x, selected):
        """Check if x can be added to selected without forming a 4-AP."""
        s = set(selected)
        # x as 4th term: need x-3d, x-2d, x-d all in s
        for d in range(1, (x) // 3 + 1):
            if (x - d) in s and (x - 2*d) in s and (x - 3*d) in s:
                return False
        # x as 3rd term: need x-2d, x-d, x+d all in s
        for d in range(1, min(x, n - x) + 1):
            if (x - d) in s and (x - 2*d) in s and (x + d) in s:
                return False
        # x as 2nd term: need x-d, x+d, x+2d all in s
        for d in range(1, min(x, (n - x) // 2 + 1) + 1):
            if (x - d) in s and (x + d) in s and (x + 2*d) in s:
                return False
        # x as 1st term: need x+d, x+2d, x+3d all in s
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an ordering of candidates."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(selected, max_iters=100):
        """Try to improve selected by adding/removing/swapping elements."""
        current = set(selected)
        current_size = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding any element that can be added
            for x in range(1, n + 1):
                if x not in current and can_add(x, current):
                    current.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try removing one element to allow adding two
                for x in list(current):
                    trial = current - {x}
                    # Count how many we can add back
                    added = 0
                    for y in range(1, n + 1):
                        if y not in trial and can_add(y, trial):
                            trial.add(y)
                            added += 1
                    if added > 1:
                        current = trial
                        current_size = len(current)
                        improved = True
                        break
            
            if not improved:
                break
        
        return current
    
    # Phase 1: Multiple randomized greedy passes
    best_set = set()
    best_size = 0
    
    # Deterministic baseline
    selected = greedy_construct(range(1, n + 1))
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
    
    # Randomized greedy
    import random
    candidates = list(range(1, n + 1))
    
    while time.time() < deadline - 10:  # Reserve 10s for local search
        random.shuffle(candidates)
        selected = greedy_construct(candidates)
        
        # Local search refinement
        selected = local_search(selected, max_iters=50)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Phase 2: More aggressive local search on best found
    if time.time() < deadline - 2:
        improved = local_search(best_set, max_iters=200)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
