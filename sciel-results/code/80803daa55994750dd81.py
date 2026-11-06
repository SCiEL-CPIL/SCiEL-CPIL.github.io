"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n_max = 256
    best_set = None
    best_len = 0
    
    # Precompute conflicts: conflict[a] = set of (b,c) pairs where a|b+c and a<b<=c
    conflict = [set() for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            for c in range(b, n_max + 1):
                if (b + c) % a == 0:
                    conflict[a].add((b, c))
    
    # Precompute for each element x, which pairs (y,z) in the set would conflict with x
    # When x is added, we need to check:
    # 1. x as 'a': any (b,c) in conflict[x] both in set
    # 2. x as 'b' or 'c': for each a in set with a<x, check if (x,c) or (b,x) in conflict[a]
    
    def check_valid(s):
        """Fast validity check using precomputed conflicts."""
        s_set = set(s)
        for a in s:
            for (b, c) in conflict[a]:
                if b in s_set and c in s_set:
                    return False
        return True
    
    def can_add(x, current_set):
        """Check if x can be added to current_set."""
        if x in current_set:
            return False
        
        # Case 1: x is the smallest (a = x)
        for (b, c) in conflict[x]:
            if b in current_set and c in current_set:
                return False
        
        # Case 2: x is b or c in a triple (a, x, c) or (a, b, x) where a < x
        for a in current_set:
            if a >= x:
                continue
            # Check (x, c) in conflict[a] for c >= x in current_set
            for c in current_set:
                if c >= x and (x, c) in conflict[a]:
                    return False
            # Check (b, x) in conflict[a] for a < b <= x in current_set
            for b in current_set:
                if a < b <= x and (b, x) in conflict[a]:
                    return False
        
        return True
    
    def greedy_extend(start_set, num_passes=5):
        """Extend a set by trying to add smaller elements, multiple passes."""
        current = set(start_set)
        
        for _ in range(num_passes):
            if time.time() > deadline:
                break
            
            min_curr = min(current) if current else n_max + 1
            candidates = list(range(1, min_curr))
            random.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(x, current):
                    current.add(x)
                    added_any = True
            
            if not added_any:
                break
        
        return current
    
    def local_search_improve(current_set):
        """Improve the set via swap operations with fast validity checks."""
        current = set(current_set)
        current_sorted = sorted(current)
        n = len(current_sorted)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            current_sorted = sorted(current)
            n = len(current_sorted)
            
            for i in range(n):
                if time.time() > deadline:
                    return current
                
                out_elem = current_sorted[i]
                # Try replacing out_elem with a smaller element not in current
                candidates = [x for x in range(1, out_elem) if x not in current]
                random.shuffle(candidates)
                
                for in_elem in candidates:
                    if time.time() > deadline:
                        return current
                    
                    # Try the swap
                    trial = current - {out_elem} | {in_elem}
                    if check_valid(trial):
                        current = trial
                        improved = True
                        break
                
                if improved:
                    break
        
        return current
    
    # Try multiple starting points - upper tails
    for start in range(128, n_max + 1):
        if time.time() > deadline:
            break
        
        start_set = set(range(start, n_max + 1))
        
        # Verify base set is valid
        if not check_valid(start_set):
            continue
        
        if len(start_set) > best_len:
            best_set = set(start_set)
            best_len = len(start_set)
        
        # Try to extend with smaller elements
        extended = greedy_extend(start_set, num_passes=3)
        
        if len(extended) > best_len:
            best_set = set(extended)
            best_len = len(best_set)
        
        # Local search to improve
        if time.time() < deadline:
            improved_set = local_search_improve(extended)
            if len(improved_set) > best_len:
                best_set = set(improved_set)
                best_len = len(best_set)
    
    # Additional: try some random initial sets and extend them
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Random subset biased toward larger elements
        size = random.randint(80, 160)
        start_set = set(random.sample(range(64, n_max + 1), min(size, n_max - 64)))
        
        if not check_valid(start_set):
            continue
        
        if len(start_set) > best_len:
            best_set = set(start_set)
            best_len = len(start_set)
        
        extended = greedy_extend(start_set, num_passes=2)
        
        if len(extended) > best_len:
            best_set = set(extended)
            best_len = len(extended)
        
        if time.time() < deadline:
            improved_set = local_search_improve(extended)
            if len(improved_set) > best_len:
                best_set = set(improved_set)
                best_len = len(best_set)
    
    # Final polish: try to add any remaining elements
    if best_set is not None and time.time() < deadline:
        final = set(best_set)
        for x in range(1, n_max + 1):
            if time.time() > deadline:
                break
            if x not in final and can_add(x, final):
                final.add(x)
        
        if len(final) > best_len:
            best_set = final
            best_len = len(final)
    
    # If we didn't find anything, return a known valid set
    if best_set is None:
        best_set = set(range(192, n_max + 1))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
