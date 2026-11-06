"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Powers of 4 form a valid base set with no triple-sum collisions
    # because in base 4, each digit position is independent for sums of 3 elements
    base = [4**i for i in range(10) if 4**i <= 4096]
    # 4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024, 4^6=4096
    # That's 7 elements. Let's verify: 1,4,16,64,256,1024,4096 - all <= 4096
    
    # Check validity of base
    def is_valid_triple_set(lst):
        seen = set()
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = lst[i] + lst[j] + lst[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True
    
    # Start with powers of 4
    selected = list(base)
    
    # Now try to insert additional elements greedily in random order
    # We'll try multiple random attempts to maximize cardinality
    
    def can_add(lst, candidate):
        """Check if adding candidate to lst keeps all triple sums unique."""
        n = len(lst)
        # Compute all existing triple sums
        existing_sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(lst[i] + lst[j] + lst[k])
        
        # Now check new triples involving candidate
        # Triples with one candidate: candidate + lst[i] + lst[j] for i <= j
        for i in range(n):
            for j in range(i, n):
                s = candidate + lst[i] + lst[j]
                if s in existing_sums:
                    return False
                existing_sums.add(s)
        
        # Triples with two candidates: 2*candidate + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s in existing_sums:
                return False
            existing_sums.add(s)
        
        # Triple with three candidates: 3*candidate
        s = 3 * candidate
        if s in existing_sums:
            return False
        
        return True
    
    # Strategy: Try to insert elements in increasing order of "gap" potential
    # Use randomized restarts with different insertion orders
    
    best_set = list(selected)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        current = list(base)
        
        # Generate candidate pool: all numbers 1..4096 not in base
        candidates = [x for x in range(1, 4097) if x not in set(base)]
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 80:
                break
            if can_add(current, cand):
                current.append(cand)
        
        if len(current) > len(best_set):
            best_set = list(current)
    
    # If we still have time, try a more systematic approach
    if time.time() < deadline - 5 and len(best_set) < 80:
        # Try inserting in sorted order from smallest gaps
        current = list(base)
        # Try all numbers in order
        for cand in range(1, 4097):
            if time.time() > deadline:
                break
            if len(current) >= 80:
                break
            if cand in current:
                continue
            if can_add(current, cand):
                current.append(cand)
        
        if len(current) > len(best_set):
            best_set = list(current)
    
    return best_set[:80]
# EVOLVE-BLOCK-END
