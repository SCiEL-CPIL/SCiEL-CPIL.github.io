"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def triple_sums_valid(selected):
        """Check if all unordered triples (with repetition) have unique sums."""
        n = len(selected)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = selected[i] + selected[j] + selected[k]
                    if s in sums:
                        return False
                    sums.add(s)
        return True

    def can_add(selected, x, existing_sums):
        """Check if x can be added to selected without creating duplicate triple sums.
        Returns True if valid, and the updated set of sums if so."""
        n = len(selected)
        # New triples involving x:
        # (x,x,x), (x,x,sel[i]), (x,sel[i],sel[j]), (sel[i],sel[j],sel[k]) where at least one is x
        # Actually, we need to check all triples that include x:
        # (x,x,x)
        # (x,x,sel[i]) for i in range(n)
        # (x,sel[i],sel[j]) for i<=j in range(n)
        new_sums = set()
        
        # (x,x,x)
        s = 3*x
        if s in existing_sums or s in new_sums:
            return False, None
        new_sums.add(s)
        
        # (x,x,sel[i])
        for i in range(n):
            s = 2*x + selected[i]
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # (x,sel[i],sel[j]) for i <= j
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in existing_sums or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums

    def build_greedy():
        """Greedy construction: add numbers in increasing order."""
        selected = []
        triple_sums = set()
        
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 15:
                break
            
            ok, new_sums = can_add(selected, x, triple_sums)
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    def local_search(selected, max_iterations=5000):
        """Try to improve by swapping elements."""
        if time.time() > DEADLINE - 15:
            return selected
        
        best = list(selected)
        best_size = len(best)
        
        # Try to add more elements by replacing existing ones
        # First, try adding elements that fit
        current_sums = set()
        n = len(best)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    current_sums.add(best[i] + best[j] + best[k])
        
        # Try to find elements that can be added
        added = False
        for x in range(1, N + 1):
            if x in best:
                continue
            ok, new_sums = can_add(best, x, current_sums)
            if ok:
                best.append(x)
                current_sums.update(new_sums)
                added = True
                if len(best) >= MAX_SIZE:
                    break
            if time.time() > DEADLINE - 15:
                break
        
        if added:
            return best
        
        # Try swaps: replace one element with another to allow more additions
        for _ in range(max_iterations):
            if time.time() > DEADLINE - 15:
                break
            if len(best) >= MAX_SIZE:
                break
            
            # Pick a random element to remove
            idx = random.randint(0, len(best) - 1)
            removed = best[idx]
            candidate = best[:idx] + best[idx+1:]
            
            # Recompute sums
            n = len(candidate)
            cand_sums = set()
            valid = True
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        s = candidate[i] + candidate[j] + candidate[k]
                        if s in cand_sums:
                            valid = False
                            break
                        cand_sums.add(s)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Try to add a new element
            for x in range(1, N + 1):
                if x in candidate or x == removed:
                    continue
                ok, new_sums = can_add(candidate, x, cand_sums)
                if ok:
                    candidate.append(x)
                    # Check if this is an improvement (it should be since we added one)
                    if len(candidate) > len(best):
                        best = candidate
                        break
                if time.time() > DEADLINE - 15:
                    break
            
            if len(best) >= MAX_SIZE:
                break
        
        return best

    # Start with greedy
    result = build_greedy()
    
    # Try to improve with local search
    if time.time() < DEADLINE - 20:
        result = local_search(result)
    
    # Final validation
    if not triple_sums_valid(result):
        # Fallback to a simple valid set
        fallback = []
        for x in range(1, N + 1):
            if len(fallback) >= 1:
                if triple_sums_valid(fallback + [x]):
                    fallback.append(x)
            else:
                fallback.append(x)
            if len(fallback) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 5:
                break
        result = fallback
    
    return result
# EVOLVE-BLOCK-END
