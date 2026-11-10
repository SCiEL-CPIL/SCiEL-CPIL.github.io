import random
import time
from fractions import Fraction

def _verify(subset):
    """Verify all reciprocal subset sums are distinct."""
    if not subset:
        return True
    sums = {Fraction(0)}
    for x in subset:
        rec = Fraction(1, x)
        new_sums = set()
        for s in sums:
            ns = s + rec
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = new_sums
    return True

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 40.0  # Safety margin
    
    # Known valid seed solution of size 9
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    elements = list(range(1, 23))
    
    # Precompute reciprocal Fractions
    recip = {x: Fraction(1, x) for x in elements}
    
    # Try to find larger sets from k=22 down to best_size+1
    for k in range(22, best_size, -1):
        found = False
        
        # Strategy 1: Deterministic greedy with multiple orderings
        # Try sorted ascending, descending, and a few deterministic orderings
        orderings = []
        orderings.append(sorted(elements))
        orderings.append(sorted(elements, reverse=True))
        # Try orderings by value mod various numbers
        for mod in [2, 3, 5, 7, 11]:
            orderings.append(sorted(elements, key=lambda x: (x % mod, x)))
        
        for order in orderings:
            if time.time() > deadline:
                break
            current_set = []
            current_sums = {Fraction(0)}
            for x in order:
                if len(current_set) == k:
                    break
                rec = recip[x]
                # Check if adding x causes collision:
                # New sums = {s + rec for s in current_sums}
                # Must be disjoint from current_sums
                collision = False
                for s in current_sums:
                    if (s + rec) in current_sums:
                        collision = True
                        break
                if not collision:
                    current_set.append(x)
                    for s in list(current_sums):
                        current_sums.add(s + rec)
            if len(current_set) == k:
                # Final verification
                if _verify(current_set):
                    best_set = current_set
                    best_size = k
                    found = True
                    break
        
        if found:
            break
        
        # Strategy 2: Randomized greedy
        attempts = 0
        while not found and attempts < 20000 and time.time() < deadline:
            attempts += 1
            pool = elements[:]
            random.shuffle(pool)
            
            current_set = []
            current_sums = {Fraction(0)}
            
            for x in pool:
                if len(current_set) == k:
                    break
                rec = recip[x]
                collision = False
                for s in current_sums:
                    if (s + rec) in current_sums:
                        collision = True
                        break
                if not collision:
                    current_set.append(x)
                    for s in list(current_sums):
                        current_sums.add(s + rec)
            
            if len(current_set) == k:
                if _verify(current_set):
                    best_set = current_set
                    best_size = k
                    found = True
                    break
        
        if found:
            break
        
        # Strategy 3: Backtracking with pruning for smaller k
        if k <= 12 and time.time() < deadline:
            # Use a more targeted search
            # Precompute which pairs of elements are "compatible" (adding one after the other doesn't cause immediate issues)
            # Actually, let's try a recursive backtracking
            def backtrack(remaining, current_set, current_sums, depth, max_depth):
                if time.time() > deadline:
                    return None
                if depth == max_depth:
                    return list(current_set)
                if not remaining:
                    return None
                
                # Try each element in remaining
                for i in range(len(remaining)):
                    x = remaining[i]
                    new_remaining = remaining[:i] + remaining[i+1:]
                    
                    rec = recip[x]
                    # Check collision
                    collision = False
                    for s in current_sums:
                        if (s + rec) in current_sums:
                            collision = True
                            break
                    
                    if not collision:
                        new_sums = set(current_sums)
                        for s in current_sums:
                            new_sums.add(s + rec)
                        
                        result = backtrack(new_remaining, current_set + [x], new_sums, depth + 1, max_depth)
                        if result is not None:
                            return result
                return None
            
            result = backtrack(elements, [], {Fraction(0)}, 0, k)
            if result is not None and len(result) == k:
                if _verify(result):
                    best_set = result
                    best_size = k
                    found = True
                    break
    
    return best_set

def construct():
    return _construct_impl()

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END
