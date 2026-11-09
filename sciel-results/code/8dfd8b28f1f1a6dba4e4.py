import time
import random

def construct():
    deadline = time.time() + 55.0
    N = 100
    T = 200
    universe = list(range(1, N + 1))
    
    def is_valid(s):
        bits = 1
        for x in s:
            bits |= (bits << x)
        return ((bits >> T) & 1) == 0

    def greedy(order):
        selected = []
        bits = 1
        for x in order:
            if x <= T:
                if (bits >> (T - x)) & 1:
                    continue
            selected.append(x)
            bits |= (bits << x)
        return selected

    def try_add(s, bits, x):
        if x <= T and ((bits >> (T - x)) & 1):
            return False
        return True

    def local_search(s):
        s = list(s)
        bits = 1
        for x in s:
            bits |= (bits << x)
        
        sset = set(s)
        candidates = [x for x in range(N, 0, -1) if x not in sset]
        
        changed = True
        while changed:
            changed = False
            # 1. Try to add single elements
            for i in range(len(candidates)):
                if time.time() > deadline:
                    return s
                x = candidates[i]
                if try_add(s, bits, x):
                    s.append(x)
                    sset.add(x)
                    bits |= (bits << x)
                    candidates.pop(i)
                    changed = True
                    break # Restart loop as bits changed
            
            if changed:
                continue
                
            # 2. Try to remove 1, add 2 (net gain +1)
            # For each y in s, for each pair x1, x2 in candidates:
            # Check if (S \ {y}) U {x1, x2} is valid.
            # This is expensive. We can prune:
            # If we remove y, bits_new = bits without y? Hard to remove from bitset.
            # Better: Recompute bits for S\{y} is O(N).
            # Then check if x1, x2 can be added.
            
            # Optimization: Only try this if we have time.
            # Given N=100, |s|~50, |cand|~50.
            # 50 * 50 * 50 = 125,000 checks.
            # Each check involves bitset ops.
            
            if time.time() < deadline - 5.0:
                found_swap = False
                for y in s:
                    if time.time() > deadline:
                        return s
                    # Compute bits without y
                    bits_wo_y = 1
                    for x in s:
                        if x != y:
                            bits_wo_y |= (bits_wo_y << x)
                    
                    # Now try to add two elements from candidates
                    # We need to find x1, x2 in candidates such that
                    # bits_wo_y does not have T, and adding x1, x2 doesn't create T.
                    
                    # Actually, if bits_wo_y is valid, we just need to check if x1, x2 can be added to it.
                    # This is equivalent to:
                    # 1. x1 addable to bits_wo_y
                    # 2. x2 addable to (bits_wo_y | (bits_wo_y << x1))
                    
                    # To speed up, we can pre-filter candidates that are addable to bits_wo_y.
                    addable = []
                    for x in candidates:
                        if try_add(s, bits_wo_y, x): # Note: try_add checks T-x in bits
                            addable.append(x)
                    
                    if len(addable) < 2:
                        continue
                        
                    # Try pairs in addable
                    for i in range(len(addable)):
                        x1 = addable[i]
                        bits_after_x1 = bits_wo_y | (bits_wo_y << x1)
                        for j in range(i + 1, len(addable)):
                            x2 = addable[j]
                            if try_add(s, bits_after_x1, x2):
                                # Valid swap found
                                s.remove(y)
                                sset.remove(y)
                                s.append(x1)
                                sset.add(x1)
                                s.append(x2)
                                sset.add(x2)
                                candidates.remove(x1)
                                candidates.remove(x2)
                                # Update bits
                                bits = bits_wo_y | (bits_wo_y << x1) | ((bits_wo_y | (bits_wo_y << x1)) << x2)
                                changed = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                            
        return s

    best = []
    best_len = 0
    
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
    ]
    
    # Add some random orders
    for _ in range(5):
        rand_order = universe[:]
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        s = local_search(s)
        if len(s) > best_len:
            best = s
            best_len = len(s)
            
    if best:
        best = local_search(best)
    else:
        best = list(range(1, 70))
        
    return best
