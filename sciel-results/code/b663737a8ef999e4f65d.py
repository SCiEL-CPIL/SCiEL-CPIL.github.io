import time
import random
from fractions import Fraction

N = 256

def _is_3ap_free(s):
    if not s:
        return True
    s_set = set(s)
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            # Check if a, b, c is AP with c > b
            c = 2 * b - a
            if c > b and c in s_set:
                return False
    return True

def _reciprocal_sum_exact(s):
    total = Fraction(0, 1)
    for x in s:
        total += Fraction(1, x)
    return total

def _greedy_descending():
    """Greedy: try to add numbers from largest to smallest, but prioritize small numbers if possible."""
    s = []
    s_set = set()
    # We want to maximize sum of 1/x.
    # Small x have high weight.
    # Large x have low weight.
    # However, including large x might block multiple small x.
    # Including small x might block many large x (which have low weight anyway).
    # So a heuristic: try to include small numbers first, unless they block other small numbers?
    # Actually, the "Reverse Greedy" (largest first) is often good for MAX SIZE, but here we want MAX WEIGHT.
    # Let's try sorting by 1/x (ascending x) but with a check.
    # If we add 1, we block (2,3), (3,5), etc.
    # If we add 2, we block (1,3), (3,4)...
    # Let's just try a few orders.
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)), # Odds only? No, 1,3,5 is AP.
        list(range(2, N + 1, 2)), # Evens only? 2,4,6 is AP.
    ]
    
    best = set()
    best_score = 0.0
    
    for order in orders:
        s = []
        s_set = set()
        for x in order:
            # Check if x can be added
            can_add = True
            for y in s:
                # Check if x, y, z form AP
                # Case 1: x is middle. y, x, z => z = 2x - y. If z in s_set, fail.
                # Case 2: x is end. y, z, x => z = (x+y)/2? No, z is existing.
                # If y, z, x is AP, then 2z = y + x => z = (y+x)/2.
                # So for each y in s:
                #   1. If 2x - y in s_set: fail (x is middle)
                #   2. If (x + y) % 2 == 0 and (x + y) // 2 in s_set: fail (x is end)
                
                z1 = 2 * x - y
                if z1 in s_set:
                    can_add = False
                    break
                if (x + y) % 2 == 0:
                    z2 = (x + y) // 2
                    if z2 in s_set and z2 != x and z2 != y:
                        can_add = False
                        break
            if can_add:
                s.append(x)
                s_set.add(x)
        
        if s:
            score = sum(1.0 / x for x in s)
            if score > best_score:
                best_score = score
                best = set(s)
    
    return list(best)

def _local_search(initial_set, time_limit=60.0):
    """1-swap and 1-add local search."""
    s = list(initial_set)
    s_set = set(s)
    current_score = sum(1.0 / x for x in s)
    start_time = time.time()
    
    # Precompute conflicts? 
    # For each x in [1..N], list of pairs (a,b) in s such that x forms AP with a,b.
    # This is expensive to recompute.
    
    # Simple approach: 
    # Try adding a number x not in s. If valid, add it.
    # Try removing a number x in s and adding y not in s. If valid and score improves, do it.
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        
        # 1. Try adding
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if x can be added
            can_add = True
            for y in s:
                z1 = 2 * x - y
                if z1 in s_set:
                    can_add = False
                    break
                if (x + y) % 2 == 0:
                    z2 = (x + y) // 2
                    if z2 in s_set and z2 != x and z2 != y:
                        can_add = False
                        break
            if can_add:
                s.append(x)
                s_set.add(x)
                current_score += 1.0 / x
                improved = True
                break # Restart loop after change
        
        if time.time() - start_time > time_limit:
            break
            
        # 2. Try swapping
        if not improved:
            best_swap = None
            best_gain = 0.0
            for x in s:
                # Try removing x
                s_set.remove(x)
                s.remove(x)
                old_score = current_score
                current_score -= 1.0 / x
                
                for y in range(1, N + 1):
                    if y in s_set:
                        continue
                    # Check if y can be added to current s (without x)
                    can_add = True
                    for z in s:
                        w1 = 2 * y - z
                        if w1 in s_set:
                            can_add = False
                            break
                        if (y + z) % 2 == 0:
                            w2 = (y + z) // 2
                            if w2 in s_set and w2 != y and w2 != z:
                                can_add = False
                                break
                    if can_add:
                        gain = 1.0 / y - 1.0 / x
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (x, y)
                
                # Restore x
                s.append(x)
                s_set.add(x)
                current_score = old_score
                
            if best_swap:
                x, y = best_swap
                s.remove(x)
                s_set.remove(x)
                s.append(y)
                s_set.add(y)
                current_score += 1.0 / y - 1.0 / x
                improved = True
    
    return s, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initial solutions
    candidates = []
    
    # 1. Greedy Ascending
    s1 = []
    s1_set = set()
    for x in range(1, N + 1):
        can_add = True
        for y in s1:
            z1 = 2 * x - y
            if z1 in s1_set:
                can_add = False
                break
            if (x + y) % 2 == 0:
                z2 = (x + y) // 2
                if z2 in s1_set and z2 != x and z2 != y:
                    can_add = False
                    break
        if can_add:
            s1.append(x)
            s1_set.add(x)
    candidates.append(s1)
    
    # 2. Greedy Descending
    s2 = []
    s2_set = set()
    for x in range(N, 0, -1):
        can_add = True
        for y in s2:
            z1 = 2 * x - y
            if z1 in s2_set:
                can_add = False
                break
            if (x + y) % 2 == 0:
                z2 = (x + y) // 2
                if z2 in s2_set and z2 != x and z2 != y:
                    can_add = False
                    break
        if can_add:
            s2.append(x)
            s2_set.add(x)
    candidates.append(s2)
    
    # 3. Randomized Greedy (small bias)
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        order = list(range(1, N + 1))
        # Bias towards small numbers: sort by 1/x + noise?
        # Or just shuffle.
        random.shuffle(order)
        
        s_rand = []
        s_rand_set = set()
        for x in order:
            can_add = True
            for y in s_rand:
                z1 = 2 * x - y
                if z1 in s_rand_set:
                    can_add = False
                    break
                if (x + y) % 2 == 0:
                    z2 = (x + y) // 2
                    if z2 in s_rand_set and z2 != x and z2 != y:
                        can_add = False
                        break
            if can_add:
                s_rand.append(x)
                s_rand_set.add(x)
        candidates.append(s_rand)
    
    # Local Search on best candidates
    best_set = []
    best_score = 0.0
    
    for c in candidates:
        if time.time() > deadline - 5:
            break
        if not c:
            continue
        # Validate
        if not _is_3ap_free(c):
            continue
        
        s_ls, score_ls = _local_search(c, time_limit=deadline - time.time() - 1)
        
        if score_ls > best_score:
            best_score = score_ls
            best_set = s_ls
    
    # If no valid set found, return [1]
    if not best_set:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    # Print score
    score = sum(1.0/x for x in res)
    print(f"Score: {score}")
    print(f"Size: {len(res)}")
    print(f"Valid: {_is_3ap_free(res)}")
