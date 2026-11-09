import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    def weight(S):
        return sum(1.0 / x for x in S)
        
    def is_3ap_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b: continue
                # c = 2b - a
                # Check if c is in S and c != a, c != b
                # Actually, simpler:
                # A 3-AP is a, a+d, a+2d.
                # Check if a+d in S and a+2d in S for all a, d.
        # More efficient:
        S_set = set(S)
        max_val = max(S_set) if S_set else 0
        for a in S_set:
            # d can be such that a+d <= max_val
            # d = b - a for b in S_set, b > a
            # Then check if a + 2d in S_set
            for b in S_set:
                if b <= a: continue
                d = b - a
                c = b + d
                if c in S_set:
                    return False
        return True

    def can_add(S_set, x):
        # Check if adding x to S_set creates a 3-AP
        # 1. x, x-d, x-2d -> x-d in S, x-2d in S
        # 2. x-d, x, x+d -> x-d in S, x+d in S
        # 3. x-2d, x-d, x -> x-2d in S, x-d in S (same as 1)
        # Basically, check if x is the middle, end, or start of a 3-AP with elements in S.
        
        # Case 1: x is the middle. a, x, c. c = 2x - a.
        # If a in S and 2x-a in S and a != x and 2x-a != x.
        for a in S_set:
            if a == x: continue
            c = 2*x - a
            if c in S_set and c != x:
                return False
        
        # Case 2: x is the end. a, b, x. b = x - d, a = x - 2d.
        # a = 2b - x.
        # Iterate b in S_set such that b < x.
        # d = x - b. a = x - 2d = 2b - x.
        for b in S_set:
            if b >= x: continue
            a = 2*b - x
            if a in S_set and a != b and a != x:
                return False
                
        return True

    def greedy_construction():
        selected = []
        S_set = set()
        for x in range(1, 257):
            if can_add(S_set, x):
                selected.append(x)
                S_set.add(x)
        return selected, S_set

    def local_search(S_list, S_set):
        improved = True
        while improved:
            improved = False
            # Try swapping a large element x with a smaller element y
            # Sort S_list to iterate
            # We want to replace x with y where y < x.
            # Iterate x in S_list
            for x in S_list:
                if time.time() - start_time > deadline:
                    return S_list, S_set
                for y in range(1, x):
                    if y in S_set:
                        continue
                    # Check if (S \ {x}) U {y} is 3-AP-free
                    # Remove x, add y
                    S_set.remove(x)
                    if can_add(S_set, y):
                        S_set.add(y)
                        # Update list
                        S_list.remove(x)
                        S_list.append(y)
                        S_list.sort()
                        improved = True
                        break # Restart loop with new set
                    else:
                        S_set.add(x)
                if improved:
                    break
        return S_list, S_set

    best_S = []
    best_weight = 0.0

    # 1. Standard Greedy
    S_list, S_set = greedy_construction()
    S_list, S_set = local_search(S_list, S_set)
    w = weight(S_list)
    if w > best_weight:
        best_weight = w
        best_S = S_list[:]

    # 2. Randomized Restarts
    # Perturb the greedy solution
    random.seed(42)
    for i in range(200): # Max 200 restarts, but limited by time
        if time.time() - start_time > deadline:
            break
            
        # Start from greedy
        S_list, S_set = greedy_construction()
        
        # Randomly remove 5-10 elements
        k = random.randint(5, 10)
        for _ in range(k):
            if not S_list: break
            idx = random.randint(0, len(S_list)-1)
            x = S_list[idx]
            S_list.pop(idx)
            S_set.remove(x)
            
        # Re-fill greedily? Or just local search?
        # Let's try to add back elements greedily first to get a maximal set again
        # Actually, local search with swaps might get stuck if the set is not maximal.
        # Let's try a simple greedy fill on the perturbed set.
        # But standard greedy fill starts from 1.
        # Let's just run local search on the perturbed set.
        # But if it's not maximal, we might be able to add elements.
        # Let's add a "fill" step: try to add any y not in S_set in increasing order.
        for y in range(1, 257):
            if y not in S_set and can_add(S_set, y):
                S_set.add(y)
                S_list.append(y)
        S_list.sort()
        
        # Now local search
        S_list, S_set = local_search(S_list, S_set)
        
        w = weight(S_list)
        if w > best_weight:
            best_weight = w
            best_S = S_list[:]

    return best_S

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    def weight(S):
        return sum(1.0 / x for x in S)
        
    def can_add(S_set, x):
        # Check if adding x to S_set creates a 3-AP
        # 1. x is middle: a, x, c => c = 2x-a
        for a in S_set:
            if a == x: continue
            c = 2*x - a
            if c in S_set and c != x:
                return False
        # 2. x is end: a, b, x => a = 2b-x
        for b in S_set:
            if b >= x: continue
            a = 2*b - x
            if a in S_set and a != b and a != x:
                return False
        return True

    def greedy_construction():
        selected = []
        S_set = set()
        for x in range(1, 257):
            if can_add(S_set, x):
                selected.append(x)
                S_set.add(x)
        return selected, S_set

    def local_search(S_list, S_set):
        improved = True
        while improved:
            improved = False
            for x in S_list:
                if time.time() - start_time > deadline:
                    return S_list, S_set
                for y in range(1, x):
                    if y in S_set:
                        continue
                    S_set.remove(x)
                    if can_add(S_set, y):
                        S_set.add(y)
                        S_list.remove(x)
                        S_list.append(y)
                        S_list.sort()
                        improved = True
                        break
                    else:
                        S_set.add(x)
                if improved:
                    break
        return S_list, S_set

    best_S = []
    best_weight = 0.0

    # 1. Standard Greedy
    S_list, S_set = greedy_construction()
    S_list, S_set = local_search(S_list, S_set)
    w = weight(S_list)
    if w > best_weight:
        best_weight = w
        best_S = S_list[:]

    # 2. Randomized Restarts
    import random
    random.seed(42)
    for i in range(1000):
        if time.time() - start_time > deadline:
            break
            
        S_list, S_set = greedy_construction()
        
        # Perturb: remove random elements
        k = random.randint(3, 8)
        for _ in range(k):
            if not S_list: break
            idx = random.randint(0, len(S_list)-1)
            x = S_list[idx]
            S_list.pop(idx)
            S_set.remove(x)
            
        # Re-fill greedily
        for y in range(1, 257):
            if y not in S_set and can_add(S_set, y):
                S_set.add(y)
                S_list.append(y)
        S_list.sort()
        
        # Local Search
        S_list, S_set = local_search(S_list, S_set)
        
        w = weight(S_list)
        if w > best_weight:
            best_weight = w
            best_S = S_list[:]

    return best_S
# EVOLVE-BLOCK-END
