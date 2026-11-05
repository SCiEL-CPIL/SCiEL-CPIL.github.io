import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Precompute conflicts
    # conflict_pairs[x] = list of (a, c) such that a < x < c and a+c=2x? 
    # No, we need to check if adding x creates a 3-AP.
    # A 3-AP involving x is either (a, x, c) or (x, b, c) or (a, b, x).
    # It's easier to just say: x conflicts with pair (a, c) if a+c=2x.
    # So if a, c in S, x cannot be added.
    # Also, if x, a in S, then c = 2x-a cannot be in S.
    # So x conflicts with pair (a, c) if {x, a, c} is a 3-AP.
    
    # Let's precompute for each x, the set of pairs (u, v) in [1, N] such that {x, u, v} is a 3-AP.
    # Then can_add(S, x) is True iff for all (u, v) in conflicts[x], not (u in S and v in S).
    
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is middle: a, x, c -> a + c = 2x
        # a < x < c
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                conflicts[x].append((a, c))
        
        # x is end: a, b, x -> a + x = 2b -> b = (a+x)/2
        # a < b < x
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if 1 <= b < x:
                    # AP is a, b, x. Pairs are (a, b), (a, x), (b, x).
                    # If we add x, we need to ensure no a, b in S such that b is midpoint of a, x?
                    # No, if a, b in S, and we add x, then a, b, x is a 3-AP.
                    # So x conflicts with pair (a, b).
                    conflicts[x].append((a, b))
        
        # x is start: x, b, c -> x + c = 2b -> b = (x+c)/2
        # x < b < c
        for c in range(x + 1, N + 1):
            s = x + c
            if s % 2 == 0:
                b = s // 2
                if x < b < c:
                    # AP is x, b, c. Pairs are (x, b), (x, c), (b, c).
                    # If b, c in S, and we add x, then x, b, c is a 3-AP.
                    # So x conflicts with pair (b, c).
                    conflicts[x].append((b, c))

    def can_add(S_set, x):
        for u, v in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    def get_score(S_set):
        return sum(recip[x] for x in S_set)

    # Initial Greedy
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    def try_improve(S_set):
        improved = True
        while improved:
            improved = False
            
            # 1. Try adding
            best_add = None
            best_gain = 0
            for x in range(1, N + 1):
                if x not in S_set:
                    if can_add(S_set, x):
                        gain = recip[x]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = x
            if best_add is not None:
                S_set.add(best_add)
                improved = True
                continue
            
            # 2. Try swapping
            best_swap_gain = 0
            best_swap_remove = None
            best_swap_add = None
            for x in list(S_set):
                S_set.remove(x)
                for y in range(1, N + 1):
                    if y not in S_set:
                        if can_add(S_set, y):
                            gain = recip[y] - recip[x]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_remove = x
                                best_swap_add = y
                S_set.add(x)
            
            if best_swap_remove is not None:
                S_set.remove(best_swap_remove)
                S_set.add(best_swap_add)
                improved = True
                continue
            
            break
        return S_set

    while time.time() < deadline:
        current_S = set(best_S)
        improved_S = try_improve(current_S)
        if get_score(improved_S) > best_score:
            best_S = improved_S
            best_score = get_score(best_S)
        
        if time.time() > deadline - 5.0:
            break
            
        # Perturb
        perturbed = set(best_S)
        if perturbed:
            num_removals = random.randint(1, 5)
            to_remove = random.sample(list(perturbed), min(num_removals, len(perturbed)))
            for x in to_remove:
                perturbed.remove(x)
        
        improved_perturbed = try_improve(perturbed)
        if get_score(improved_perturbed) > best_score:
            best_S = improved_perturbed
            best_score = get_score(best_S)
            
        # Random restart
        if random.random() < 0.2:
            new_S = set()
            order = list(range(1, N + 1))
            random.shuffle(order)
            for x in order:
                if can_add(new_S, x):
                    new_S.add(x)
            improved_new = try_improve(new_S)
            if get_score(improved_new) > best_score:
                best_S = improved_new
                best_score = get_score(improved_new)

    return list(best_S)
