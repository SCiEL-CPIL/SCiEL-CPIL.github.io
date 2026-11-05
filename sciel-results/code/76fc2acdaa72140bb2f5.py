"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    elements = list(range(1, 257))
    
    # Precompute: for each pair (b, c) with b <= c, store the set of a < b such that a | (b+c)
    # conflict[a][b][c] = True if a < b <= c and a | (b+c)
    # Use a 2D structure: conflict_bc[b][c] = bitmask of a values that conflict
    
    # Actually, let's precompute for each (b,c) with b<=c, the forbidden a's
    forbidden_a = {}  # (b,c) -> set of a values
    for b in range(1, n+1):
        for c in range(b, n+1):
            s = b + c
            divs = set()
            for a in range(1, b):
                if s % a == 0:
                    divs.add(a)
            forbidden_a[(b,c)] = divs
    
    # For a set S, it's valid iff for all a<b<=c in S, a not in forbidden_a[(b,c)]
    
    # Greedy with random order
    def can_add(s_set, s_list, elem):
        """Check if elem can be added to s."""
        for i, a in enumerate(s_list):
            if a >= elem:
                continue
            # a < elem
            # Check if elem as c: exists b in s with a<b<elem and a in forbidden_a[(b,elem)]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                if b >= elem:
                    break
                if a in forbidden_a[(b, elem)]:
                    return False
            # Check if elem as b: exists c in s with c>=elem and a in forbidden_a[(elem,c)]
            for j in range(len(s_list)):
                c = s_list[j]
                if c < elem:
                    continue
                if a in forbidden_a[(elem, c)]:
                    return False
            # Check if elem as a: exists b,c in s with elem<b<=c and elem in forbidden_a[(b,c)]
            for j in range(len(s_list)):
                b = s_list[j]
                if b <= elem:
                    continue
                for k in range(j, len(s_list)):
                    c = s_list[k]
                    if c < b:
                        continue
                    if elem in forbidden_a[(b, c)]:
                        return False
        return True
    
    def greedy_random():
        order = list(range(1, 257))
        random.shuffle(order)
        s = set()
        s_list = []
        for elem in order:
            if can_add(s, s_list, elem):
                s.add(elem)
                s_list.append(elem)
                s_list.sort()
        return s_list
    
    def local_search(initial_list):
        s = set(initial_list)
        s_list = sorted(s)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding
            candidates = [x for x in range(1, 257) if x not in s]
            random.shuffle(candidates)
            for elem in candidates:
                if can_add(s, s_list, elem):
                    s.add(elem)
                    s_list.append(elem)
                    s_list.sort()
                    improved = True
                    break
            # Try 1-remove, multiple-add
            if not improved and len(s_list) >= 2:
                for _ in range(100):
                    if time.time() > deadline - 3:
                        break
                    idx = random.randint(0, len(s_list)-1)
                    removed = s_list[idx]
                    s.remove(removed)
                    s_list = sorted(s_list)
                    added_count = 0
                    for elem in range(1, 257):
                        if elem in s:
                            continue
                        if can_add(s, s_list, elem):
                            s.add(elem)
                            s_list.append(elem)
                            s_list.sort()
                            added_count += 1
                            if added_count >= 2:
                                break
                    if len(s_list) > len([x for x in initial_list if x in s]) + 1:
                        pass  # net gain
        
        return s_list
    
    best_set = list(range(192, 257))
    best_size = len(best_set)
    
    # Try several strategies
    # Strategy 1: Greedy with different orderings
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        s_list = greedy_random()
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = list(s_list)
    
    # Strategy 2: Local search from good starting points
    if time.time() < deadline - 5:
        # Start from upper tail and try to expand
        s_list = local_search(list(range(160, 257)))
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = list(s_list)
    
    if time.time() < deadline - 5:
        s_list = local_search(list(range(128, 257)))
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = list(s_list)
    
    # More local search iterations
    while time.time() < deadline - 5:
        s_list = greedy_random()
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = list(s_list)
        if time.time() < deadline - 5:
            s_list2 = local_search(s_list)
            if len(s_list2) > best_size:
                best_size = len(s_list2)
                best_set = list(s_list2)
    
    return best_set
# EVOLVE-BLOCK-END
