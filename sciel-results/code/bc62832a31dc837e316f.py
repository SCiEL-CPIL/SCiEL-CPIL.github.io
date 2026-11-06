import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    def conflicts(a, b):
        """Check if (a*b) % (a+b) == 0 for distinct a, b"""
        if a == b:
            return False
        return (a * b) % (a + b) == 0
    
    def check_pairwise_valid(S):
        """Check if all distinct pairs in S are valid"""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                if conflicts(S_list[i], S_list[j]):
                    return False
        return True
    
    # Greedy construction: try multiple orderings
    best_set = set()
    best_size = 0
    
    # Try adding in natural order
    def greedy_add(order):
        S = set()
        for x in order:
            if time.time() > deadline - 1.0:
                break
            # Check if x conflicts with any in S
            ok = True
            for y in S:
                if conflicts(x, y):
                    ok = False
                    break
            if ok:
                S.add(x)
        return S
    
    # Natural order
    S1 = greedy_add(range(1, 257))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
    
    # Reverse order
    S2 = greedy_add(range(256, 0, -1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2
    
    # Random orders
    for trial in range(200):
        if time.time() > deadline - 2.0:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        S = greedy_add(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search: try to add back excluded numbers
    excluded = set(range(1, 257)) - best_set
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for x in list(excluded):
            if time.time() > deadline - 3.0:
                break
            # Try adding x to best_set
            ok = True
            for y in best_set:
                if conflicts(x, y):
                    ok = False
                    break
            if ok:
                best_set.add(x)
                excluded.discard(x)
                improved = True
    
    # Local search: try swapping - remove one, add two
    # For each element in best_set, try removing it and see if we can add more
    if time.time() < deadline - 5.0:
        for x in list(best_set):
            if time.time() > deadline - 5.0:
                break
            # Try removing x
            best_set.remove(x)
            # Try adding all excluded
            added = []
            for y in list(excluded):
                if time.time() > deadline - 5.0:
                    break
                ok = True
                for z in best_set:
                    if conflicts(y, z):
                        ok = False
                        break
                if ok:
                    best_set.add(y)
                    excluded.discard(y)
                    added.append(y)
            
            if len(added) >= 2:
                # Net gain
                pass
            else:
                # Revert: put x back, remove added
                best_set.difference_update(added)
                excluded.update(added)
                best_set.add(x)
                excluded.discard(x)
    
    return sorted(best_set)
