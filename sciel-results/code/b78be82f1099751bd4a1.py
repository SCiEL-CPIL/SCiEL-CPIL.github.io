import time
import random
import sys

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin

def _time_left():
    return _DEADLINE - time.time()

def _can_add(S, x):
    """
    Check if x can be added to set S.
    S is a set of ints.
    Condition 1: x does not divide sum of any 3 distinct elements in S.
    Condition 2: For each a in S, a does not divide sum of x and any 2 distinct other elements in S \ {a}.
    """
    if not S:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    # Condition 1: x does not divide sum of any 3 in S
    # Iterate all triples in S
    # To speed up, we can just iterate
    for i in range(n):
        b = S_list[i]
        for j in range(i+1, n):
            c = S_list[j]
            for k in range(j+1, n):
                d = S_list[k]
                if (b + c + d) % x == 0:
                    return False
    
    # Condition 2: For each a in S, a does not divide (x + b + c) for any b,c in S\{a}
    for a in S_list:
        # We need to check pairs (b, c) from S \ {a}
        # Sum = x + b + c
        # Check (x + b + c) % a == 0
        for i in range(n):
            b = S_list[i]
            if b == a: continue
            for j in range(i+1, n):
                c = S_list[j]
                if c == a: continue
                if (x + b + c) % a == 0:
                    return False
    return True

def _greedy(order):
    S = set()
    for x in order:
        if x in S:
            continue
        if len(S) >= _MAX:
            break
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, time_budget=10.0):
    S = set(S)
    start = time.time()
    best = set(S)
    best_len = len(S)
    
    while time.time() - start < time_budget and _time_left() > 5:
        improved = False
        
        # 1. Try to add any missing element
        if len(S) < _MAX:
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            for x in candidates:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break # Restart loop to check further additions
        
        # 2. Try 1-out-1-in swap
        if not improved and len(S) > 0:
            elems = list(S)
            random.shuffle(elems)
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            
            # Limit iterations to avoid long stalls
            for a in elems[:20]:
                S_temp = S - {a}
                for x in candidates[:50]:
                    if _can_add(S_temp, x):
                        # Valid swap
                        S = S_temp
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
                if improved:
                    break
        
        # 3. Try 1-out-2-in (Net +1)
        if not improved and len(S) >= 1 and len(S) < _MAX:
            elems = list(S)
            random.shuffle(elems)
            candidates = [v for v in range(1, _N + 1) if v not in S]
            random.shuffle(candidates)
            
            for a in elems[:10]:
                S_temp = S - {a}
                # Try to add two
                found = False
                for x in candidates[:30]:
                    if _can_add(S_temp, x):
                        S_temp2 = S_temp | {x}
                        for y in candidates[:30]:
                            if y == x: continue
                            if _can_add(S_temp2, y):
                                S = S_temp2 | {y}
                                improved = True
                                found = True
                                if len(S) > best_len:
                                    best_len = len(S)
                                    best = set(S)
                                break
                        if found:
                            break
                if found:
                    break
        
        if not improved:
            # Perturb: Remove 2 random elements to escape local optimum
            if len(S) > 2:
                to_remove = random.sample(list(S), 2)
                S -= set(to_remove)
            else:
                # Stuck, try random restart of local search part?
                # Just break and let outer loop handle
                break
                
    return best

def construct():
    random.seed(42)
    best_set = set()
    
    # Phase 1: Greedy with different orders
    orders = [
        list(range(_N, 0, -1)),
        list(range(1, _N + 1)),
        # Biased towards high
        list(range(100, _N + 1)) + list(range(1, 100)),
        list(range(150, _N + 1)) + list(range(1, 150)),
    ]
    
    for order in orders:
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Phase 2: Randomized Greedy
    for _ in range(10):
        if _time_left() < 30: break
        # Weighted order
        weights = {v: random.random() + v/100.0 for v in range(1, _N+1)}
        order = sorted(range(1, _N+1), key=lambda v: -weights[v])
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S

    # Phase 3: Local Search
    if _time_left() > 10:
        budget = min(30.0, _time_left() - 5)
        S = _local_search(best_set, time_budget=budget)
        if len(S) > len(best_set):
            best_set = S

    return list(best_set)
