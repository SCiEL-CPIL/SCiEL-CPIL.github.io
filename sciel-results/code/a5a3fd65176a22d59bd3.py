import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    def is_bad(a, b, c):
        return gcd(a, b) == gcd(a, c) == gcd(b, c)
    
    def can_add(S, x):
        for a, b in combinations(S, 2):
            if is_bad(a, b, x):
                return False
        return True
    
    def greedy(order):
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        return S
    
    def improve(S):
        S = list(S)
        Sset = set(S)
        changed = True
        while changed and (time.time() - start_time) < deadline:
            changed = False
            for x in range(1, 129):
                if x not in Sset and can_add(S, x):
                    S.append(x)
                    Sset.add(x)
                    changed = True
                    break
        return S
    
    best = []
    import random
    
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (x % 4, x)),
        sorted(range(1, 129), key=lambda x: (x % 3, x)),
    ]
    for _ in range(100):
        orders.append(random.sample(range(1, 129), 128))
    
    for order in orders:
        if (time.time() - start_time) >= deadline:
            break
        S = greedy(order)
        S = improve(S)
        if len(S) > len(best):
            best = S
    
    if not best:
        best = greedy(list(range(1, 129)))
    
    return best
