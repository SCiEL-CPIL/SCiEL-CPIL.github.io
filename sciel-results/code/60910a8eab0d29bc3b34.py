import time
from math import isqrt

def _squarefree_kernel(n):
    if n <= 1:
        return frozenset()
    result = set()
    f = n
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result.add(d)
        d += 1 if d == 2 else 2
    if f > 1:
        result.add(f)
    return frozenset(result)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    target = frozenset([2, 3, 5, 7])
    empty = frozenset()
    
    state_parent = {empty: (None, None)}
    ach = {empty}
    
    found_M = None
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        kern = _squarefree_kernel(M)
        new_states = set()
        for s in list(ach):
            new_s = s ^ kern
            if new_s not in state_parent:
                state_parent[new_s] = (s, M)
                new_states.add(new_s)
        ach |= new_states
        
        if target in ach:
            found_M = M
            break
    
    if found_M is None:
        return [840]
    
    result = []
    state = target
    while state_parent[state][1] is not None:
        result.append(state_parent[state][1])
        state = state_parent[state][0]
    result.reverse()
    
    # Verify
    prod = 210
    for x in result:
        prod *= x
    sq = isqrt(prod)
    if sq * sq != prod:
        return [840]
    
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool):
            return [840]
        if x <= 210 or x > 10210:
            return [840]
    if len(result) < 1 or len(result) > 100:
        return [840]
    if len(set(result)) != len(result):
        return [840]
    
    return list(result)
