import time
import random

N = 192
MAX_CARD = 48

def _check_add(c, S):
    """
    Check if c can be added to set S without violating constraints.
    S is a set of distinct integers.
    """
    S_list = list(S)
    n = len(S_list)
    
    # 1. c must not divide sum of any 3 distinct elements in S
    # (a + b + d) % c == 0
    if c > 0:
        res_map_c = {}
        for x in S_list:
            r = x % c
            if r not in res_map_c:
                res_map_c[r] = []
            res_map_c[r].append(x)
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                target = (-(a + b)) % c
                if target in res_map_c:
                    # Need d != a, d != b
                    found = False
                    for d in res_map_c[target]:
                        if d != a and d != b:
                            found = True
                            break
                    if found:
                        return False

    # 2. For each a in S, a must not divide c + x + y for distinct x, y in S \ {a, c}
    # (c + x + y) % a == 0  =>  (x + y) % a == (-c) % a
    for a in S_list:
        if a == c:
            continue # c is not in S yet, so a != c always? 
                     # Wait, S does not contain c. So a != c.
        
        target_res = (-c) % a
        
        # Build residue map for elements in S \ {a, c}
        # Since c is not in S, just S \ {a}
        res_map_a = {}
        for x in S_list:
            if x == a:
                continue
            r = x % a
            if r not in res_map_a:
                res_map_a[r] = []
            res_map_a[r].append(x)
            
        for x in S_list:
            if x == a:
                continue
            # We need y in S \ {a, x} such that y % a == (target_res - x % a) % a
            req_res = (target_res - (x % a)) % a
            if req_res in res_map_a:
                # Need y != x
                found = False
                for y in res_map_a[req_res]:
                    if y != x:
                        found = True
                        break
                if found:
                    return False
                    
    return True

def find_violation(S):
    """
    Finds a violation in S.
    Returns (s, a, b, c) or None.
    """
    S_list = list(S)
    n = len(S_list)
    
    for s in S_list:
        # Check if s divides a+b+c for distinct a,b,c in S \ {s}
        res_map = {}
        for x in S_list:
            if x == s:
                continue
            r = x % s
            if r not in res_map:
                res_map[r] = []
            res_map[r].append(x)
            
        for i in range(n):
            a = S_list[i]
            if a == s:
                continue
            for j in range(i + 1, n):
                b = S_list[j]
                if b == s:
                    continue
                target = (-(a + b)) % s
                if target in res_map:
                    for c in res_map[target]:
                        if c != a and c != b:
                            return (s, a, b, c)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
            
        if attempt == 0:
            init_set = set(range(100, N + 1))
        elif attempt == 1:
            init_set = set(range(120, N + 1))
        else:
            cand = list(range(60, N + 1))
            random.shuffle(cand)
            init_set = set(cand[:60])
            
        S = set(init_set)
        
        # Repair
        while time.time() < deadline - 5.0:
            viol = find_violation(S)
            if viol is None:
                break
            s, a, b, c = viol
            min_elem = min(s, a, b, c)
            S.remove(min_elem)
            
        if time.time() > deadline - 5.0:
            continue
            
        # Backfill
        remaining = [x for x in range(1, N + 1) if x not in S]
        # Try larger first?
        remaining.sort(reverse=True)
        
        for c in remaining:
            if time.time() > deadline - 2.0:
                break
            if len(S) >= MAX_CARD:
                break
            if _check_add(c, S):
                S.add(c)
                
        if len(S) > len(best_set):
            best_set = set(S)
            
        if len(best_set) >= MAX_CARD:
            break
            
    return list(best_set)
