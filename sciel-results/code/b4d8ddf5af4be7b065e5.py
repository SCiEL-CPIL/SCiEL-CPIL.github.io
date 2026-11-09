import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

L = 1
for i in range(1, 25):
    L = _lcm(L, i)

vals = [0] * 25
for i in range(1, 25):
    vals[i] = L // i

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = set()
    
    def greedy(order):
        current = set()
        reachable = {0}
        for x in order:
            if vals[x] not in reachable:
                current.add(x)
                new_sums = set()
                for s in reachable:
                    ns = s + vals[x]
                    if ns <= L:
                        new_sums.add(ns)
                reachable = reachable.union(new_sums)
        return current
    
    # Try multiple orderings for greedy warm-start
    orderings = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        sorted(range(1, 25), key=lambda x: vals[x]),  # smallest vals first
        sorted(range(1, 25), key=lambda x: -vals[x]),  # largest vals first
    ]
    
    for order in orderings:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Backtracking search
    current_set = set()
    reachable_sums = {0}
    
    def solve(x, remaining):
        nonlocal best_size, best_set, current_set, reachable_sums
        
        if time.time() > deadline:
            return
            
        if len(current_set) + remaining <= best_size:
            return
            
        if x == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Option 1: Include x (if valid)
        if vals[x] not in reachable_sums:
            current_set.add(x)
            old_reachable = reachable_sums
            new_sums = set()
            for s in old_reachable:
                ns = s + vals[x]
                if ns <= L:
                    new_sums.add(ns)
            reachable_sums = old_reachable.union(new_sums)
            
            solve(x - 1, remaining - 1)
            
            reachable_sums = old_reachable
            current_set.remove(x)
        
        # Option 2: Exclude x
        solve(x - 1, remaining)
    
    solve(24, 24)
    
    if not best_set:
        best_set = {24}
        
    return sorted(best_set)
