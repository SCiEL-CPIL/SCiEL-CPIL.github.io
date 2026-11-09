import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _precompute_conflicts(n):
    """
    Precompute all conflict pairs for each element a.
    For 1/a = 1/b + 1/c with distinct a,b,c:
    (b-a)(c-a) = a^2
    So for each a, and each divisor d of a^2 where d != a^2/d:
      b = a + d, c = a + a^2/d
    We store for each element x, the set of elements y that conflict with x
    (i.e., there exists a third element z such that x,y,z form a conflict triple).
    """
    # conflict_graph[x] = set of y such that x and y are in some conflict triple
    # But actually, we need: for a given set S, checking if adding element e
    # creates a conflict means: does there exist a, b in S such that e, a, b form
    # a conflict triple?
    # A conflict triple is (a, b, c) with 1/a = 1/b + 1/c, all distinct.
    # So for a new element e, we need to check if there exist a, b in S such that
    # {e, a, b} is a conflict triple.
    # This means: either e plays the role of a (the "sum" side), or b, or c.
    
    # Let's precompute all conflict triples in [1, n]
    # Then for each element e, store the pairs (a, b) such that {e, a, b} is a conflict triple.
    
    # conflict_triples[e] = list of (a, b) such that e, a, b form a conflict triple
    conflict_triples = {i: [] for i in range(1, n+1)}
    
    for a in range(1, n+1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            d1 = d
            d2 = a2 // d
            if d1 == d2:
                continue
            b = a + d1
            c = a + d2
            if b > n or c > n:
                continue
            # Triple (a, b, c) is a conflict triple
            # For element a: pairs (b, c)
            # For element b: pairs (a, c)
            # For element c: pairs (a, b)
            conflict_triples[a].append((b, c))
            conflict_triples[b].append((a, c))
            conflict_triples[c].append((a, b))
    
    return conflict_triples

def _can_add(s_set, e, conflict_triples):
    """Check if adding element e to s_set creates a conflict."""
    # For each pair (a, b) in conflict_triples[e], check if both a and b are in s_set
    for (a, b) in conflict_triples[e]:
        if a in s_set and b in s_set:
            return False
    return True

def _check_valid(s_set, conflict_triples):
    """Check if the entire set is valid (no conflict triples)."""
    for a in s_set:
        for (b, c) in conflict_triples[a]:
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(n, conflict_triples, order, deadline):
    """Greedy construction following a given order."""
    s = set()
    for e in order:
        if time.time() > deadline:
            break
        if _can_add(s, e, conflict_triples):
            s.add(e)
    return s

def _local_search_1swap(s_set, n, conflict_triples, deadline, max_iter=5000):
    """1-swap local search: remove one element, add another."""
    current = set(s_set)
    best = set(current)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to add a random element not in current
        not_in = list(set(range(1, n+1)) - current)
        if not not_in:
            break
        
        # First, try pure additions
        random.shuffle(not_in)
        added = False
        for e in not_in:
            if _can_add(current, e, conflict_triples):
                current.add(e)
                added = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                break
        
        if added:
            continue
        
        # No pure addition possible. Try 1-swap: remove x, add y
        # Pick a random x from current to remove
        x = random.choice(list(current))
        temp = set(current)
        temp.discard(x)
        
        # Try to add a random y not in temp
        not_in_temp = list(set(range(1, n+1)) - temp)
        random.shuffle(not_in_temp)
        swapped = False
        for y in not_in_temp:
            if _can_add(temp, y, conflict_triples):
                temp.add(y)
                swapped = True
                if len(temp) > best_size:
                    best = set(temp)
                    best_size = len(temp)
                current = temp
                break
        
        if not swapped:
            # Try removing x and not adding anything (if it helps? No, we want to maximize)
            # Just skip
            pass
    
    return best

def _local_search_2swap(s_set, n, conflict_triples, deadline, max_iter=2000):
    """2-swap local search: remove two elements, add two others."""
    current = set(s_set)
    best = set(current)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # First, try pure additions
        not_in = list(set(range(1, n+1)) - current)
        if not_in:
            random.shuffle(not_in)
            for e in not_in:
                if _can_add(current, e, conflict_triples):
                    current.add(e)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    break
            continue
        
        # No pure addition. Try 2-swap
        if len(current) < 2:
            break
        
        x1, x2 = random.sample(list(current), 2)
        temp = set(current)
        temp.discard(x1)
        temp.discard(x2)
        
        not_in_temp = list(set(range(1, n+1)) - temp)
        random.shuffle(not_in_temp)
        
        # Try to add two elements
        added_count = 0
        for y in not_in_temp:
            if added_count >= 2:
                break
            if _can_add(temp, y, conflict_triples):
                temp.add(y)
                added_count += 1
        
        if added_count == 2 and len(temp) > best_size:
            best = set(temp)
            best_size = len(temp)
            current = temp
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    n = 180
    random.seed(42)
    
    # Precompute conflict triples
    conflict_triples = _precompute_conflicts(n)
    
    # Multi-greedy with different orderings
    best_set = set()
    best_size = 0
    
    # Ordering 1: ascending
    orders = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        list(range(1, n+1, 2)) + list(range(2, n+1, 2)),  # odds then evens
        list(range(2, n+1, 2)) + list(range(1, n+1, 2)),  # evens then odds
    ]
    
    # Add some random orderings
    for _ in range(10):
        order = list(range(1, n+1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_build(n, conflict_triples, order, deadline)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search on best
    best_set = _local_search_1swap(best_set, n, conflict_triples, deadline, max_iter=3000)
    best_size = len(best_set)
    
    best_set = _local_search_2swap(best_set, n, conflict_triples, deadline, max_iter=1000)
    best_size = len(best_set)
    
    # More restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        random.seed(10000 + restart * 7)
        order = list(range(1, n+1))
        random.shuffle(order)
        s = _greedy_build(n, conflict_triples, order, deadline)
        s = _local_search_1swap(s, n, conflict_triples, deadline, max_iter=1000)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Final polish: try to add more elements
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        not_in = list(set(range(1, n+1)) - best_set)
        random.shuffle(not_in)
        for e in not_in:
            if _can_add(best_set, e, conflict_triples):
                best_set.add(e)
                improved = True
                break
    
    # Final validation
    if not _check_valid(best_set, conflict_triples):
        # Fallback
        best_set = set(range(91, 181))
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)

# EVOLVE-BLOCK-END
