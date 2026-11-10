"""Rewritten program: greedy + local search to find large sum-free-triple-avoiding sets."""
import time
import random

N = 128
DEADLINE = 55  # seconds, safety margin before 60s limit

def _is_valid(s):
    """Check if set s is valid: no three distinct a,b,c in s have all three sums in s."""
    s = set(s)
    for a in s:
        for b in s:
            if a == b:
                continue
            for c in s:
                if a == c or b == c:
                    continue
                if a + b in s and a + c in s and b + c in s:
                    return False
    return True

def _can_add(s, x):
    """Check if adding x to set s keeps it valid."""
    s_list = list(s)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            # Check triple {a, b, x}
            if a + b in s and a + x in s and b + x in s:
                return False
            # Check triple {a, x, ?} - need two other elements
            # Actually we need to check all triples that include x
            # Triples including x: {x, a, b} for all pairs a,b in s
            # We already check {a, b, x}
            # But we also need to check if x with any two elements creates the forbidden pattern
            # The triple is {a, b, c} where a+b, a+c, b+c are all in S
            # If x is one of a,b,c, say x=a, then we need b+c, x+c, b+x all in S
            # x+c = x+c, b+x = b+x, b+c = b+c
            # So we check: for all pairs (b,c) in s, if b+c in s and x+c in s and b+x in s
            # But x is being added, so x+c and b+x are NOT in s yet (they're not in the current s)
            # Wait, no. The condition is about the FINAL set. If we add x, then x is in the set.
            # So x+c could be in the final set if x+c is already in s or equals x (impossible) or is some other value.
            # Actually, x+c is a specific integer. If x+c is in s (the current set), then after adding x,
            # the triple {x, c, b} has sums x+c (in s, so in final), x+b (not in s necessarily), c+b (in s).
            # Hmm, this is getting confusing. Let me re-think.
            
            # The forbidden condition: three DISTINCT members a,b,c such that a+b, a+c, b+c are ALL in the set.
            # When we add x, we need to check all triples that include x.
            # Triples including x: {x, u, v} for distinct u,v in s.
            # Sums: x+u, x+v, u+v.
            # All three must be in the final set S ∪ {x}.
            # x+u is in final set iff x+u in s or x+u == x (no) or x+u == u (no, x≠0).
            # So x+u in final set iff x+u in s.
            # Similarly x+v in final set iff x+v in s.
            # u+v in final set iff u+v in s.
            # So we need: x+u in s AND x+v in s AND u+v in s.
            
            pass  # Already handled below
    
    # Check all pairs (u, v) in s
    for i in range(len(s_list)):
        u = s_list[i]
        for j in range(i + 1, len(s_list)):
            v = s_list[j]
            # Triple {x, u, v}: sums x+u, x+v, u+v must all be in S ∪ {x}
            # x+u in S∪{x} iff x+u in s (since x+u != x for u > 0)
            # x+v in S∪{x} iff x+v in s
            # u+v in S∪{x} iff u+v in s or u+v == x
            if (x + u in s or x + u == x) and (x + v in s or x + v == x) and (u + v in s or u + v == x):
                return False
    return True

def _greedy(order, start_set):
    """Greedy: try to add elements in given order to start_set."""
    s = set(start_set)
    for x in order:
        if x not in s and _can_add(s, x):
            s.add(x)
    return s

def _local_search(s, deadline):
    """Try to improve by swapping or adding elements."""
    s = set(s)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for x in range(1, N + 1):
            if x not in s and _can_add(s, x):
                s.add(x)
                improved = True
        # Try swapping: remove one, add another
        for x in list(s):
            s2 = s - {x}
            for y in range(1, N + 1):
                if y not in s2 and _can_add(s2, y):
                    s_new = s2 | {y}
                    if len(s_new) > len(s):
                        s = s_new
                        improved = True
                        break
            if improved:
                break
    return s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best = set()
    
    # Strategy 1: Start with upper half, greedily add lower elements
    upper = list(range(N // 2 + 1, N + 1))
    lower = list(range(1, N // 2 + 1))
    
    s1 = _greedy(lower, upper)
    if len(s1) > len(best):
        best = set(s1)
    
    if time.time() > deadline:
        return list(best)
    
    # Strategy 2: Random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = _greedy(order, set())
        if len(s) > len(best):
            best = set(s)
    
    if time.time() > deadline:
        return list(best)
    
    # Strategy 3: Local search from best
    if len(best) < N:
        best_ls = _local_search(best, deadline)
        if len(best_ls) > len(best):
            best = best_ls
    
    if time.time() > deadline:
        return list(best)
    
    # Strategy 4: More random restarts with local search
    for _ in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = _greedy(order, set())
        s = _local_search(s, deadline)
        if len(s) > len(best):
            best = set(s)
    
    return list(best)
