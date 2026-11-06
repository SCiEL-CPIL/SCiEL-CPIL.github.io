import time
import random
import numpy as np

def _count_violations(adj, n):
    """Count 4-sets with no common dominator using bitmasks."""
    violations = 0
    # Precompute out-masks
    out_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j]:
                mask |= (1 << j)
        out_masks[i] = mask
    
    # Check all 4-combinations
    from itertools import combinations
    for a, b, c, d in combinations(range(n), 4):
        # Common dominators: vertices that point to a, b, c, d
        # A vertex v dominates {a,b,c,d} if v->a, v->b, v->c, v->d
        # That means v is in the in-neighborhood of each of a,b,c,d
        # Equivalently, v has arcs TO a,b,c,d, so a,b,c,d are in out[v]
        # So we need: for each of a,b,c,d, the dominator v must have v->x for x in {a,b,c,d}
        # This means v must be such that out[v] contains a,b,c,d
        # i.e., out[v] & ((1<<a)|(1<<b)|(1<<c)|(1<<d)) == ((1<<a)|(1<<b)|(1<<c)|(1<<d))
        target = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        # Find if any v (not in {a,b,c,d}) has out[v] containing target
        found = False
        for v in range(n):
            if v == a or v == b or v == c or v == d:
                continue
            if (out_masks[v] & target) == target:
                found = True
                break
        if not found:
            violations += 1
    return violations

def _local_search(adj, n, max_time):
    """Local search to minimize violations."""
    start = time.time()
    violations = _count_violations(adj, n)
    
    while time.time() - start < max_time:
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        # Flip the edge
        adj[i][j] = not adj[i][j]
        adj[j][i] = not adj[j][i]
        
        new_violations = _count_violations(adj, n)
        
        if new_violations < violations:
            violations = new_violations
            if violations == 0:
                break
        else:
            # Revert
            adj[i][j] = not adj[i][j]
            adj[j][i] = not adj[j][i]
        
        if time.time() - start > max_time:
            break
    
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    # Try increasing n from lower bound
    # Erdos lower bound: f(4) >= 31
    for n in range(31, 70):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each n
        best_adj = None
        best_violations = float('inf')
        
        num_restarts = 5 if n <= 40 else 3
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Create random tournament
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = False
                    else:
                        adj[i][j] = False
                        adj[j][i] = True
            
            # Local search with time limit
            remaining = deadline - time.time()
            if remaining < 5:
                break
            search_time = min(remaining * 0.8, 10)  # Cap search time per restart
            violations = _local_search(adj, n, search_time)
            
            if violations < best_violations:
                best_violations = violations
                best_adj = [row[:] for row in adj]
            
            if best_violations == 0:
                break
        
        if best_violations == 0 and best_adj is not None:
            # Convert to orientation list
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i][j]:
                        orientation.append(0)  # i -> j
                    else:
                        orientation.append(1)  # j -> i
            return {'n': n, 'orientation': orientation}
    
    # Fallback: use Paley tournament at n=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
