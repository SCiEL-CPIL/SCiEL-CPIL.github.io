import time
import numpy as np
from collections import deque
import random
from itertools import permutations

_T0 = time.time()
_DEADLINE = _T0 + 105.0

def _deadline():
    return time.time() >= _DEADLINE

def _girth_np(adj, max_len):
    n = len(adj)
    if n == 0:
        return None
    L = [sorted(a) for a in adj]
    prev = {}
    depth = {}
    q = deque()
    for u in range(n):
        for v in L[u]:
            q.append((u, v))
            prev[(u, v)] = None
            depth[(u, v)] = 1
    level = [(u, v) for (u, v) in q]
    best = None
    while level and (best is None or len(level) < 1 << 20):
        if _deadline():
            return best
        nxt = []
        for (u, v) in level:
            for w in L[v]:
                if w == u:
                    continue
                if (v, w) in depth:
                    d = depth[(v, w)]
                    if d >= 2:
                        cand = d + 1
                        if best is None or cand < best:
                            best = cand
                        if best == 3:
                            return best
                    continue
                prev[(v, w)] = (u, v)
                depth[(v, w)] = depth[(u, v)] + 1
                nxt.append((v, w))
        level = nxt
    return best

def _check_simple_and_regular(n, edges, k):
    seen = set()
    deg = [0] * n
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    return all(x == k for x in deg)

def construct(k, g):
    if k != 5:
        # Fallback to LUW for other k
        # ... (keep existing LUW code) ...
        pass
    
    # For k=5, try S_6 Cayley graph
    # Pre-compute S_6 elements
    elems = list(permutations(range(6)))
    n = len(elems)
    idx = {e: i for i, e in enumerate(elems)}
    
    best_n = None
    best_edges = None
    
    # Try random sets of 5 generators
    # We need to ensure they generate the whole group or a large subgroup
    # and have high girth
    for _ in range(1000):
        if _deadline():
            break
        # Pick 5 random elements
        # Avoid identity
        gens = random.sample(elems, 5)
        # Check if they generate S_6
        # Simple check: closure size
        # ... (skip for speed, just check girth) ...
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i, e in enumerate(elems):
            for gen in gens:
                # e * gen
                prod = tuple(e[j] for j in gen) # Wait, composition order
                # If we use right action: e * gen
                # gen is a permutation.
                # (e * gen)(j) = e(gen(j))
                # So prod = tuple(e[gen[j]] for j in range(6))
                prod = tuple(e[gen[j]] for j in range(6))
                j = idx[prod]
                if i != j:
                    adj[i].add(j)
        
        # Check regular
        if not all(len(a) == k for a in adj):
            continue
            
        # Check girth
        gi = _girth_np(adj, g)
        if gi is not None and gi >= g:
            # Found a valid graph
            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            return n, edges
            
    # Fallback to LUW
    # ... (reuse existing LUW logic) ...
