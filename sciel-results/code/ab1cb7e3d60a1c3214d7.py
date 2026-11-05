import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Identify bad vertices (2v is a perfect square)
    bad = set()
    for v in range(1, 257):
        if 2 * v in squares:
            bad.add(v)
    
    candidates = [v for v in range(1, 257) if v not in bad]
    n = len(candidates)
    
    # Build conflict adjacency: adj[i] contains j if candidates[i]+candidates[j] is a square
    adj = [set() for _ in range(n)]
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            if ci + candidates[j] in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Complement adjacency for clique finding
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Tomita's MCS
    best_clique = []
    best_size = 0
    
    def mcs(P, clique):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        # Greedy coloring of P
        color = [0] * len(P)
        remaining = list(range(len(P)))
        color_num = 0
        while remaining:
            color_num += 1
            assigned = []
            new_remaining = []
            for idx in remaining:
                # Can P[idx] get this color? Check against already-assigned with this color
                ok = True
                for a in assigned:
                    if P[a] in comp_adj[P[idx]]:
                        ok = False
                        break
                if ok:
                    color[idx] = color_num
                    assigned.append(idx)
                else:
                    new_remaining.append(idx)
            remaining = new_remaining
        
        # Upper bound
        if len(clique) + color_num <= best_size:
            return
        
        # Process in reverse color order
        for pos in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = P[pos]
            new_P = [P[i] for i in range(pos) if P[i] in comp_adj[v]]
            mcs(new_P, clique + [v])
    
    mcs(list(range(n)), [])
    
    result = sorted([candidates[i] for i in best_clique])
    if not result:
        return [1]
    return result
