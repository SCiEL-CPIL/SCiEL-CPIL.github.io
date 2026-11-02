import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def certify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        N2 = n * n
        INF = g
        dist = [INF] * N2
        q = deque()

        for a in range(n):
            for b in adj[a]:
                dist[a * n + b] = 1
                q.append(a * n + b)

        while q:
            s = q.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            p = s // n
            c = s % n
            rev = c * n + p
            dr = dist[rev]
            if dr != INF:
                if d + dr + 1 < g:
                    return False
            for nb in adj[c]:
                if nb == p:
                    continue
                ns = c * n + nb
                nd = d + 1
                if nd < dist[ns]:
                    dist[ns] = nd
                    q.append(ns)
        return True

    def luw_construct(k, g, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n, best_edges = None, None

    if k == 3 and g == 16:
        # Try randomized Bosak construction
        # Target n ~ 936 (L=78) or slightly higher
        # Start L from 78 upwards
        # Use random permutations for each layer
        for L in range(78, 150):
            if time.time() - t0 > deadline - 10:
                break
            
            n = 12 * L
            # Try a few random seeds
            for seed in range(5):
                if time.time() - t0 > deadline - 5:
                    break
                random.seed(seed)
                
                # Define permutations for the 3 edge types
                # Type 0: Forward (i -> i+1)
                # Type 1: Within (i -> i)
                # Type 2: Backward (i -> i-1)
                
                # For each layer i, we need permutations of the 9 vertices (a,b)
                # Actually, the edges are defined globally or per layer?
                # Standard Bosak: edges depend on i.
                # Let's define random permutations for each i and each edge type.
                
                # To simplify, we can use a fixed set of random permutations for all layers,
                # or vary them. Varying them helps break symmetry and cycles.
                
                edges = []
                eset = set()
                
                def vid(i, a, b):
                    return i * 9 + a * 3 + b
                
                # Generate random permutations for the 9 vertices
                # We need 3 permutations per layer?
                # Forward: (a,b) -> (a', b')
                # Backward: (a,b) -> (a'', b'')
                # Within: (a,b) -> (a''', b''')
                
                # To ensure 3-regularity and simplicity, the permutations must be bijections.
                # Also, we must avoid loops (u==v) and multi-edges.
                
                valid = True
                adj = [[] for _ in range(n)]
                
                # Pre-generate random permutations for each layer and type
                # perm_fwd[i] is a permutation of 0..8
                # perm_bwd[i] is a permutation of 0..8
                # perm_within[i] is a permutation of 0..8
                
                # Actually, the "within" edge in standard Bosak is often fixed (e.g., (a,b)->(a,b+1)).
                # But we can randomize it too.
                
                # Let's randomize all three.
                
                for i in range(L):
                    # Random permutation of 0..8
                    p_fwd = list(range(9))
                    random.shuffle(p_fwd)
                    p_bwd = list(range(9))
                    random.shuffle(p_bwd)
                    p_within = list(range(9))
                    random.shuffle(p_within)
                    
                    for a in range(3):
                        for b in range(3):
                            v_idx = a * 3 + b
                            v = vid(i, a, b)
                            
                            # Forward
                            nb_idx = p_fwd[v_idx]
                            nb_a, nb_b = nb_idx // 3, nb_idx % 3
                            nb_i = (i + 1) % L
                            fv = vid(nb_i, nb_a, nb_b)
                            key = (min(v, fv), max(v, fv))
                            if key in eset:
                                valid = False
                                break
                            eset.add(key)
                            adj[v].append(fv)
                            adj[fv].append(v)
                            
                            # Backward
                            nb_idx = p_bwd[v_idx]
                            nb_a, nb_b = nb_idx // 3, nb_idx % 3
                            nb_i = (i - 1) % L
                            bv = vid(nb_i, nb_a, nb_b)
                            key = (min(v, bv), max(v, bv))
                            if key in eset:
                                valid = False
                                break
                            eset.add(key)
                            adj[v].append(bv)
                            adj[bv].append(v)
                            
                            # Within
                            nb_idx = p_within[v_idx]
                            nb_a, nb_b = nb_idx // 3, nb_idx % 3
                            wv = vid(i, nb_a, nb_b)
                            key = (min(v, wv), max(v, wv))
                            if key in eset:
                                valid = False
                                break
                            eset.add(key)
                            adj[v].append(wv)
                            adj[wv].append(v)
                            
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check degree
                if any(len(x) != 3 for x in adj):
                    continue
                
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                
                # Certify girth
                if certify_girth(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break # Found valid for this L, try larger L? No, we want min n.
                    # But maybe a smaller L with different seed works?
                    # We are iterating L upwards. So the first valid L is the smallest for this seed.
                    # But we should check smaller L first?
                    # We started at 78. If 78 fails, we go to 79.
                    # If 78 succeeds, we return it.
                    # But what if L=77 works? We didn't check it.
                    # Moore bound is 510 (L=42.5).
                    # Let's start from L=43 upwards.
        
        # If Bosak didn't find anything, fallback to LUW
        if best_n is None:
            d = 11
            best_n, best_edges = luw_construct(k, g, d)

    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = luw_construct(k, g, d)

    return best_n, best_edges
