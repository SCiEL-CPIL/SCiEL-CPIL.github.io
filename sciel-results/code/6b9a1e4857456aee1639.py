import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def check_triangle_free(n, S):
        # S is a set of steps
        D = set()
        for s in S:
            D.add(s)
            D.add(n - s)
        D.discard(0)
        
        # Check if D contains a triangle
        # Triangle if a, b, c in D such that a+b=c or a-b=c (mod n)
        # a, b, c distinct? 
        # Vertices 0, a, a+b.
        # Diffs: a, b, a+b.
        # We need a, b, a+b in D.
        # Also 0, a, a-b. Diffs: a, -b, a-b.
        # So we need to check if there exist a, b in D such that (a+b)%n in D or (a-b)%n in D.
        # Note: if a+b == 0, then a = -b. This is not a triangle (vertices 0, a, 0).
        # So we require a+b != 0 and a-b != 0.
        
        for a in D:
            for b in D:
                if a == b:
                    continue
                c1 = (a + b) % n
                if c1 != 0 and c1 in D:
                    return False
                c2 = (a - b) % n
                if c2 != 0 and c2 in D:
                    return False
        return True

    def has_independent_set_10(n, adj):
        # adj is a list of bitmasks for neighbors
        # We want to find an independent set of size 10.
        # Independent set: no edges between them.
        # So we look for a set of 10 vertices such that for all u, v in set, adj[u] & (1<<v) == 0.
        
        # Greedy first? No, exact.
        # Use DFS with bitmasks.
        # candidates: bitmask of available vertices.
        # We pick a vertex, add to set, remove it and its neighbors from candidates.
        
        found = [False]
        
        def dfs(candidates, count):
            if found[0]:
                return
            if count == 10:
                found[0] = True
                return
            
            # If not enough candidates left
            if candidates.bit_count() < 10 - count:
                return
            
            # Pick a vertex. 
            # Heuristic: pick vertex with max degree in candidate subgraph? 
            # Or just first one.
            # Let's pick the one that removes the most candidates (max degree in induced subgraph).
            # This is expensive to compute. 
            # Just pick first set bit for now, as n is small.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            # New candidates: candidates without v and without neighbors of v
            new_cand = candidates & ~adj[v] & ~lsb
            dfs(new_cand, count + 1)
            
            if found[0]:
                return
            
            # Exclude v
            dfs(candidates & ~lsb, count)
        
        dfs((1 << n) - 1, 0)
        return found[0]

    def build_adj(n, S):
        S_set = set(S)
        adj = [0] * n
        for u in range(n):
            mask = 0
            for s in S_set:
                v1 = (u + s) % n
                v2 = (u - s) % n
                mask |= (1 << v1)
                mask |= (1 << v2)
            adj[u] = mask
        return adj

    def get_edges(n, S):
        S_set = set(S)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                diff = (v - u) % n
                if diff in S_set or (n - diff) in S_set:
                    edges.append([u, v])
        return edges

    best_n = 0
    best_edges = []

    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        max_step = n // 2
        # List of all steps
        steps = list(range(1, max_step + 1))
        
        # Try to find a valid S
        found = False
        
        # Random search first
        for attempt in range(2000):
            if time.time() > deadline:
                break
                
            # Random subset of steps
            # Bias towards higher density?
            # Let's try sizes from n//4 to n//3
            size = random.randint(n // 4, n // 3)
            S = set(random.sample(steps, size))
            
            if not check_triangle_free(n, S):
                continue
                
            adj = build_adj(n, S)
            if not has_independent_set_10(n, adj):
                # Valid!
                best_n = n
                best_edges = get_edges(n, S)
                found = True
                break
        
        if found:
            break
            
        # If not found, try systematic search? 
        # For n=39, 2^19 = 524k subsets. 
        # We can iterate through them.
        # But check_triangle_free is O(|D|^2). 
        # We can optimize check_triangle_free by precomputing conflicts.
        
        # Let's try a few more random attempts with different sizes
        for attempt in range(5000):
            if time.time() > deadline:
                break
            size = random.randint(10, 18)
            S = set(random.sample(steps, size))
            if not check_triangle_free(n, S):
                continue
            adj = build_adj(n, S)
            if not has_independent_set_10(n, adj):
                best_n = n
                best_edges = get_edges(n, S)
                found = True
                break
                
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: n=26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
