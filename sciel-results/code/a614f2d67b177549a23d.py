"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_csp(n, edges, lists, timeout):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        domains = [set(l) for l in lists]
        
        # Check initial validity
        for d in domains:
            if not d:
                return None
                
        color = [None] * n
        start_time = time.time()
        
        # Increase recursion limit to handle depth up to n
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, n + 100))
        
        def backtrack(assigned_count):
            if time.time() - start_time > timeout:
                return None
                
            if assigned_count == n:
                return color[:]
                
            # MRV Heuristic
            min_size = float('inf')
            var = -1
            for i in range(n):
                if color[i] is None:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var = i
                        if min_size == 0:
                            return None
                        if min_size == 1:
                            break
            
            if var == -1:
                return color[:]
            
            # Save domains for backtracking
            saved_domains = [d.copy() for d in domains]
            
            # Try each value in domain[var]
            # Order: Least Constraining Value (LCV) could help, but let's try simple iteration first
            # To make it deterministic and potentially faster for finding *a* solution, 
            # we can just iterate.
            for val in domains[var]:
                color[var] = val
                
                consistent = True
                for u in adj[var]:
                    if color[u] is None:
                        if val in domains[u]:
                            domains[u].remove(val)
                            if not domains[u]:
                                consistent = False
                                break
                    else:
                        if color[u] == val:
                            consistent = False
                            break
                
                if consistent:
                    res = backtrack(assigned_count + 1)
                    if res is not None:
                        return res
                
                # Restore domains and color for next value
                for i in range(n):
                    domains[i] = saved_domains[i].copy()
                color[var] = None
            
            return None
        
        try:
            return backtrack(0)
        finally:
            sys.setrecursionlimit(old_limit)

    out = []
    global_start = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        remaining = deadline - (time.time() - global_start)
        if remaining <= 1.0:
            # Not enough time for solving, use fallback
            sol = [l[0] for l in lists]
        else:
            sol = solve_csp(n, edges, lists, timeout=remaining)
            if sol is None:
                sol = [l[0] for l in lists]
        
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
