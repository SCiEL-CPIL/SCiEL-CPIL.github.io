import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_c4_free(n, adj):
        # adj is list of sets
        for i in range(n):
            for j in range(i + 1, n):
                if len(adj[i] & adj[j]) > 1:
                    return False
        return True

    def find_c4(n, adj):
        # Returns (u, v, a, b) such that u-a-v-b-u is a C4
        # i.e., a, b are common neighbors of u, v
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    common_list = list(common)
                    a = common_list[0]
                    b = common_list[1]
                    return (i, j, a, b)
        return None

    def would_create_c4(n, adj, u, v):
        # Check if adding edge (u,v) creates a C4
        # C4 exists if there is a path u-a-b-v of length 3
        # a in N(u), b in N(v), a-b in E
        # Also need a != b, a != v, b != u
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[v]:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    return True
        return False

    def try_construct(n, target_deg, max_time=10.0):
        t_start = time.time()
        # Start with a graph with min degree > target_deg
        start_deg = target_deg + 4
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Generate random edges until min degree is start_deg
        # Use random shuffling of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if min(deg) >= start_deg:
                break
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        
        if min(deg) < start_deg:
            # If we couldn't reach start_deg (unlikely for small n), try higher density
            # Or just proceed with what we have if it's close
            pass

        # Local search: remove edges to break C4s
        iterations = 0
        while time.time() - t_start < max_time:
            c4 = find_c4(n, adj)
            if c4 is None:
                if min(deg) >= target_deg:
                    return {'n': n, 'edges': [[u, v] for u, v in edges]}
                else:
                    # Need to add edges
                    # Try to add random edges that don't create C4
                    added = False
                    for u, v in all_edges: # Reuse shuffled list? No, need new random order
                        if v in adj[u]:
                            continue
                        if not would_create_c4(n, adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            deg[u] += 1
                            deg[v] += 1
                            edges.append((u, v))
                            added = True
                            if min(deg) >= target_deg:
                                # Check if still C4 free (it should be)
                                if is_c4_free(n, adj):
                                    return {'n': n, 'edges': [[u, v] for u, v in edges]}
                            break
                    if not added:
                        # Stuck, restart or break
                        break
            else:
                u, v, a, b = c4
                # Remove a random edge from the C4
                # Edges are (u,a), (a,v), (v,b), (b,u)
                candidates = [(u, a), (a, v), (v, b), (b, u)]
                random.shuffle(candidates)
                for ru, rv in candidates:
                    if rv in adj[ru]:
                        adj[ru].remove(rv)
                        adj[rv].remove(ru)
                        deg[ru] -= 1
                        deg[rv] -= 1
                        # Remove from edges list
                        # This is O(E), but E is small
                        try:
                            edges.remove((ru, rv))
                        except ValueError:
                            try:
                                edges.remove((rv, ru))
                            except ValueError:
                                pass
                        break
            
            iterations += 1
            if iterations > 10000:
                break
        
        return None

    best_result = None
    
    # Try n=47 first
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
            
        target_deg = n - 39
        
        # Multiple attempts
        for attempt in range(10):
            if time.time() > deadline:
                break
            res = try_construct(n, target_deg, max_time=5.0)
            if res is not None:
                # Verify
                adj = [set() for _ in range(n)]
                for u, v in res['edges']:
                    adj[u].add(v)
                    adj[v].add(u)
                if is_c4_free(n, adj):
                    degs = [len(adj[i]) for i in range(n)]
                    if min(degs) >= target_deg:
                        best_result = res
                        break
        
        if best_result is not None:
            break

    if best_result is None:
        # Fallback: n=39 empty graph
        best_result = {'n': 39, 'edges': []}
        
    return best_result
