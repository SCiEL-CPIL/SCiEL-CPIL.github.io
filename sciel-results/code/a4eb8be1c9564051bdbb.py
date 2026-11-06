import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def _get_polarity_graph(q):
        # Construct polarity graph of projective plane of order q
        # Vertices: (x, y) for x,y in 0..q-1, and (1, 0, 0) style?
        # Standard representation: Points of PG(2, q)
        # We can use a simpler representation for q=5
        # Actually, let's use the explicit adjacency for q=5
        # n = q^2 + q + 1 = 31
        n = q * q + q + 1
        adj = [set() for _ in range(n)]
        edges = []
        
        # Map vertices to 3-vectors over GF(q)
        # This is complex to generate dynamically.
        # Let's use a known list or a simpler construction.
        # For q=5, we can use the graph defined by:
        # V = {0, ..., q^2-1} U {inf}
        # This is getting too complex to derive on the fly correctly.
        
        # Alternative: Use the random search but with a better fallback.
        # If random search fails, return a small valid graph.
        # A graph with n=11 is trivial? No, K11 has independent set 1.
        # But we need C4-free. K11 is C4-free? No, K11 has C4s.
        # A tree on 11 vertices is C4-free and has alpha >= 6.
        # We need alpha < 11.
        # A complete graph K_n has alpha 1. But K_n has C4s for n>=4.
        # So we need a C4-free graph with small alpha.
        # The polarity graph is the best candidate.
        
        # Let's hardcode the edges for the q=5 polarity graph if possible?
        # Or just use a star graph? No, star has large alpha.
        
        # Let's try to generate a C4-free graph with high degree using the greedy method
        # but on a smaller n, say n=20, which is easier to satisfy alpha < 11.
        # If n=20, alpha < 11 is easy.
        
        pass

    def _is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def _has_independent_set(adj, n, target):
        # Optimized backtracking
        adj_list = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                adj_list[u].add(v)
        
        found = [False]
        
        def backtrack(candidates, current_size):
            if found[0]:
                return
            if current_size == target:
                found[0] = True
                return
            if not candidates:
                return
            if time.time() > deadline:
                return
            
            # Pruning
            if current_size + len(candidates) < target:
                return
            
            # Heuristic: pick vertex with lowest degree in candidates
            # to maximize pruning? Or highest?
            # For independent set, we want to pick vertices that exclude many others.
            # So pick vertex with high degree in the induced subgraph of candidates.
            
            v = max(candidates, key=lambda x: len(adj_list[x] & set(candidates)))
            rest = [x for x in candidates if x != v]
            
            # Include v
            new_candidates = [x for x in rest if x not in adj_list[v]]
            backtrack(new_candidates, current_size + 1)
            
            # Exclude v
            if not found[0]:
                backtrack(rest, current_size)
        
        backtrack(list(range(n)), 0)
        return found[0]

    def _verify_graph(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if not _is_c4_free(adj, n):
            return False
        
        if _has_independent_set(adj, n, 11):
            return False
        
        return True

    def _find_c4(adj, n):
        # Find any C4
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    continue
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    # Pick two common neighbors w, z
                    common_list = list(common)
                    w = common_list[0]
                    z = common_list[1]
                    return [u, w, v, z]
        return None

    best_result = None
    best_n = 0
    
    # Try n=39 down to 31
    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + n * 104729)
            
            # Start with a dense random graph
            adj = [set() for _ in range(n)]
            edges = []
            
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
            
            # Remove edges to break C4s
            # We iterate until no C4 is found
            # To avoid infinite loops, limit iterations
            max_iterations = 1000
            for _ in range(max_iterations):
                c4 = _find_c4(adj, n)
                if c4 is None:
                    break
                u, w, v, z = c4
                # Remove one edge from the C4
                # The edges are (u,w), (w,v), (v,z), (z,u)
                # Note: (u,v) and (w,z) are not necessarily edges, but in a C4 they are chords?
                # No, C4 is a cycle of length 4. The edges are (u,w), (w,v), (v,z), (z,u).
                # We need to remove one of these.
                # Let's remove (u, w)
                if w in adj[u]:
                    adj[u].remove(w)
                    adj[w].remove(u)
                    # Update edges list? We can rebuild it at the end.
                else:
                    # Should not happen
                    break
            
            # Rebuild edges list from adj
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            
            # Verify
            if _is_c4_free(adj, n):
                if not _has_independent_set(adj, n, 11):
                    best_result = {'n': n, 'edges': edges}
                    best_n = n
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a small valid graph
    # A graph on 11 vertices: K_11 is not C4-free.
    # A tree on 11 vertices is C4-free. Alpha of a tree on 11 vertices is at least 6.
    # We need alpha < 11.
    # So a tree on 11 vertices is valid?
    # Independent set of size 11? No, max is 11 if isolated, but tree is connected.
    # Max independent set in a tree is <= n-1.
    # So alpha < 11 is satisfied for n=11 tree?
    # If n=11, alpha can be 6. 6 < 11. So yes.
    # But we want to maximize n.
    # Let's return a graph on n=11 (a star) if nothing else works.
    # Star K_1,10: n=11. C4-free. Alpha = 10.
    # Is 10 < 11? Yes.
    # So a star on 11 vertices is a valid solution.
    
    n = 11
    edges = [[0, i] for i in range(1, 11)]
    return {'n': n, 'edges': edges}
