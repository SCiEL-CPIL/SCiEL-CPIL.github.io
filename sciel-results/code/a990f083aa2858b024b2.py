"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Grötzsch graph (Mycielski of C5) on vertices 0..10
    # C5: 0-1-2-3-4-0
    # Mycielski: vertices 0-4 (cycle), 5-9 (copies), 10 (apex)
    edges = set()
    # Cycle edges
    for i in range(5):
        edges.add((i, (i + 1) % 5))
    # Copy edges
    for i in range(5):
        edges.add((i, 5 + (i + 1) % 5))
        edges.add(((i + 1) % 5, 5 + i))
    # Apex edges
    for i in range(5):
        edges.add((5 + i, 10))
    
    edge_list = [list(e) for e in edges]
    
    # Helper: build adjacency
    def build_adj(edge_set, n):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Helper: check triangle-free
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    # Helper: check 3-colorability via backtracking
    def is_3_colorable(adj, n):
        colors = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Get forbidden colors from neighbors
            forbidden = set()
            for u in adj[v]:
                c = colors[u]
                if c != -1:
                    forbidden.add(c)
            for c in range(3):
                if c not in forbidden:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Helper: find maximum independent set (greedy + local search)
    def find_max_independent_set(adj, n):
        # Greedy: pick vertex with min degree, add to IS, remove it and its neighbors
        remaining = set(range(n))
        mis = set()
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            mis.add(chosen)
            remaining.discard(chosen)
            remaining -= adj[chosen]
        return mis
    
    # Helper: find all maximal independent sets (for better search)
    def find_maximal_independent_sets(adj, n, limit=20):
        # Bron-Kerbosch style on complement
        mis_list = []
        
        def bronk(R, P, X):
            if time.time() > deadline - 5:
                return
            if not P and not X:
                if R:
                    mis_list.append(frozenset(R))
                return
            # Choose pivot from P ∪ X
            if P or X:
                pivot_set = P | X
                pivot = next(iter(pivot_set))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                if len(mis_list) > limit:
                    return
        
        bronk(set(), set(range(n)), set())
        return mis_list
    
    # Current state
    current_edges = set(edges)
    current_n = 11
    adj = build_adj(current_edges, 16)
    
    # Verify initial graph
    assert is_triangle_free(adj, 11), "Initial Grötzsch not triangle-free"
    assert not is_3_colorable(adj, 11), "Initial Grötzsch should be non-3-colorable"
    
    # Add vertices 11-15 one by one
    for new_v in range(11, 16):
        if time.time() > deadline - 10:
            break
        
        # Find maximum independent set in current graph (on vertices 0..current_n-1)
        cur_adj = build_adj(current_edges, current_n)
        mis = find_max_independent_set(cur_adj, current_n)
        
        # Try to connect new_v to this MIS
        new_edges = current_edges.copy()
        for u in mis:
            new_edges.add((min(new_v, u), max(new_v, u)))
        
        new_n = current_n + 1
        new_adj = build_adj(new_edges, new_n)
        
        # Check triangle-free
        if not is_triangle_free(new_adj, new_n):
            # Try smaller subsets
            # Remove edges that create triangles
            valid_edges = set()
            for e in new_edges:
                u, v = e
                if u == new_v or v == new_v:
                    other = v if u == new_v else u
                    # Check if adding this edge creates a triangle
                    # Triangle would be new_v - other - w where w is in adj[other] and w in adj[new_v]
                    # Actually just check: is there w such that (other,w) in edges and (new_v,w) in edges?
                    # For now, just skip if it creates a triangle
                    creates_tri = False
                    for w in adj[other]:
                        if (min(new_v, w), max(new_v, w)) in new_edges:
                            creates_tri = True
                            break
                    if not creates_tri:
                        valid_edges.add(e)
                else:
                    valid_edges.add(e)
            new_edges = valid_edges
            new_adj = build_adj(new_edges, new_n)
        
        # Check 3-colorability
        if is_3_colorable(new_adj, new_n):
            # This addition makes it 3-colorable; try a different MIS or skip
            # Try connecting to a smaller set
            # Try each vertex in MIS individually
            best_add = None
            best_count = -1
            for u in sorted(mis, key=lambda x: -len(adj[x])):
                test_edges = current_edges.copy()
                test_edges.add((min(new_v, u), max(new_v, u)))
                test_n = current_n + 1
                test_adj = build_adj(test_edges, test_n)
                if is_triangle_free(test_adj, test_n) and not is_3_colorable(test_adj, test_n):
                    if 1 > best_count:
                        best_add = test_edges
                        best_count = 1
            if best_add is not None:
                current_edges = best_add
            # If nothing works, just add isolated vertex
            else:
                pass  # vertex stays isolated
        else:
            current_edges = new_edges
        
        current_n = new_n
        adj = build_adj(current_edges, current_n)
    
    # Now do local search: try to add more edges
    if time.time() < deadline - 20:
        # Try adding edges between existing vertices
        # For each pair (u,v) not in edges, check if adding it keeps triangle-free and non-3-colorable
        added = True
        while added and time.time() < deadline - 10:
            added = False
            adj = build_adj(current_edges, current_n)
            # Sort candidate edges by potential (both endpoints have high degree)
            candidates = []
            for u in range(current_n):
                for v in range(u + 1, current_n):
                    if (u, v) not in current_edges and (v, u) not in current_edges:
                        # Check triangle-free: no common neighbor
                        common = adj[u] & adj[v]
                        if not common:
                            candidates.append((u, v))
            
            # Try adding edges one at a time (greedy)
            for u, v in candidates:
                if time.time() > deadline - 5:
                    break
                test_edges = current_edges | {(u, v)}
                test_adj = build_adj(test_edges, current_n)
                if is_triangle_free(test_adj, current_n):
                    if not is_3_colorable(test_adj, current_n):
                        current_edges = test_edges
                        added = True
                        adj = build_adj(current_edges, current_n)
                        break
    
    # Final: also try to add edges involving vertices that might have been left isolated
    if time.time() < deadline - 10:
        adj = build_adj(current_edges, current_n)
        for v in range(current_n):
            if time.time() > deadline - 5:
                break
            for u in range(v):
                if (u, v) in current_edges or (v, u) in current_edges:
                    continue
                # Check triangle-free
                if adj[u] & adj[v]:
                    continue
                test_edges = current_edges | {(u, v)}
                test_adj = build_adj(test_edges, current_n)
                if is_triangle_free(test_adj, current_n):
                    if not is_3_colorable(test_adj, current_n):
                        current_edges = test_edges
                        adj = build_adj(current_edges, current_n)
    
    return [list(e) for e in current_edges]
# EVOLVE-BLOCK-END
