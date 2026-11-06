import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def min_vertex_cover(adj, edges, time_limit):
        n = len(adj)
        if not edges:
            return []
        
        # Initial greedy heuristic for an upper bound
        def greedy_cover():
            remaining_edges = set(edges)
            cover = set()
            while remaining_edges:
                # Find vertex with max degree in remaining edges
                degs = {v: 0 for v in range(n) if v not in cover}
                for u, v in remaining_edges:
                    if u not in cover:
                        degs[u] = degs.get(u, 0) + 1
                    if v not in cover:
                        degs[v] = degs.get(v, 0) + 1
                best_v = max(degs, key=lambda x: degs[x])
                cover.add(best_v)
                # Remove all edges incident to best_v
                new_remaining = set()
                for u, v in remaining_edges:
                    if u != best_v and v != best_v:
                        new_remaining.add((u, v))
                remaining_edges = new_remaining
            return cover

        best_cover = greedy_cover()
        best_size = len(best_cover)
        
        # Branch and Bound
        # State: (current_cover_set, remaining_edges_set)
        # To speed up, we track edges as a list of tuples and use a set for O(1) lookup if needed, 
        # but for small graphs, iterating is fine.
        
        # Pre-compute neighbors for quick access
        # We will use a recursive approach with backtracking
        
        # Lower bound: size of a maximal matching in remaining edges
        def lower_bound(remaining_edges):
            if not remaining_edges:
                return 0
            used = set()
            count = 0
            # Greedy matching for lower bound
            # Sort edges to make it deterministic? Not strictly necessary for LB, but good for consistency
            # Actually, any maximal matching gives a valid LB.
            for u, v in remaining_edges:
                if u not in used and v not in used:
                    used.add(u)
                    used.add(v)
                    count += 1
            return count

        # We store the current cover as a set of vertices
        # And the remaining edges as a set of frozensets or tuples
        # To make it hashable and efficient, let's use a set of tuples (u,v) with u<v
        
        initial_remaining = set(edges)
        
        # Order vertices by degree to help branching? 
        # Branching on an edge (u,v): add u or add v.
        
        def solve(cover, remaining_edges):
            nonlocal best_cover, best_size
            
            # Check time
            if time.time() > time_limit:
                return
                
            if not remaining_edges:
                if len(cover) < best_size:
                    best_size = len(cover)
                    best_cover = set(cover)
                return
            
            # Pruning: if current size + LB >= best_size, stop
            if len(cover) + lower_bound(remaining_edges) >= best_size:
                return
            
            # Pick an edge to branch on
            # Heuristic: pick an edge incident to a high-degree vertex in remaining graph
            # For simplicity, just pick the first one, or one with high degree sum
            # Let's pick an edge (u,v) where u or v has high degree in remaining_edges
            
            # Calculate degrees in remaining graph
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find edge with max degree sum
            max_deg_sum = -1
            branch_edge = None
            for u, v in remaining_edges:
                s = deg[u] + deg[v]
                if s > max_deg_sum:
                    max_deg_sum = s
                    branch_edge = (u, v)
            
            if not branch_edge:
                return
                
            u, v = branch_edge
            
            # Branch 1: Add u to cover
            new_cover_1 = cover | {u}
            new_remaining_1 = set()
            for e in remaining_edges:
                if e[0] != u and e[1] != u:
                    new_remaining_1.add(e)
            solve(new_cover_1, new_remaining_1)
            
            # Branch 2: Add v to cover
            new_cover_2 = cover | {v}
            new_remaining_2 = set()
            for e in remaining_edges:
                if e[0] != v and e[1] != v:
                    new_remaining_2.add(e)
            solve(new_cover_2, new_remaining_2)

        # Start solving
        solve(set(), initial_remaining)
        
        return sorted(list(best_cover))

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        # Check time before starting new graph
        if time.time() > deadline - 5:
             # Not enough time, return best effort (empty or trivial) for remaining
             # But we should have finished previous ones.
             # If we are here, it means we are late.
             # Let's just append empty lists to satisfy return type, though it will be invalid.
             # Better to break and return what we have? No, must return 3 lists.
             # We'll just use the trivial cover for the rest if we run out of time.
             adj, edges = instance(n, p, seed)
             touched = {x for e in edges for x in e}
             results.append(sorted(touched))
             continue
             
        adj, edges = instance(n, p, seed)
        
        # Allocate time for this graph
        # We have roughly (deadline - time.time()) / (3 - index) left
        # But B&B is fast for these sizes. Give it 30s each.
        time_budget = min(30.0, deadline - time.time() - 2.0)
        
        cover = min_vertex_cover(adj, edges, time.time() + time_budget)
        results.append(cover)

    return results
