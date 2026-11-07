"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge to index for faster lookup if needed, though sets of tuples are fine for n=15
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    def get_degrees_and_adj(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations(edges):
        """Returns the number of violating paths."""
        if not edges:
            return 0
        deg, adj = get_degrees_and_adj(edges)
        violations = 0
        for u in range(n):
            # Neighbors of u excluding v
            # We iterate over edges (u, v) in the graph
            for v in adj[u]:
                # S = {deg(s) for s in adj[u] if s != v}
                # T = {deg(t) for t in adj[v] if t != u}
                # If S and T intersect, there is a violation.
                # Specifically, for each d in S ∩ T, there is at least one s with deg(s)=d and one t with deg(t)=d.
                # The number of violating paths for this (u,v) is the sum over d in S∩T of (count_s[d] * count_t[d]).
                
                deg_s_counts = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        deg_s_counts[d] = deg_s_counts.get(d, 0) + 1
                
                if not deg_s_counts:
                    continue
                
                for t in adj[v]:
                    if t != u:
                        dt = deg[t]
                        if dt in deg_s_counts:
                            violations += deg_s_counts[dt]
        return violations

    def is_valid(edges):
        return count_violations(edges) == 0

    def find_violating_edges(edges):
        """Identify edges involved in violations to guide removal."""
        if not edges:
            return set()
        deg, adj = get_degrees_and_adj(edges)
        violating_edges = set()
        
        # We need to check each edge (u,v) and see if it causes a violation.
        # A violation on edge (u,v) exists if there is s~u, t~v (s!=v, t!=u) such that deg(s)==deg(t).
        # If such a pair exists, the edge (u,v) is "implicated".
        
        # To be efficient, we can just check all edges.
        for u, v in edges:
            # Check if adding/keeping this edge causes a conflict
            # Actually, the violation is defined by the path s-u-v-t.
            # The edge (u,v) is the middle edge.
            # If there is any s in N(u)\{v} and t in N(v)\{u} with deg(s)==deg(t), 
            # then the path s-u-v-t is invalid.
            # This implies the edge (u,v) is part of an invalid path.
            
            conflict_found = False
            # Optimization: Pre-compute degree sets for neighbors? 
            # For n=15, direct check is fast enough.
            
            # Let's check degrees of neighbors
            # deg_s set = {deg[s] for s in adj[u] if s != v}
            # deg_t set = {deg[t] for t in adj[v] if t != u}
            
            # If these sets intersect, there is a violation involving edge (u,v).
            deg_s_set = set()
            for s in adj[u]:
                if s != v:
                    deg_s_set.add(deg[s])
            
            for t in adj[v]:
                if t != u:
                    if deg[t] in deg_s_set:
                        conflict_found = True
                        break
            
            if conflict_found:
                violating_edges.add((u, v))
        
        return violating_edges

    def greedy_construct_with_repair():
        """
        Strategy:
        1. Start with empty graph.
        2. Order edges randomly (or by some heuristic).
        3. Try to add edges one by one.
        4. If adding an edge causes a violation, try to repair by removing an offending edge.
           If repair fails, skip the edge.
        5. Repeat with different edge orderings (random restarts).
        """
        best_edges = []
        
        num_restarts = 200
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Shuffle edges to create different orderings
            current_edges = []
            current_set = set()
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
            
            # We can also bias towards high-degree vertices or random.
            # Let's try a mix: sometimes prioritize edges between low-degree vertices (to keep degrees balanced?)
            # Or just random.
            
            for e in shuffled_edges:
                if time.time() > deadline:
                    break
                
                u, v = e
                # Try adding e
                test_edges = current_edges + [e]
                
                if is_valid(test_edges):
                    current_edges = test_edges
                    current_set.add(e)
                else:
                    # Attempt Repair
                    # Find edges in test_edges that are involved in violations
                    violating = find_violating_edges(test_edges)
                    
                    # The new edge e might be the cause, or existing edges might be.
                    # If e is in violating, we might try to remove another violating edge to keep e.
                    # If e is NOT in violating, then e didn't cause a new violation? 
                    # Wait, if test_edges is invalid, there is at least one violating path.
                    # The new edge e could be part of that path.
                    
                    # Heuristic: If e is in violating, try to remove a random other violating edge.
                    # If e is NOT in violating, that means the violation existed before? 
                    # No, current_edges was valid. So e MUST be part of the violation.
                    # Therefore e is always in violating if test_edges is invalid and current_edges was valid.
                    
                    # Let's verify: If current_edges is valid, and we add e, and it becomes invalid,
                    # the new violation must involve e. So e is in violating.
                    
                    # Try to remove one of the OTHER violating edges to allow e to stay.
                    candidates_to_remove = [x for x in violating if x != e]
                    
                    if candidates_to_remove:
                        # Pick the one that seems most "redundant" or random
                        # Let's pick random
                        rem_edge = random.choice(candidates_to_remove)
                        new_edges = [x for x in test_edges if x != rem_edge]
                        
                        if is_valid(new_edges):
                            current_edges = new_edges
                            current_set.remove(rem_edge)
                            current_set.add(e)
                            # Note: We kept the same number of edges, but changed structure.
                            # This might allow adding more edges later.
                    
                    # If no repair worked, we just skip e.
                    
            # End of this restart's construction
            if len(current_edges) > len(best_edges):
                best_edges = current_edges[:]
                
        return best_edges

    # Run the guided greedy strategy
    best_edges = greedy_construct_with_repair()
    
    # Final Polish: Try to add any remaining edge that keeps it valid
    if best_edges:
        best_set = set(best_edges)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for e in all_edges:
                if e in best_set:
                    continue
                new_edges = best_edges + [e]
                if is_valid(new_edges):
                    best_edges = new_edges
                    best_set.add(e)
                    improved = True
                    if time.time() > deadline:
                        return [list(x) for x in best_edges]
    
    # If we didn't find anything (unlikely), fall back to a star
    if not best_edges:
        best_edges = [(0, i) for i in range(1, n)]

    # Ensure types are correct
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
