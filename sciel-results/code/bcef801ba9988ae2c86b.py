import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    # Total possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_edges = len(all_edges)
    
    # Precompute neighbors for each edge to quickly check triangle constraints
    # For an edge (u, v), the set of vertices that cannot be connected to both u and v
    # Actually, triangle-free means: if (u,v) and (u,w) exist, then (v,w) cannot exist.
    
    # We will search for the maximum size subset of edges that is:
    # 1. Triangle-free
    # 2. Not 3-colorable
    
    # To do this efficiently, we can use a branch and bound approach on the edge set.
    # However, 16 vertices have 120 edges. 2^120 is too large.
    
    # Alternative Strategy: 
    # The graph must be triangle-free. By Mantel's theorem, max edges is floor(16^2/4) = 64.
    # A complete bipartite graph K_{8,8} has 64 edges and is 2-colorable (hence 3-colorable).
    # We need a triangle-free graph that is NOT 3-colorable.
    # The Grötzsch graph has 11 vertices and 20 edges.
    # Mycielski construction: M(G) is triangle-free, chi(M(G)) = chi(G)+1.
    # M(K_2) = C_5 (5 edges, 5 vertices, chi=3)
    # M(C_5) = Grötzsch (20 edges, 11 vertices, chi=4)
    # M(Grötzsch) would have 2*11+1=23 vertices, 2*20 + 11 = 51 edges, chi=5.
    # We only have 16 vertices.
    
    # Let's try to construct a specific dense graph.
    # Known result: The maximum number of edges in a triangle-free graph on n vertices that is not 3-colorable.
    # For n=16, let's look at known extremal graphs.
    # The "Mycielskian" of a smaller graph might be too big.
    # Let's try to find a subgraph of a larger structure or use a specific construction.
    
    # One strong candidate: Take the Grötzsch graph (11 verts, 20 edges) and add 5 more vertices to maximize edges while keeping triangle-free and 4-chromatic.
    # Or, use a search algorithm specifically for "Maximum Triangle-Free 4-Chromatic Graph".
    
    # Let's implement a fast backtracking solver that tries to build the graph edge by edge.
    # State: (current_edges_set, next_edge_index)
    # We want to maximize |edges|.
    # Pruning:
    # 1. If current graph is 3-colorable, we might still add edges to make it 4-colorable?
    #    No, adding edges can only increase chromatic number or keep it same.
    #    If it's 3-colorable now, it might become 4-colorable later.
    #    But if it's 4-colorable now, we can stop checking colorability and just add more edges if possible.
    #    
    # Better approach: 
    # We know we need chi >= 4.
    # Let's use a heuristic search: 
    # 1. Start with a known 4-chromatic triangle-free graph (e.g., Grötzsch).
    # 2. Try to add remaining vertices/edges to maximize density.
    
    # Let's try a specific construction based on the idea of "blowing up" or adding independent sets.
    
    # Actually, let's use a randomized local search (Simulated Annealing or Hill Climbing) 
    # but with a very strong evaluation function and restarts, which is often faster than exact backtracking for NP-hard problems on small n.
    # However, the prompt asks for a "DIFFERENT strategy" from the current random/greedy one.
    # The current one is basically random restarts + greedy.
    
    # New Strategy: **Exact Branch and Bound with Bitmasks**.
    # Represent the graph as a bitmask of edges.
    # Since n=16, there are 120 edges. A 120-bit integer fits in Python integers.
    # We can iterate through possible edge subsets? No, 2^120 is huge.
    
    # Let's refine the "Exact" approach.
    # We can fix the structure of the graph.
    # Triangle-free graphs are bipartite? No.
    # But they have no K3.
    
    # Let's try a different angle: **Integer Linear Programming (ILP) using scipy.optimize.milp**
    # We want to maximize sum(x_e)
    # Subject to:
    # 1. For every triangle (u,v,w), x_uv + x_uw + x_vw <= 2
    # 2. The graph is not 3-colorable. This is hard to express in linear constraints directly.
    #    However, we can enforce that it contains an odd cycle of length > 3? No, that's not enough.
    #    We need to enforce chi >= 4.
    
    # Alternative: **Backtracking on Colorings**.
    # We want to find a graph G such that no 3-coloring exists.
    # This is equivalent to: For every assignment of colors c: V -> {0,1,2}, there exists an edge (u,v) such that c(u) == c(v).
    # This is a "No-Instance" problem for 3-coloring.
    
    # Let's try a **Constructive Algorithm** based on known dense triangle-free 4-chromatic graphs.
    # The Grötzsch graph has 20 edges on 11 vertices.
    # Can we add 5 vertices to it?
    # Let G0 be Grötzsch. Vertices 0..10.
    # Add vertices 11..15.
    # To keep it triangle-free, new vertices can connect to non-adjacent pairs in G0?
    # If we connect a new vertex v to u and w, u and w must not be adjacent.
    # To maximize edges, we want to connect v to as many vertices as possible, subject to the triangle-free constraint.
    # This is equivalent to finding a maximum independent set in the complement of the neighborhood?
    
    # Let's try to build a graph that is a subgraph of a complete 4-partite graph?
    # No, complete 4-partite graphs contain triangles if any part has size >= 1 and another has size >= 1 and we pick 3 vertices from 3 different parts? 
    # Wait, K_{a,b,c,d} is triangle-free only if at most 2 parts have size >= 1? No.
    # A complete multipartite graph is triangle-free iff it is complete bipartite.
    # So we cannot use complete multipartite graphs with >2 parts.
    
    # Let's go back to **Local Search with a Stronger Move Set**.
    # Current: Random restarts + Greedy Add.
    # New: **Swap-based Local Search**.
    # Start with a valid 4-chromatic triangle-free graph.
    # Move: Remove an edge (u,v) and add a different edge (w,z) if it maintains triangle-free and 4-chromatic, and increases edge count? 
    # Actually, edge count doesn't change with a swap.
    # We want to *increase* edge count.
    # So we need "Add" moves.
    # The problem with greedy add is it gets stuck.
    # What if we do "Add" and "Remove" moves?
    # If we remove an edge, we lose 1 edge. If we then add 2 edges, we gain 1.
    # This is a "2-opt" style move.
    
    # Let's implement **Simulated Annealing** on the edge set.
    # State: Set of edges.
    # Energy: -num_edges (we want to minimize energy, so maximize edges).
    # Constraint: Must be triangle-free and 4-chromatic.
    # If a move violates constraints, reject it (or accept with high penalty).
    # Since we need validity, we only accept moves that maintain validity.
    # This is a "Hard Constraint" search.
    
    # Moves:
    # 1. Add edge (u,v) if triangle-free.
    # 2. Remove edge (u,v).
    # 3. Swap: Remove (u,v), add (w,z).
    
    # We will run this for a fixed time, keeping the best valid state.
    
    import random
    import math
    
    def get_adjacency(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free_fast(adj):
        for u in range(n):
            for v in adj[u]:
                if v < u: continue
                # Check if any common neighbor
                # Intersection of adj[u] and adj[v]
                if adj[u] & adj[v]:
                    return False
        return True
        
    def is_3_colorable_fast(adj):
        # DSATUR heuristic
        color = [-1] * n
        # Order vertices by degree
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # Map original index to position in order for faster access? 
        # Or just use original indices and a visited set.
        
        # We'll do backtracking with forward checking.
        # To speed up, we can use bitmasks for colors.
        
        def dfs(idx, colors_used_mask):
            if idx == n:
                return True
            
            u = order[idx]
            
            # Determine forbidden colors
            forbidden = 0
            for v in adj[u]:
                if color[v] != -1:
                    forbidden |= (1 << color[v])
            
            available = (~forbidden) & 0b111
            if available == 0:
                return False
            
            # Try colors
            while available:
                lsb = available & (-available)
                c = lsb.bit_length() - 1
                available ^= lsb
                
                color[u] = c
                if dfs(idx + 1, colors_used_mask | lsb):
                    return True
                color[u] = -1
                
            return False
            
        return dfs(0, 0)

    # Initialize with Grötzsch Graph
    def get_grotzsch():
        # 5-cycle: 0-1-2-3-4-0
        edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        # Shadow vertices 5-9
        # 5 connected to 0,1
        # 6 connected to 1,2
        # 7 connected to 2,3
        # 8 connected to 3,4
        # 9 connected to 4,0
        for i in range(5):
            edges.append((5+i, i))
            edges.append((5+i, (i+1)%5))
        # Apex 10
        for i in range(5):
            edges.append((10, 5+i))
        return edges

    best_edges = get_grotzsch()
    best_count = len(best_edges)
    
    # Local Search
    current_edges = set(best_edges)
    current_count = best_count
    
    # Precompute all potential edges
    all_potential = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Helper to check if adding an edge keeps it triangle-free
    def can_add_edge(adj, u, v):
        if (u, v) in current_edges or (v, u) in current_edges:
            return False
        # Check common neighbors
        if adj[u] & adj[v]:
            return False
        return True

    # Helper to check if removing an edge keeps it 4-chromatic
    # We only care about 4-chromaticity if we are currently 4-chromatic.
    # If we remove an edge, it might become 3-colorable.
    
    temp_edges = set(current_edges)
    
    while time.time() < deadline - 5:
        # Try random additions
        # Pick a random edge not in the graph
        remaining = [e for e in all_potential if e not in temp_edges]
        if not remaining:
            break
            
        random.shuffle(remaining)
        
        found_improvement = False
        for u, v in remaining:
            if time.time() > deadline - 5:
                break
            
            # Check triangle-free
            adj = get_adjacency(temp_edges)
            if can_add_edge(adj, u, v):
                # Tentatively add
                temp_edges.add((u, v))
                
                # Check 4-chromatic
                if not is_3_colorable_fast(get_adjacency(temp_edges)):
                    # Valid and 4-chromatic!
                    if len(temp_edges) > best_count:
                        best_count = len(temp_edges)
                        best_edges = list(temp_edges)
                    found_improvement = True
                    # Keep this edge, continue searching for more
                    # We don't break because we can add more
                else:
                    # It became 3-colorable, so we must remove it
                    temp_edges.remove((u, v))
        
        if not found_improvement:
            # Try swaps? 
            # Or restart with a perturbation?
            # Let's try to remove a random edge and see if we can add 2 back
            if temp_edges:
                # Pick a random edge to remove
                e_to_remove = random.choice(list(temp_edges))
                temp_edges.remove(e_to_remove)
                # Now try to add edges again
                # This is a "kick"
                pass
            else:
                break

    return best_edges

if __name__ == "__main__":
    # Self-test
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
