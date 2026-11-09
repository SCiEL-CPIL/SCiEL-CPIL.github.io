import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit (150s wall-clock kill)
    
    # Known valid graph: 8 disjoint K4s
    # n=32, alpha=8, C5-free
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find a valid graph with n > 32
    # Strategy: Start with 8 K4s and try to add vertices/edges while maintaining constraints
    
    def has_c5(adj, n):
        """Check if graph with adjacency matrix adj on n vertices contains a C5."""
        # For each pair of non-adjacent vertices, check if they form part of a 5-cycle
        # A C5: v1-v2-v3-v4-v5-v1 where all consecutive pairs are edges and non-consecutive are not
        # Better approach: check all 5-tuples of distinct vertices
        # But that's O(n^5) which is too slow for n=33+
        
        # Alternative: use the fact that in a C5-free graph, the neighborhood structure is restricted
        # For small n (33-35), we can use a more efficient check
        
        # Check if any 5 vertices form a cycle
        # A 5-cycle exists if there exist vertices a,b,c,d,e such that:
        # a-b, b-c, c-d, d-e, e-a are all edges, and no other edges among {a,b,c,d,e}
        
        # More efficient: for each vertex v, look at its neighbors
        # If v has neighbors u1, u2, u3, u4, u5 and there's a path of length 4 between some pair...
        
        # Actually, let's use a simpler but still efficient approach:
        # For each edge (u,v), check if there's a path of length 3 from u to v through 3 intermediate vertices
        # Wait, that would be a 4-cycle, not 5-cycle.
        
        # For a 5-cycle containing edge (u,v): u-w1-w2-w3-v-u
        # So we need: u-w1, w1-w2, w2-w3, w3-v, v-u all edges
        # And w1,w2,w3 distinct and not equal to u or v
        
        # This is still expensive. Let's just check all 5-subsets for n up to 35.
        # C(35,5) = 324,632 which is manageable
        
        from itertools import combinations
        
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            # Check if these 5 vertices form exactly a 5-cycle
            # We need to check all 12 possible 5-cycles on these 5 vertices
            # A 5-cycle is a cyclic ordering where each consecutive pair (including last-first) is an edge
            # and no other pairs are edges
            
            # The 5 vertices can form a C5 in multiple ways. Let's check if the induced subgraph is a C5.
            # Induced subgraph on {a,b,c,d,e} has exactly 5 edges and is a cycle.
            
            vertices = [a, b, c, d, e]
            # Count edges among these 5 vertices
            edge_count = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[vertices[i]][vertices[j]]:
                        edge_count += 1
            
            if edge_count != 5:
                continue
            
            # Check if the induced subgraph is a 5-cycle (each vertex has degree 2 in the induced subgraph)
            is_c5 = True
            for v in vertices:
                deg = 0
                for w in vertices:
                    if v != w and adj[v][w]:
                        deg += 1
                if deg != 2:
                    is_c5 = False
                    break
            
            if is_c5:
                return True
        
        return False
    
    def max_independent_set_size(adj, n):
        """Find the size of the maximum independent set using branch and bound."""
        # Use a greedy + backtracking approach
        # For n up to 35, this can be done with memoization or clever pruning
        
        # Convert to adjacency sets for faster lookup
        adj_sets = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and adj[i][j]:
                    adj_sets[i].add(j)
        
        # Independent set: no two vertices are adjacent
        # Use branch and bound
        best = [0]
        
        def backtrack(candidates, current_size, chosen):
            if time.time() - start_time > deadline - 5:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best[0]:
                return
            
            # Choose a vertex to branch on
            # Pick vertex with highest degree among candidates for better pruning
            v = max(candidates, key=lambda x: len(adj_sets[x] & set(candidates)))
            
            # Option 1: include v
            new_candidates = [u for u in candidates if u != v and u not in adj_sets[v]]
            backtrack(new_candidates, current_size + 1, chosen + [v])
            
            # Option 2: exclude v
            backtrack([u for u in candidates if u != v], current_size, chosen)
        
        backtrack(list(range(n)), 0, [])
        return best[0]
    
    def is_valid(adj, n):
        """Check if graph is C5-free and has alpha <= 8."""
        if has_c5(adj, n):
            return False
        if max_independent_set_size(adj, n) > 8:
            return False
        return True
    
    # Try to construct graphs with n = 33, 34, 35
    # Strategy: Start from the 8-K4 graph and try to add vertices with carefully chosen edges
    
    for target_n in [33, 34, 35]:
        if time.time() - start_time > deadline - 10:
            break
        
        # Start with the 8 K4s (32 vertices)
        adj = [[0] * target_n for _ in range(target_n)]
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[base + i][base + j] = 1
                    adj[base + j][base + i] = 1
        
        # Try to add the new vertex(es) with various edge patterns
        # New vertex indices: 32, 33, 34
        new_vertices = list(range(32, target_n))
        
        # Try different strategies for connecting new vertices
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and time.time() - start_time < deadline - 10:
            attempts += 1
            
            # Randomly assign edges from new vertices to old vertices
            for nv in new_vertices:
                for ov in range(32):
                    # Probability of adding edge
                    if random.random() < 0.3:
                        adj[nv][ov] = 1
                        adj[ov][nv] = 1
            
            # Randomly assign edges between new vertices
            for i in range(len(new_vertices)):
                for j in range(i + 1, len(new_vertices)):
                    if random.random() < 0.3:
                        adj[new_vertices[i]][new_vertices[j]] = 1
                        adj[new_vertices[j]][new_vertices[i]] = 1
            
            # Check validity
            if is_valid(adj, target_n):
                # Extract edges
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                break  # Found a valid graph for this target_n, move to next
    
    # Return the best result found
    return {'n': best_n, 'edges': best_edges}
