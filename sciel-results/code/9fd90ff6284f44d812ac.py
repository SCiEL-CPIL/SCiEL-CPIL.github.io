"""Construct a C5-free graph with independence number < 9, maximizing n."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Base: 8 disjoint K4s, n=32, alpha=8
    base_n = 32
    base_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    best_n = 32
    best_edges = base_edges
    
    # Try to find a valid graph with n=33
    # Strategy: start from a 33-vertex graph that is close to 8 K4s + 1 vertex
    # The 9th vertex must connect to vertices in different K4s to keep alpha <= 8
    # and must not create C5
    
    def has_c5(n, edge_set):
        """Check if graph contains a 5-cycle (as subgraph, not necessarily induced)."""
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each ordered pair (u,v) with u<v, check if there's a path of length 3 from u to v
        # going through two intermediate vertices, forming a 5-cycle
        # Actually, let's check for C5 by looking at paths of length 4 between adjacent vertices
        # A C5 is: v0-v1-v2-v3-v4-v0 where all are distinct
        
        # More efficient: for each edge (u,v), check if there's a path of length 3 from u to v
        # that doesn't use the edge (u,v) directly, forming a 5-cycle
        # Wait, a C5 has 5 vertices. If we have edge (u,v) and a path v-a-b-c-v of length 4... no.
        
        # Let me think again. A 5-cycle v0-v1-v2-v3-v4-v0.
        # Pick an edge (v0,v1). Then we need a path v1-v2-v3-v4-v0 of length 3 (3 edges) from v1 to v0.
        # So for each edge (u,v), check if there exists a path of exactly 3 edges from u to v not using (u,v).
        # But this path could pass through u or v again... no, for a simple cycle, all vertices distinct.
        
        # Alternative approach: enumerate 5-tuples? Too slow for n=33.
        # Better: for each pair of vertices at distance 2, check if their neighbors have an edge.
        
        # Actually, let's use a simpler check:
        # A graph contains C5 iff there exists a 5-cycle. 
        # For n<=35, we can try: for each vertex v, look at its neighbors, and for each pair of neighbors (a,b),
        # check if a-b edge exists (triangle) - no, that's C3.
        
        # Let me just do: for each edge (u,v), find all common non-adjacent paths of length 3.
        # Simpler: iterate over all 5-combinations? C(33,5) = 237336, times checking if they form C5.
        # That's feasible.
        
        from itertools import combinations
        
        # Precompute adjacency matrix for fast lookup
        adj_mat = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj_mat[u][v] = True
            adj_mat[v][u] = True
        
        for combo in combinations(range(n), 5):
            # Check if these 5 vertices form a C5 in some ordering
            # A C5 on vertices {v0,v1,v2,v3,v4} means there's a cyclic ordering where each consecutive pair is an edge
            # We can check: the induced subgraph on these 5 vertices must have exactly 5 edges forming a cycle
            # Actually, it's a subgraph C5, so we need to find a permutation that forms a cycle
            
            # For efficiency: check if the 5-vertex induced subgraph contains a Hamiltonian cycle
            # The number of distinct 5-cycles on 5 labeled vertices is 3 (since (5-1)!/2 = 3)
            a, b, c, d, e = combo
            vertices = [a, b, c, d, e]
            
            # Check the 3 distinct cycle orderings (up to reversal and rotation)
            # Ordering 1: a-b-c-d-e-a
            if (adj_mat[a][b] and adj_mat[b][c] and adj_mat[c][d] and adj_mat[d][e] and adj_mat[e][a]):
                return True
            # Ordering 2: a-b-c-e-d-a
            if (adj_mat[a][b] and adj_mat[b][c] and adj_mat[c][e] and adj_mat[e][d] and adj_mat[d][a]):
                return True
            # Ordering 3: a-b-d-c-e-a
            if (adj_mat[a][b] and adj_mat[b][d] and adj_mat[d][c] and adj_mat[c][e] and adj_mat[e][a]):
                return True
        return False
    
    def alpha_leq_k(n, edge_set, k):
        """Check if independence number <= k using a simple greedy + exact for small n."""
        # Build adjacency matrix
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Use Bron-Kerbosch style or simple backtracking for max independent set
        # For n=33, this could be slow. Let's use a heuristic first, then verify.
        
        # Actually, let's use a simple exact algorithm: branch and bound
        # Or, since we know the structure (mostly K4s), let's use a smarter approach
        
        # For the 33-vertex candidate, it will be 8 K4s + 1 extra vertex connected to some others
        # The independent set can have at most 1 from each K4, plus possibly the extra vertex
        # So alpha <= 8 + (1 if extra vertex is isolated from the chosen ones)
        # Let's just do a backtracking search
        
        sys.setrecursionlimit(10000)
        
        def find_max_independent(vertices_remaining, current_size, current_set):
            if not vertices_remaining:
                return current_size
            
            # If current_size + len(vertices_remaining) <= current_best, prune
            # We'll track global best
            
            v = vertices_remaining[0]
            rest = vertices_remaining[1:]
            
            # Option 1: don't include v
            result1 = find_max_independent(rest, current_size, current_set)
            
            # Option 2: include v (only if v is not adjacent to any in current_set)
            if not any(v in adj[u] for u in current_set):
                # Remove v and all its neighbors
                new_remaining = [u for u in rest if u not in adj[v]]
                result2 = find_max_independent(new_remaining, current_size + 1, current_set + [v])
                return max(result1, result2)
            
            return result1
        
        # This naive approach is too slow for n=33. Let's use a different strategy.
        # Since the graph is close to disjoint K4s, we can exploit that structure.
        
        # Alternative: use the fact that for 8 K4s + 1 vertex, the max independent set is at most 9.
        # We need to check it's at most 8.
        
        # Let's just try all combinations of at most 1 vertex from each "group"
        # But we don't have groups explicitly. Let's just do a simple random sampling + greedy.
        
        # Actually, for the specific construction we're trying, let's just check directly.
        # The 33rd vertex x connects to some vertices. An independent set of size 9 would need
        # to pick 1 from 8 K4s (8 vertices) and also x, with x not connected to any of the 8.
        # So we just need to check: is there a choice of one vertex from each K4 such that
        # x is not connected to any of them? If no such choice exists, then alpha <= 8.
        
        # But this is specific to the structure. Let's do a general check using a better algorithm.
        
        # Use: repeatedly find maximum independent set via greedy coloring bound
        # For now, let's use a simple but potentially slow exact method for n=33
        
        # Actually, let's just return True optimistically and verify separately for the final answer.
        # Or better: use networkx? No, only stdlib, numpy, scipy.
        
        # Let's do a simple backtracking with good pruning
        vertices = list(range(n))
        vertices.sort(key=lambda v: len(adj[v]))  # high degree first
        
        max_size = [0]
        
        def backtrack(remaining, current_size):
            if not remaining:
                if current_size > max_size[0]:
                    max_size[0] = current_size
                return
            
            # Pruning: can't exceed current_size + len(remaining)
            if current_size + len(remaining) <= max_size[0]:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Don't take v
            backtrack(rest, current_size)
            
            # Take v
            new_remaining = [u for u in rest if u not in adj[v]]
            backtrack(new_remaining, current_size + 1)
        
        backtrack(vertices, 0)
        return max_size[0] <= k
    
    # Now try to construct a 33-vertex graph
    # Start with 8 K4s (vertices 0-31) and add vertex 32
    # Vertex 32 must connect to at least one vertex in each K4 to prevent alpha=9
    # (because if there's a K4 with no connection to 32, we can pick one from each K4 + 32 = 9 independent)
    # Also must not create C5
    
    # Simplest: connect vertex 32 to exactly one vertex in each K4
    # Let's try connecting to the first vertex of each K4: 0, 4, 8, 12, 16, 20, 24, 28
    
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Choose which vertex in each K4 to connect to vertex 32
        connections = [random.randint(0, 3) for _ in range(8)]
        edges_33 = base_edges + [[32, 4 * i + connections[i]] for i in range(8)]
        
        n_33 = 33
        edge_set_33 = set(tuple(sorted(e)) for e in edges_33)
        
        # Check C5
        if has_c5(n_33, edge_set_33):
            continue
        
        # Check alpha <= 8
        if alpha_leq_k(n_33, edge_set_33, 8):
            best_n = 33
            best_edges = edges_33
            break
    
    # Return best found
    return {'n': best_n, 'edges': best_edges}
