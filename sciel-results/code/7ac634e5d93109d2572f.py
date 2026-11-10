import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 18
    best_edges = None
    best_score = 12  # initial upper bound from K6,6,6
    
    def is_k4_free(edge_set):
        """Check if graph with given edges is K4-free using adjacency matrix."""
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        for a, b, c in combinations(range(n), 3):
            if adj[a][b] and adj[a][c] and adj[b][c]:
                # Triangle (a,b,c) exists; check if any vertex connects to all three
                for d in range(n):
                    if d not in (a, b, c) and adj[d][a] and adj[d][b] and adj[d][c]:
                        return False
        return True
    
    def max_triangle_free_size(edge_set):
        """Find the largest triangle-free induced subgraph using branch and bound."""
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        
        # Build triangle set: which triples form triangles
        # A set S is triangle-free if no triangle is fully contained in S
        
        # Upper bound: for any vertex v, the set of vertices that can be added
        # to {v} without creating a triangle with v is limited.
        
        # Use a simpler approach: iterate over subsets using bitmasks for n=18
        # Actually for n=18, we can do a smarter search.
        
        # Precompute for each pair (u,v) whether they're adjacent
        # A subset S is triangle-free iff for every triangle (a,b,c), not all three are in S
        
        # Strategy: find all maximal triangle-free sets via branching
        # Upper bound for a partial set S: |S| + (number of vertices not in S that can be added)
        
        # For speed, use the fact that if S contains a triangle, it's invalid.
        # We want the maximum |S| such that S is triangle-free.
        
        # Bitmask approach: represent subsets as 18-bit integers
        # For each vertex, precompute which other vertices it's connected to
        # A set S is triangle-free iff for each edge (u,v) in S, u and v share no common neighbor in S
        
        # Better: S is triangle-free iff for each triple in S, at least one edge is missing.
        # Equivalent: the subgraph induced by S has no triangles.
        
        # Let's use a recursive search with pruning.
        # State: (current_set as frozenset, candidates)
        # candidates = vertices not in current_set that can be added without creating a triangle
        
        # A vertex v can be added to set S if v doesn't form a triangle with any pair in S.
        # i.e., for all u, w in S, not (adj[u][v] and adj[w][v] and adj[u][w])
        
        # Precompute: for each vertex v, the set of pairs (u,w) in S that would form a triangle with v
        # i.e., v can be added if there's no edge (u,w) in S such that adj[u][v] and adj[w][v]
        
        # Let's use a different approach: iterate over all 2^18 subsets? That's 262144, feasible!
        
        best = 0
        # For each subset, check if it's triangle-free
        # To speed up: a subset S is triangle-free iff for every triangle in the graph, S doesn't contain all 3 vertices
        
        # Precompute all triangles
        triangles = []
        for a, b, c in combinations(range(n), 3):
            if adj[a][b] and adj[a][c] and adj[b][c]:
                triangles.append((a, b, c))
        
        # Now find max subset avoiding all triangles
        # This is a maximum independent set in the 3-uniform hypergraph of triangles
        # For n=18, we can use bitmask enumeration with pruning
        
        # For each triangle, create a bitmask of the 3 vertices
        tri_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triangles]
        
        # We want max popcount(S) such that for all t in tri_masks, S & t != t
        # i.e., S does not fully contain any triangle
        
        # Use DP or branch and bound on bitmask
        # For n=18, full enumeration with early termination is feasible
        
        # Branch and bound: process vertices in order, decide include/exclude
        # Prune if current_size + remaining <= best
        
        # Sort vertices by degree in triangle hypergraph for better pruning
        # Vertex v is in triangle if it's part of any triangle
        
        # Let's just do a simple recursive search with memoization-like pruning
        # Actually, let's use the approach: try all subsets of size k from n down
        
        # More efficient: use the fact that we want to AVOID triangles
        # Greedy upper bound: start with all vertices, remove one vertex from each triangle
        # This is a hitting set / vertex cover problem on hypergraph
        
        # For n=18, let's do a branch and bound on which vertices to EXCLUDE
        # We want to minimize excluded vertices such that every triangle has at least one excluded vertex
        
        # This is minimum vertex cover in 3-uniform hypergraph
        # Branch on a triangle: exclude one of its 3 vertices
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Find a triangle to branch on
        def min_excluded(vertices_remaining, triangles_remaining):
            """Find minimum number of vertices to exclude from vertices_remaining 
            to hit all triangles in triangles_remaining."""
            if not triangles_remaining:
                return 0
            
            # Upper bound: exclude all vertices in the first triangle
            # Lower bound: 0
            
            # Pick a triangle with fewest options
            # Actually, just pick the first triangle
            tri = triangles_remaining[0]
            
            best = len(vertices_remaining)  # worst case
            
            for v in tri:
                if v not in vertices_remaining:
                    continue
                # Exclude v
                new_vertices = vertices_remaining - {v}
                new_tris = [t for t in triangles_remaining if v not in t]
                
                # Upper bound: 1 + sum of... just recurse
                cost = 1 + min_excluded(new_vertices, new_tris)
                if cost < best:
                    best = cost
            
            return best
        
        # This might be too slow. Let's use a different approach.
        
        # For n=18, let's use a simple approach:
        # The answer is at most 12 (from the tripartite construction)
        # We can check: is there a triangle-free set of size 12? 11? etc.
        
        # Use the complement: we want to find the minimum vertex cover of the triangle hypergraph
        # Then max triangle-free = 18 - min_vertex_cover
        
        # Let's use a greedy + local search for the vertex cover, or
        # use the fact that for small n, we can do a smart enumeration
        
        # Actually, let me just do a direct search for the max triangle-free set
        # using the following: a set S is triangle-free iff for each edge (u,v) in G[S],
        # u and v have no common neighbor in S.
        
        # Let's use a recursive approach with good pruning:
        # Build S one vertex at a time. At each step, we can only add vertices
        # that don't form a triangle with any pair already in S.
        
        # Key insight: if S is current set, a vertex v can be added iff
        # for all u, w in S: NOT (adj[u][v] and adj[w][v] and adj[u][w])
        # i.e., v is not adjacent to both endpoints of any edge in S.
        
        # Let's precompute for efficiency
        
        # For the branch and bound, we'll use:
        # - Current set S (as a list)
        # - Available vertices: those not in S that can be added
        # - Upper bound: len(S) + len(available)
        
        # To make this fast, we'll use bitmasks
        
        # Let me just use a simpler but effective approach for n=18:
        # Iterate through all subsets using a smart ordering
        
        # Actually, the most practical approach for n=18:
        # Use the fact that we know the answer is <= 12
        # Check if there's a triangle-free set of size 12, 11, ...
        
        # For checking if a set of size k is triangle-free:
        # We can use inclusion-exclusion or just check all triangles
        
        # Let me use a different strategy: since n=18 is small,
        # use a greedy randomized approach to find large triangle-free sets,
        # and use that as a lower bound, then verify with exact method.
        
        # For the exact method, let's use the following:
        # The maximum triangle-free induced subgraph can be found by
        # solving a maximum independent set problem on a derived graph.
        
        # Actually, let's just do a straightforward but optimized search:
        
        # Represent the graph as adjacency bitmasks
        adj_mask = [0]*n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # A set S is triangle-free iff for every u in S, 
        # the neighbors of u in S form an independent set in S (no edges among them)
        # Equivalently: for every u in S, adj_mask[u] & mask(S) & adj_mask[u] == 0
        # Wait, that's not right. Let me think again.
        
        # S is triangle-free iff for every triple (a,b,c) in S, at least one of 
        # adj[a][b], adj[a][c], adj[b][c] is False.
        
        # For a given S (as bitmask), S is triangle-free iff:
        # for every a in S, the set (adj_mask[a] & S) has no edges among its vertices
        # i.e., for every a in S, and every b in (adj_mask[a] & S), 
        # (adj_mask[b] & S & adj_mask[a]) == 0
        
        # This is O(n^2) to check for a given S.
        
        # For the search, let's use a branch and bound:
        # Order vertices 0..17
        # At step i, decide whether to include vertex i
        # Prune if current_size + (n - i) <= best
        
        # To improve pruning, use a better upper bound:
        # If we've included some vertices, the remaining vertices that can still be added
        # are limited by the triangle constraints.
        
        # Let me implement a clean branch and bound:
        
        best_size = [0]
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            if idx == n:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                return
            
            # Upper bound
            if current_size + (n - idx) <= best_size[0]:
                return
            
            # Option 1: don't include vertex idx
            search(idx + 1, current_mask, current_size)
            
            # Option 2: include vertex idx if it doesn't create a triangle
            # Check: vertex idx can be added to current_mask iff
            # for every pair (u,v) in current_mask with adj[u][v] and adj[u][idx] and adj[v][idx],
            # that would form a triangle.
            # More precisely: adding idx creates a triangle iff there exist u, v in current_mask
            # such that adj[u][v] and adj[u][idx] and adj[v][idx]
            # i.e., (adj_mask[idx] & current_mask) contains an edge
            
            new_neighbors = adj_mask[idx] & current_mask
            # Check if new_neighbors has any edge
            has_edge = False
            temp = new_neighbors
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if adj_mask[v] & new_neighbors:
                    has_edge = True
                    break
                temp ^= lsb
            
            if not has_edge:
                search(idx + 1, current_mask | (1 << idx), current_size + 1)
        
        search(0, 0, 0)
        return best_size[0]
    
    # Strategy: start with random K4-free graphs and improve via edge addition
    
    def random_k4_free_graph():
        """Generate a random K4-free graph by random edge ordering."""
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        edges = []
        edge_set = set()
        
        for u, v in all_edges:
            # Check if adding (u,v) creates a K4
            # K4 is created if there exist a,b such that {u,v,a,b} forms a K4
            # i.e., u,v,a,b all mutually connected
            # So check: for any a, b where a,b != u,v:
            #   adj[u][a] and adj[v][a] and adj[u][b] and adj[v][b] and adj[a][b]
            #   would create K4 {u,v,a,b}
            
            # More efficiently: a K4 containing edge (u,v) requires two other vertices a,b
            # such that a and b are both connected to u and v, and a is connected to b.
            
            common = []
            for x in range(n):
                if x != u and x != v:
                    if (1 << x) & adj_mask_temp[u] and (1 << x) & adj_mask_temp[v]:
                        common.append(x)
            
            # Check if any pair in common is connected
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    a, b = common[i], common[j]
                    if (1 << b) & adj_mask_temp[a]:
                        break
                else:
                    continue
                break
            else:
                edges.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
                adj_mask_temp[u] |= (1 << v)
                adj_mask_temp[v] |= (1 << u)
                continue
            # If we found a K4, skip this edge
        
        return edges
    
    # Better: build adjacency incrementally
    def generate_random_k4_free():
        adj = [[False]*n for _ in range(n)]
        edges = []
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u,v) creates a K4
            # A K4 with edge (u,v) needs two other vertices a,b both connected to u and v, with a-b edge
            can_add = True
            for a in range(n):
                if a == u or a == v:
                    continue
                if not adj[u][a] or not adj[v][a]:
                    continue
                for b in range(a+1, n):
                    if b == u or b == v:
                        continue
                    if adj[u][b] and adj[v][b] and adj[a][b]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                adj[u][v] = True
                adj[v][u] = True
                edges.append((u, v))
        
        return edges
    
    # Simulated annealing / local search
    # Start with a random K4-free graph, then try to add edges that reduce the max triangle-free set
    
    # First, generate several random K4-free graphs and pick the best
    num_initial = 5
    candidates = []
    
    for i in range(num_initial):
        if time.time() > deadline - 10:
            break
        edges = generate_random_k4_free()
        score = max_triangle_free_size(edges)
        candidates.append((score, edges))
    
    # Also try the tripartite graph as a baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = [[a, b] for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]
    candidates.append((12, tripartite_edges))
    
    # Pick the best candidate
    candidates.sort(key=lambda x: x[0])
    best_score, best_edges = candidates[0]
    
    # Now try to improve by adding edges
    # An edge can be added if it doesn't create a K4
    # Adding an edge can only decrease or maintain the max triangle-free set
    
    if time.time() < deadline - 10:
        # Convert to adjacency for fast K4 check
        adj = [[False]*n for _ in range(n)]
        for u, v in best_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        current_edges = list(best_edges)
        current_score = best_score
        
        # Try adding edges in order of potential benefit
        # An edge (u,v) that is not present can be added if no K4 is formed
        # The benefit is: it might reduce the max triangle-free set
        
        # Generate all non-edges
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
        
        # Shuffle for diversity
        random.shuffle(non_edges)
        
        # Try adding each edge if it helps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in non_edges:
                if time.time() > deadline - 5:
                    break
                if adj[u][v]:
                    continue
                
                # Check if adding (u,v) creates a K4
                can_add = True
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if not adj[u][a] or not adj[v][a]:
                        continue
                    for b in range(a+1, n):
                        if b == u or b == v:
                            continue
                        if adj[u][b] and adj[v][b] and adj[a][b]:
                            can_add = False
                            break
                    if not can_add:
                        break
                
                if not can_add:
                    continue
                
                # Try adding this edge
                new_edges = current_edges + [(u, v)]
                new_score = max_triangle_free_size(new_edges)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    adj[u][v] = True
                    adj[v][u] = True
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = list(current_edges)
    
    # Return the best edges found
    if best_edges is None:
        # Fallback to tripartite
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        best_edges = [[a, b] for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]
    
    # Convert to the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
