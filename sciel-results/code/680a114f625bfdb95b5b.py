import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Known valid construction: 8 disjoint K4s, n=32, alpha=8
    # This is the best known lower bound for R(C5, K9)
    def make_k4_union(n_blocks=8, block_size=4):
        n = n_blocks * block_size
        edges = []
        for b in range(n_blocks):
            start = b * block_size
            for i in range(block_size):
                for j in range(i + 1, block_size):
                    edges.append([start + i, start + j])
        return {'n': n, 'edges': edges}
    
    best_result = make_k4_union(8, 4)
    
    # Try to find a valid graph with n > 32
    # Strategy: Start with K4 union and try to add vertices/edges
    # or try random graphs with constraints
    
    # First, try n=33 with a modified structure
    # We need alpha < 9, so at most 8 vertices can be independent
    # In a K4 union of 8 blocks, alpha = 8
    # To add a 33rd vertex, we need to ensure it doesn't create a C5
    # and that alpha remains < 9
    
    # Try: 7 K4s + 1 K5? No, K5 has C5.
    # Try: 7 K4s + 1 K3 + 1 vertex? That's 28+3+1=32.
    # Try: 8 K4s + 1 vertex connected to some vertices
    
    # Let's try a random approach for n=33
    # A C5-free graph with high minimum degree
    # By Turán's theorem, C5-free graphs have limited edge density
    
    # Actually, let's try a different known construction
    # The Clebsch graph is C5-free? No, it has C5.
    # The Petersen graph has C5.
    
    # Let's try: complement of a C5-free graph with alpha < 9
    # Or: use a strongly regular graph
    
    # For now, let's try to extend the K4 union
    # Add vertex 32, connect it to one vertex from each K4
    # This gives alpha: we can pick the unconnected vertices from each K4 (7 per K4? No)
    # If v32 is connected to one vertex in each K4, then from each K4 we can pick at most 3 vertices that are not v32's neighbors
    # But we need an independent set, so we pick vertices not adjacent to each other
    # Within a K4, all vertices are adjacent, so we can pick at most 1 from each K4
    # Plus v32 if it's not adjacent to the picked vertices
    # If v32 is adjacent to one vertex in each K4, and we pick a different vertex from each K4,
    # then v32 is not adjacent to any of the picked vertices
    # So alpha >= 8 (one from each K4) + possibly v32 = 9. Bad.
    
    # If v32 is connected to 2 vertices in each K4:
    # From each K4, we can pick at most 2 non-adjacent vertices? No, K4 is a clique.
    # We can pick at most 1 from each K4.
    # v32 is adjacent to 2 in each K4, so if we pick a vertex not adjacent to v32,
    # v32 can be added to the independent set
    # But we can only pick 1 from each K4, and that vertex must not be adjacent to v32
    # So alpha >= 8 + 1 = 9. Still bad.
    
    # If v32 is connected to 3 vertices in each K4:
    # From each K4, we can pick 1 vertex not adjacent to v32
    # v32 is not adjacent to those 8 vertices
    # So alpha >= 9. Bad.
    
    # If v32 is connected to 4 vertices in each K4 (i.e., all):
    # Then v32 is adjacent to all, so we can't include v32 in an independent set
    # alpha = 8 (one from each K4). Good!
    # But does this create a C5?
    # v32 is connected to all vertices in all K4s
    # A C5 would need 5 vertices in a cycle
    # Since each K4 is a clique, any 2 vertices in the same K4 are adjacent
    # A C5 cannot have 2 adjacent vertices from the same K4 unless they're consecutive
    # But if we have v32, v1, v2, v3, v4 in a cycle, and v1,v2 in same K4, then v1-v2 is an edge
    # v32-v1, v1-v2, v2-v3, v3-v4, v4-v32
    # v1 and v2 are in same K4, so edge exists
    # v3 and v4 could be in same or different K4s
    # This is getting complex. Let's just check computationally.
    
    def is_c5_free(edges, n):
        """Check if graph has no 5-cycle"""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all 5-tuples (a,b,c,d,e) with a<b<c<d<e
        # and check if a-b-c-d-e-a is a cycle
        # This is O(n^5) which is too slow for n=33
        # Better: check all 5-cycles by looking at paths of length 4
        # that close into a cycle
        
        # For each pair (u,v) that is an edge, check if there's a path of length 4 from u to v
        # that doesn't use the edge (u,v)
        # This is still expensive
        
        # Alternative: use the fact that in a C5-free graph, the neighborhood of any vertex
        # is triangle-free
        # So for each vertex v, the subgraph induced by N(v) must be triangle-free
        # This is a necessary condition, not sufficient
        
        # Let's do a direct check: for each 5-set, check if it contains a C5
        # There are C(33,5) = 237336 five-sets, and for each we check if it has a C5
        # A 5-vertex graph has a C5 iff it's exactly C5 (since it's 5 vertices)
        # So we check if the induced subgraph on the 5 vertices is isomorphic to C5
        
        from itertools import combinations
        
        for combo in combinations(range(n), 5):
            # Get edges within this 5-set
            sub_edges = []
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if v in adj[u]:
                        sub_edges.append((i, j))
            
            # Check if this 5-vertex graph is C5
            # C5 has exactly 5 edges and is 2-regular
            if len(sub_edges) != 5:
                continue
            
            # Check 2-regularity
            deg = [0]*5
            for i, j in sub_edges:
                deg[i] += 1
                deg[j] += 1
            if any(d != 2 for d in deg):
                continue
            
            # It's a 2-regular graph on 5 vertices, so it's C5
            return False
        
        return True
    
    def alpha_less_than(edges, n, k=9):
        """Check if independence number is less than k"""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Use a simple branch and bound
        # Or use the fact that for n=33, we can try all subsets of size 8
        # and check if any is independent, then try size 9
        # C(33,9) is too large
        
        # Better: use a greedy + backtracking approach
        # Or use the Lovasz theta bound (but we need exact)
        
        # Let's use a simple recursive approach with pruning
        # This might be slow, but let's try for n=33
        
        # Actually, let's use a different approach:
        # The complement graph has clique number < 9
        # So we check if the complement has a clique of size 9
        
        # Build complement adjacency
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
        
        # Check for clique of size 9 in complement
        # Use Bron-Kerbosch or a simple backtracking
        
        def has_clique(size, candidates):
            if size == 0:
                return True
            if len(candidates) < size:
                return False
            if not candidates:
                return False
            
            v = candidates[0]
            # Include v
            new_candidates = [u for u in candidates[1:] if u in comp_adj[v]]
            if has_clique(size - 1, new_candidates):
                return True
            # Exclude v
            if has_clique(size, candidates[1:]):
                return True
            return False
        
        return not has_clique(9, list(range(n)))
    
    # Try to construct a valid graph with n=33
    # Approach: Start with 8 K4s (32 vertices), add vertex 32
    # Connect vertex 32 to all 32 other vertices
    # Then check if it's C5-free and alpha < 9
    
    if time.time() < deadline - 60:
        # Try: 8 K4s + universal vertex
        n_test = 33
        edges_test = []
        for b in range(8):
            start = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges_test.append([start + i, start + j])
        # Connect vertex 32 to all others
        for i in range(32):
            edges_test.append([32, i])
        
        # Check C5-free
        c5_free = is_c5_free(edges_test, n_test)
        if c5_free:
            # Check alpha < 9
            alpha_ok = alpha_less_than(edges_test, n_test, 9)
            if alpha_ok:
                best_result = {'n': n_test, 'edges': edges_test}
                return best_result
        
        # Try other structures
        # 7 K4s + 1 K5? K5 has C5, so no.
        # 6 K4s + 2 K5? No.
        
        # Try: complete multipartite graph
        # K_{4,4,4,4,4,4,4,4,1} - 9 parts, 8 of size 4, 1 of size 1
        # n = 33
        # This is C5-free? A complete multipartite graph is C5-free iff no 5 parts have edges between them
        # Actually, C5 in a complete multipartite graph: we need 5 vertices in 5 different parts
        # So if we have 5 or more parts, there's a C5
        # With 9 parts, there's definitely a C5
        
        # Try: blow-up of C5? No, that has C5.
        
        # Let's try random graphs with high minimum degree
        random.seed(42)
        for trial in range(100):
            if time.time() > deadline - 30:
                break
            
            n_test = 33
            # Generate a random graph with edge probability ~0.3
            edges_test = []
            for i in range(n_test):
                for j in range(i+1, n_test):
                    if random.random() < 0.3:
                        edges_test.append([i, j])
            
            # Check C5-free
            if not is_c5_free(edges_test, n_test):
                continue
            
            # Check alpha < 9
            if not alpha_less_than(edges_test, n_test, 9):
                continue
            
            best_result = {'n': n_test, 'edges': edges_test}
            return best_result
    
    return best_result
