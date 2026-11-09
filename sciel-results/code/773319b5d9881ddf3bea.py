import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill
    
    n = 16
    # All possible edges in K_16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_edges = len(all_edges)  # 120
    
    # We want to find the maximum number of edges in a Q3-free graph.
    # The seed program returns K_{3,13} which has 3*13 = 39 edges.
    # We can do better.
    
    # Strategy: Start with a dense graph and remove edges to break all Q3 subgraphs.
    # A Q3 (3-cube) has 8 vertices and 12 edges.
    # 
    # Key insight: A graph is Q3-free if and only if it does not contain a subgraph
    # isomorphic to Q3. Q3 is bipartite with parts of size 4 each.
    #
    # Approach: Use a randomized greedy algorithm with backtracking, or use a
    # constraint satisfaction approach.
    #
    # Better approach: Start with K_16 and try to remove as few edges as possible
    # while eliminating all Q3 subgraphs. This is equivalent to finding a minimum
    # edge deletion set that hits all Q3 subgraphs.
    #
    # Alternative: Build the graph edge by edge, adding an edge only if it doesn't
    # create a Q3. This gives a maximal Q3-free graph (not necessarily maximum).
    #
    # Let's try a hybrid approach:
    # 1. Use a randomized greedy construction to build a dense Q3-free graph.
    # 2. Try to improve by swapping edges or adding more.
    
    # First, let's implement a fast check for whether adding an edge creates a Q3.
    
    # Precompute: For each pair of vertices, check if adding edge (u,v) to current graph
    # creates a Q3. This requires checking if there exists a Q3 containing edge (u,v).
    
    # A Q3 can be identified by its 8 vertices. The 3-cube Q3 has vertices that can be
    # labeled as binary 3-tuples. Two vertices are adjacent if they differ in exactly one bit.
    
    # To check if a set of 8 vertices forms a Q3:
    # - It must be bipartite with parts of size 4
    # - Each vertex has degree 3 within the induced subgraph
    # - The graph is connected
    
    # For efficiency, let's use a different approach:
    # A Q3 is determined by choosing 4 vertices for each part (say left and right),
    # where each left vertex is connected to exactly 3 right vertices, and vice versa,
    # and the overall structure is the cube.
    
    # Actually, let's use a simpler characterization:
    # Q3 is the 3-dimensional hypercube. It has 8 vertices, 12 edges.
    # The complement of Q3 in K_8 has 28-12=16 edges.
    
    # Let me think about this differently. Let's just do a randomized search:
    # 1. Start with a random ordering of edges.
    # 2. Greedily add edges if they don't create a Q3.
    # 3. Use multiple random seeds and keep the best result.
    
    def is_q3_free(adj):
        """Check if graph with adjacency matrix adj is Q3-free."""
        # Check all subsets of 8 vertices
        # This is too slow for 16 vertices: C(16,8) = 12870, and for each we check if it's Q3
        # But we can be smarter.
        
        # A faster approach: For each set of 4 vertices, check if they can be the "left"
        # part of a Q3.
        
        # Actually, let's use the fact that Q3 has a specific structure.
        # Two vertices in Q3 are at distance 0, 1, 2, or 3.
        # There are: 1 vertex at distance 0, 3 at distance 1, 3 at distance 2, 1 at distance 3.
        
        # Let's just check all 8-vertex subsets. C(16,8) = 12870.
        # For each, check if the induced subgraph is isomorphic to Q3.
        
        from itertools import combinations
        
        # Precompute Q3 adjacency
        # Q3 vertices: 0,1,2,3,4,5,6,7 (binary 000 to 111)
        # Edge if Hamming distance = 1
        q3_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    q3_edges.add((i, j))
        
        # For an 8-vertex induced subgraph to be Q3, it must have exactly 12 edges
        # and the adjacency pattern must match Q3 (up to isomorphism).
        
        # A quick necessary condition: exactly 12 edges.
        # Then check if it's isomorphic to Q3.
        
        # For speed, let's first check edge count, then do a more careful check.
        
        for subset in combinations(range(n), 8):
            # Count edges in induced subgraph
            edge_count = 0
            for i_idx in range(8):
                for j_idx in range(i_idx+1, 8):
                    u = subset[i_idx]
                    v = subset[j_idx]
                    if adj[u][v]:
                        edge_count += 1
            
            if edge_count != 12:
                continue
            
            # Check if this 8-vertex graph is isomorphic to Q3
            # Q3 is 3-regular, so check if all vertices have degree 3
            degrees = [0]*8
            for i_idx in range(8):
                for j_idx in range(8):
                    if i_idx != j_idx:
                        u = subset[i_idx]
                        v = subset[j_idx]
                        if adj[u][v]:
                            degrees[i_idx] += 1
            
            if any(d != 3 for d in degrees):
                continue
            
            # Now check if it's actually Q3 and not some other 3-regular graph on 8 vertices.
            # The only 3-regular graphs on 8 vertices are Q3 and the complete bipartite K_{4,4} minus a perfect matching? No.
            # Actually, there are multiple 3-regular graphs on 8 vertices.
            # Q3 is bipartite. Check bipartiteness.
            
            # Check if bipartite
            color = [-1]*8
            color[0] = 0
            queue = [0]
            is_bipartite = True
            while queue:
                u = queue.pop(0)
                for v in range(8):
                    if u != v and adj[subset[u]][subset[v]]:
                        if color[v] == -1:
                            color[v] = 1 - color[u]
                            queue.append(v)
                        elif color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
            
            if not is_bipartite:
                continue
            
            # If it's 3-regular, bipartite, and has 8 vertices, it must be Q3
            # (since the only 3-regular bipartite graph on 8 vertices with 4+4 split is Q3)
            # Wait, is that true? Let me verify.
            # A 3-regular bipartite graph with parts of size 4 each has 4*3=12 edges.
            # Q3 is one such graph. Are there others?
            # The complete bipartite graph K_{4,4} is 4-regular.
            # A 3-regular bipartite graph on 4+4 vertices: each left vertex connects to 3 of 4 right vertices.
            # This is equivalent to a 3-regular bipartite multigraph... no, simple graph.
            # Number of such graphs: it's related to the number of 4x4 (0,1)-matrices with row and column sums 3.
            # This is the number of 4x4 binary matrices with all row/col sums 3, which equals the number of 4x4 binary matrices with all row/col sums 1 (by complement), which is 4! = 24.
            # But many of these are isomorphic. Up to isomorphism, there might be just one: Q3.
            # Actually, I believe all 3-regular bipartite graphs on 4+4 vertices are isomorphic to Q3.
            # Let me verify: Q3 is the only connected 3-regular bipartite graph on 8 vertices with parts of size 4.
            # Yes, I'm fairly confident this is true.
            
            return False  # Found a Q3
        
        return True  # No Q3 found
    
    # This is_q3_free is too slow. Let's use a different approach.
    
    # Better approach: Use the fact that we can check incrementally.
    # When adding edge (u,v), we only need to check if it creates a NEW Q3.
    # A new Q3 must contain the edge (u,v).
    
    # How many Q3s contain a given edge? 
    # In Q3, each edge is in exactly C(6,0) = 1... no.
    # Actually, let's count: fix an edge. The remaining 6 vertices must complete the Q3.
    # This is getting complicated. Let's just use a timeout-based approach with the full check.
    
    # Alternative: Use a known result or construction.
    # The ex(16, Q3) problem. Let's try to find a good construction.
    
    # Known: Q3 is bipartite. By Turan-type results, we might get a lower bound.
    # But let's just do a practical search.
    
    # Approach: 
    # 1. Generate a random permutation of edges.
    # 2. Greedily add edges if the graph remains Q3-free.
    # 3. Repeat with different random seeds, keep the best.
    
    # To make the Q3 check fast, let's precompute something.
    
    # Actually, let's use a different strategy:
    # Start with a dense bipartite graph. Q3 is bipartite, so if our graph is bipartite
    # with parts of size 8 and 8, and we use a random bipartite graph, it might be Q3-free.
    
    # Even better: Use a random 7-regular bipartite graph on 8+8 vertices.
    # Or use a structured construction.
    
    # Let me try: Take a complete bipartite graph K_{8,8} and remove a random regular graph.
    
    # Actually, let's try a simpler approach:
    # Use a random graph G(n, p) and check if it's Q3-free, adjusting p.
    
    # For n=16, let's try p=0.7, 0.6, 0.5, etc.
    
    best_edges = []
    best_count = 0
    
    # Seed the initial result from the seed program
    initial_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_edges = initial_edges
    best_count = len(initial_edges)  # 39
    
    # Try randomized greedy construction
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        
        # Random edge order
        edges = all_edges[:]
        random.shuffle(edges)
        
        # Build adjacency matrix
        adj = [[False]*n for _ in range(n)]
        current_edges = []
        
        for (u, v) in edges:
            if time.time() > deadline - 5:
                break
            
            # Try adding edge (u, v)
            adj[u][v] = adj[v][u] = True
            current_edges.append((u, v))
            
            # Check if this creates a Q3
            # Fast check: only check Q3s that contain edge (u,v)
            # A Q3 containing (u,v): the other 6 vertices must form the rest of the cube.
            
            # For efficiency, let's do a targeted check.
            # In Q3, given an edge (u,v), the remaining structure is determined.
            # u and v are adjacent in Q3. The Q3 has 8 vertices.
            # Let's enumerate possible Q3s containing (u,v).
            
            # Actually, let's just do the full check but with early termination.
            # Since we're adding edges one at a time, we can check if the new edge
            # is part of any Q3.
            
            # A Q3 containing edge (u,v) must have u and v as adjacent vertices.
            # In Q3, fix u=000, v=001. The other 6 vertices are 010,011,100,101,110,111.
            # The edges are: 000-010, 000-100, 001-011, 001-101, 010-011, 010-110,
            # 100-101, 100-110, 110-111, 101-111, 011-111, and 000-001 (the fixed edge).
            
            # So given (u,v), we need to find 6 other vertices such that:
            # - 3 vertices adjacent to u (besides v)
            # - 3 vertices adjacent to v (besides u)
            # - The remaining edges form the cube structure.
            
            # This is complex. Let's just use the full is_q3_free check but optimize it.
            
            if not is_q3_free_fast(adj, u, v):
                # Remove the edge
                adj[u][v] = adj[v][u] = False
                current_edges.pop()
                continue
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [[u, v] for (u, v) in current_edges]
    
    return best_edges


def is_q3_free_fast(adj, u, v):
    """Check if adding edge (u,v) to the graph creates a Q3.
    Only checks Q3s that contain edge (u,v)."""
    n = 16
    
    # A Q3 containing edge (u,v): we need to find 6 more vertices.
    # In Q3, u and v are at distance 1. 
    # Let's enumerate all ways to complete the cube.
    
    # The neighbors of u in Q3 (besides v): 3 vertices
    # The neighbors of v in Q3 (besides u): 3 vertices
    # These 6 vertices must be distinct and not include u or v.
    
    # Let's find candidates:
    # Candidates for neighbors of u (besides v): vertices w such that adj[u][w] is True
    # Candidates for neighbors of v (besides u): vertices w such that adj[v][w] is True
    
    neigh_u = [w for w in range(n) if w != v and adj[u][w]]
    neigh_v = [w for w in range(n) if w != u and adj[v][w]]
    
    # We need to choose 3 from neigh_u and 3 from neigh_v, all distinct,
    # and check if they form a Q3 with u and v.
    
    from itertools import combinations
    
    # For each choice of 3 neighbors of u and 3 neighbors of v:
    for combo_u in combinations(neigh_u, 3):
        for combo_v in combinations(neigh_v, 3):
            # Check that all 6 are distinct
            if len(set(combo_u) | set(combo_v)) != 6:
                continue
            
            # Now check if {u, v} ∪ combo_u ∪ combo_v forms a Q3
            # with u and v adjacent.
            
            # In Q3, the structure is:
            # u is connected to v and to the 3 in combo_u.
            # v is connected to u and to the 3 in combo_v.
            # Each vertex in combo_u is connected to u and to 2 vertices in combo_v.
            # Each vertex in combo_v is connected to v and to 2 vertices in combo_u.
            
            # Let's label: combo_u = (a, b, c), combo_v = (d, e, f)
            # In Q3, if u=000, v=001, then:
            # combo_u corresponds to 010, 100, and one more? No.
            # Let me re-derive.
            
            # Q3 vertices: 000, 001, 010, 011, 100, 101, 110, 111
            # Edges (Hamming distance 1):
            # 000-001, 000-010, 000-100
            # 001-011, 001-101
            # 010-011, 010-110
            # 100-101, 100-110
            # 011-111
            # 101-111
            # 110-111
            
            # So if u=000, v=001:
            # Neighbors of u (besides v): 010, 100, and... wait, 000 is only connected to 001, 010, 100.
            # So combo_u = {010, 100, ?} - but 000 only has 3 neighbors total, one is v=001.
            # So combo_u has 2 elements: 010 and 100.
            
            # I made an error. In Q3, each vertex has degree 3.
            # u=000 has neighbors: 001, 010, 100.
            # v=001 has neighbors: 000, 011, 101.
            
            # So given edge (u,v) where u=000, v=001:
            # combo_u (neighbors of u besides v) = {010, 100} -- only 2 vertices!
            # combo_v (neighbors of v besides u) = {011, 101} -- only 2 vertices!
            
            # And the remaining 2 vertices are 110 and 111.
            
            # So the structure is:
            # u connected to: v, a, b (where a,b are 2 vertices)
            # v connected to: u, c, d (where c,d are 2 vertices)
            # a connected to: u, and 2 of {c,d,e,f}
            # b connected to: u, and 2 of {c,d,e,f}
            # c connected to: v, and 2 of {a,b,e,f}
            # d connected to: v, and 2 of {a,b,e,f}
            # e and f are the remaining 2 vertices.
            
            # Let me re-examine. Q3 has 8 vertices, each of degree 3.
            # Fix u and v as an edge.
            # u has 2 other neighbors: call them a, b.
            # v has 2 other neighbors: call them c, d.
            # The remaining 2 vertices: e, f.
            
            # In Q3 with u=000, v=001:
            # a=010, b=100, c=011, d=101, e=110, f=111
            # Edges:
            # u-a, u-b, u-v
            # v-c, v-d, v-u
            # a-c, a-e
            # b-d, b-e
            # c-f
            # d-f
            # e-f
            
            # So the required edges (besides u-v) are:
            # u-a, u-b, v-c, v-d, a-c, a-e, b-d, b-e, c-f, d-f, e-f
            
            # Total: 11 edges + u-v = 12 edges. Correct.
            
            # So to check if a Q3 exists containing (u,v):
            # Choose a, b from neighbors of u (besides v): C(k_u, 2) where k_u = |neigh_u|
            # Choose c, d from neighbors of v (besides u): C(k_v, 2) where k_v = |neigh_v|
            # The remaining 2 vertices e, f are determined? No, they can be any 2 from the remaining vertices.
            # But they must satisfy the edge constraints.
            
            # Actually, e and f must be specific vertices. Let's think again.
            # Given u, v, a, b, c, d, the vertices e and f are the remaining 2 from the 8-vertex set.
            # But we don't know the 8-vertex set yet.
            
            # Let's just enumerate: for each choice of a,b from neigh_u and c,d from neigh_v,
            # with all 6 distinct, check if there exist e,f (from remaining vertices) such that
            # all required edges are present.
            
            a, b = combo_u  # But we need 2, not 3!
            
            # I see the issue. Let me redo with correct counts.
            pass
    
    # Let me rewrite this function properly.
    return True  # Placeholder, will fix


def is_q3_free_fast(adj, u, v):
    """Check if the graph contains a Q3 that uses edge (u,v)."""
    n = 16
    
    # In Q3, each vertex has degree 3.
    # Given edge (u,v), u has 2 other neighbors in Q3, v has 2 other neighbors.
    
    neigh_u = [w for w in range(n) if w != v and adj[u][w]]
    neigh_v = [w for w in range(n) if w != u and adj[v][w]]
    
    from itertools import combinations
    
    # Choose 2 from neigh_u and 2 from neigh_v
    for (a, b) in combinations(neigh_u, 2):
        for (c, d) in combinations(neigh_v, 2):
            # All 6 must be distinct
            if len({a, b, c, d}) != 4:
                continue
            
            # Now we need to find e, f such that the Q3 structure is satisfied.
            # Required edges:
            # u-a, u-b (already true by construction)
            # v-c, v-d (already true by construction)
            # a-c, a-e
            # b-d, b-e
            # c-f
            # d-f
            # e-f
            
            # So e must be adjacent to a and b.
            # f must be adjacent to c and d.
            # e and f must be adjacent to each other.
            # e and f must be distinct from u, v, a, b, c, d.
            
            # Find candidates for e: vertices adjacent to both a and b
            cand_e = [w for w in range(n) if w not in {u, v, a, b, c, d} and adj[a][w] and adj[b][w]]
            
            # Find candidates for f: vertices adjacent to both c and d
            cand_f = [w for w in range(n) if w not in {u, v, a, b, c, d} and adj[c][w] and adj[d][w]]
            
            for e in cand_e:
                for f in cand_f:
                    if e == f:
                        continue
                    # Check e-f edge
                    if adj[e][f]:
                        # All conditions satisfied! Found a Q3.
                        return False
    
    return True
# EVOLVE-BLOCK-END
