"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Q3 (3-cube) has 8 vertices, 12 edges
    # We need to check if adding an edge creates a Q3 subgraph
    
    # Precompute all Q3 embeddings (sets of 8 vertices that form a Q3)
    # A Q3 is determined by choosing 8 vertices and checking if they induce a Q3
    # But for efficiency, we'll use a different approach:
    # When adding edge (u,v), check if there exists a Q3 containing this edge
    
    # A Q3 can be seen as two copies of Q2 (4-cycle) connected by 4 edges
    # Or: Q3 has vertices {0,1,2,3} x {0,1,2,3} no wait, it's {0,1}^3
    
    # Let's use a direct approach: for each pair of edges that could be opposite in Q3,
    # check if the remaining 6 vertices form the rest of the Q3
    
    # Actually, let's use a smarter approach:
    # Q3 is bipartite with parts of size 4. 
    # A Q3 subgraph is determined by its 8 vertices.
    # For a given edge (u,v) in Q3, u and v are at distance 1 in Q3.
    # The other 6 vertices must be specific neighbors.
    
    # Let me enumerate all possible Q3 patterns on 16 vertices
    # A Q3 embedding is an injective mapping from {0,1}^3 to V such that
    # (x,y) is an edge in Q3 iff x and y differ in exactly one coordinate
    
    # For efficiency, I'll precompute all "Q3 templates" - for each edge (u,v),
    # what other edges must exist for (u,v) to be part of a Q3?
    
    # Alternative approach: maintain adjacency sets and check Q3 existence incrementally
    
    # Let's use a greedy approach with backtracking
    # Start with empty graph, add edges one by one if they don't create a Q3
    
    n = 16
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    def has_q3_with_edge(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # In Q3, each vertex has degree 3.
        # If (u,v) is an edge in Q3, then in the Q3:
        # - u has 2 other neighbors in the Q3 (call them u1, u2)
        # - v has 2 other neighbors in the Q3 (call them v1, v2)
        # - u1, u2 are not adjacent to each other in Q3 (they're at distance 2 from u)
        # - Similarly for v1, v2
        # - u1 is adjacent to exactly one of v1, v2 (the one that differs in the same coordinate as u-v)
        # - Actually let me think more carefully
        
        # Q3 structure: vertices are binary triples (x,y,z)
        # Edge between vertices differing in exactly one bit
        # If (u,v) is an edge, say they differ in bit 0: u=(0,a,b), v=(1,a,b)
        # Then u's other Q3-neighbors: (0,a,1-b) and (0,1-a,b)
        # v's other Q3-neighbors: (1,a,1-b) and (1,1-a,b)
        # The remaining edges in Q3:
        # (0,a,1-b)-(1,a,1-b), (0,1-a,b)-(1,1-a,b)
        # (0,a,b)-(0,1-a,b) is u to u's neighbor
        # etc.
        
        # So if (u,v) is a Q3 edge, there must exist 6 other vertices a,b,c,d,e,f such that:
        # - a,b are neighbors of u (in Q3), a!=b
        # - c,d are neighbors of v (in Q3), c!=d  
        # - a is adjacent to exactly one of {c,d}, say c
        # - b is adjacent to exactly one of {c,d}, say d
        # - a is NOT adjacent to d
        # - b is NOT adjacent to c
        # - c and d are not adjacent
        # - a and b are not adjacent
        # - e = the vertex adjacent to both a and c (and not u or v)... 
        
        # Let me just check all possibilities directly
        # For (u,v) to be in a Q3, we need to find 6 other vertices that complete the Q3
        
        # The Q3 containing edge (u,v) is uniquely determined up to the choice of
        # which "direction" the other edges go. But since we're checking existence,
        # let's enumerate:
        
        # u's neighbors in current graph (excluding v)
        u_neighbors = adj[u] - {v}
        v_neighbors = adj[v] - {u}
        
        # For each pair (a,b) of u's neighbors and (c,d) of v's neighbors,
        # check if they form a valid Q3 with u,v
        for a in u_neighbors:
            for b in u_neighbors:
                if a == b:
                    continue
                # Check a and b are not adjacent (in Q3, two neighbors of same vertex aren't adjacent)
                if b in adj[a]:
                    continue
                for c in v_neighbors:
                    for d in v_neighbors:
                        if c == d:
                            continue
                        if d in adj[c]:
                            continue
                        # Check the cross edges
                        # In Q3: a is adjacent to exactly one of c,d
                        # b is adjacent to exactly one of c,d
                        # And they're "opposite": if a-c exists, then b-d exists, not b-c
                        # Also a-d should NOT exist, b-c should NOT exist
                        
                        a_c = c in adj[a]
                        a_d = d in adj[a]
                        b_c = c in adj[b]
                        b_d = d in adj[b]
                        
                        # Pattern 1: a-c and b-d exist, a-d and b-c don't
                        if a_c and b_d and not a_d and not b_c:
                            # Check c and d are not adjacent
                            if d in adj[c]:
                                continue
                            # Check the remaining 2 edges:
                            # The vertex adjacent to a and c (other than u and v) - call it e
                            # In Q3, e is adjacent to a, c, and one more vertex
                            # Actually, in Q3 with vertices {u,v,a,b,c,d,e,f}:
                            # u-v, u-a, u-b, v-c, v-d
                            # a-c, b-d (as above)
                            # e is adjacent to a, c, and... let's see
                            # If u=(0,0,0), v=(1,0,0), a=(0,1,0), b=(0,0,1)
                            # c=(1,1,0), d=(1,0,1)
                            # e=(0,1,1) is adjacent to a=(0,1,0), b=(0,0,1), and... (1,1,1)
                            # f=(1,1,1) is adjacent to c=(1,1,0), d=(1,0,1), and e=(0,1,1)
                            # So e-a, e-b, e-f, f-c, f-d, f-e
                            # We need to check if there exist e,f such that:
                            # e is adjacent to a, b, f
                            # f is adjacent to c, d, e
                            # e and f are adjacent
                            # e not equal to any of u,v,a,b,c,d
                            # f not equal to any of u,v,a,b,c,d
                            
                            used = {u, v, a, b, c, d}
                            candidates_e = set()
                            for x in range(n):
                                if x not in used:
                                    if a in adj[x] and b in adj[x]:
                                        candidates_e.add(x)
                            
                            for e in candidates_e:
                                candidates_f = set()
                                for x in range(n):
                                    if x not in used and x != e:
                                        if c in adj[x] and d in adj[x]:
                                            candidates_f.add(x)
                                for f in candidates_f:
                                    if e in adj[f]:
                                        return True
        
        # Pattern 2: a-d and b-c exist, a-c and b-d don't
        for a in u_neighbors:
            for b in u_neighbors:
                if a == b:
                    continue
                if b in adj[a]:
                    continue
                for c in v_neighbors:
                    for d in v_neighbors:
                        if c == d:
                            continue
                        if d in adj[c]:
                            continue
                        a_c = c in adj[a]
                        a_d = d in adj[a]
                        b_c = c in adj[b]
                        b_d = d in adj[b]
                        
                        if a_d and b_c and not a_c and not b_d:
                            if d in adj[c]:
                                continue
                            used = {u, v, a, b, c, d}
                            candidates_e = set()
                            for x in range(n):
                                if x not in used:
                                    if a in adj[x] and b in adj[x]:
                                        candidates_e.add(x)
                            for e in candidates_e:
                                candidates_f = set()
                                for x in range(n):
                                    if x not in used and x != e:
                                        if c in adj[x] and d in adj[x]:
                                            candidates_f.add(x)
                                for f in candidates_f:
                                    if e in adj[f]:
                                        return True
        return False
    
    # Greedy: try to add edges in some order
    # Try multiple random orderings
    best_edges = []
    best_count = 0
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        random.seed(trial * 12345 + 67890)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        current_edges = []
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u,v) creates a Q3
            if not has_q3_with_edge(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_edges.append([u, v])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    return best_edges

# EVOLVE-BLOCK-END
