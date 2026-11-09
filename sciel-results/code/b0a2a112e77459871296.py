import time
import itertools
import numpy as np

def construct():
    """
    Construct a Q3-free graph on 16 vertices with maximum edges.
    Strategy: Greedy addition of edges with Q3-subgraph check.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Helper to check if adding edge (u, v) creates a Q3 subgraph
    # A Q3 subgraph exists if there is a set of 8 vertices forming a 3-cube.
    # Instead of checking all subsets (expensive), we can use the fact that 
    # if we add (u,v), any Q3 containing (u,v) must have u and v at distance 1 in the cube.
    # In Q3, vertices at distance 1 are connected. 
    # This check is complex. A simpler heuristic for greedy:
    # We check if the current graph + (u,v) contains Q3.
    # To speed up, we only check if (u,v) is part of a potential Q3.
    
    # A faster check: 
    # A graph contains Q3 if it contains a cycle of length 8 that is "cube-like"? No.
    # Let's use a specific check: 
    # For any pair (u,v) to be in a Q3, there must be 6 other vertices.
    # This is still O(N^8) in worst case if done naively.
    
    # Alternative Strategy: 
    # The seed is K3,13 (39 edges).
    # We can try to improve by starting with a dense random graph or a known dense Q3-free graph 
    # and removing edges if Q3 is present.
    # Or, use the fact that the complement of a Q3-free graph is a graph where no Q3 is missing?
    
    # Let's stick to the insight from inspiration 1: 
    # "Correcting the subgraph check from induced-only to general subgraph (via bipartite matching on missing edges) prevents adding edges that create Q3 subgraphs"
    
    # How to check if G + (u,v) contains Q3?
    # Q3 is bipartite with parts of size 4.
    # If we add (u,v), we need to see if there is a Q3 containing this edge.
    # In Q3, each edge belongs to exactly 2 squares (4-cycles).
    # And each square belongs to 2 cubes? No, in Q3, each edge is in 2 faces.
    
    # Let's implement a check: does G contain a Q3?
    # We can iterate over all pairs of vertices (u, v) that are NOT adjacent in G (missing edges).
    # If G is Q3-free, then for any Q3 homomorphism... no.
    
    # Let's use a different approach: 
    # Start with K16. Remove edges until Q3-free.
    # This is likely too slow.
    
    # Let's try the greedy addition with a fast check.
    # Check: Does G contain Q3?
    # We can precompute all Q3 subgraphs? No, 16 choose 8 is large.
    
    # Let's use the bipartite matching idea from the lesson.
    # A graph G contains Q3 iff there exist two sets of 4 vertices A, B such that:
    # 1. A and B are independent sets in G? No, Q3 is bipartite, so A and B are the bipartition.
    #    So no edges within A, no edges within B.
    # 2. The edges between A and B form a Q3.
    #    Q3 edges: Each vertex in A has degree 3 in the subgraph. Each vertex in B has degree 3.
    #    And the structure is specific.
    
    # Actually, a simpler check for "contains Q3" when adding an edge (u,v):
    # If G is Q3-free, and we add (u,v), we only need to check if (u,v) is part of a Q3 in G+(u,v).
    # If (u,v) is in a Q3, then there are 2 other edges incident to u in the Q3, say (u,a) and (u,b).
    # And 2 other edges incident to v, say (v,c) and (v,d).
    # This branching is small.
    
    def contains_q3_with_edge(adj, u, v):
        """Check if adding (u,v) to adj creates a Q3 containing (u,v)."""
        # In Q3, u and v are connected.
        # Neighbors of u in Q3: v, a, b.
        # Neighbors of v in Q3: u, c, d.
        # a and b are not connected to v (distance 2).
        # c and d are not connected to u (distance 2).
        # a and c? In Q3, if u-v is an edge, and u-a, v-c are edges.
        # If a and c are adjacent, then u-v-c-a-u is a 4-cycle.
        # Q3 has 6 squares.
        
        # Let's find candidates for a, b (neighbors of u in G, excluding v)
        # and c, d (neighbors of v in G, excluding u)
        
        # Note: u and v are not adjacent in current adj (we are checking if we CAN add them)
        # So we look for neighbors in current adj.
        
        nbrs_u = [x for x in range(n) if adj[u, x] == 1 and x != v]
        nbrs_v = [x for x in range(n) if adj[v, x] == 1 and x != u]
        
        # We need to pick 2 from nbrs_u and 2 from nbrs_v.
        # If len(nbrs_u) < 2 or len(nbrs_v) < 2, no Q3 can be formed with (u,v) as an edge.
        if len(nbrs_u) < 2 or len(nbrs_v) < 2:
            return False
        
        for a, b in itertools.combinations(nbrs_u, 2):
            for c, d in itertools.combinations(nbrs_v, 2):
                # Check if {u, v, a, b, c, d, e, f} forms a Q3?
                # We have u, v, a, b, c, d. We need 2 more vertices e, f.
                # In Q3:
                # u is connected to v, a, b.
                # v is connected to u, c, d.
                # a is connected to u, and two others. One is likely c or d?
                # Let's trace the cube structure.
                # Let u=0, v=1.
                # a=2, b=4 (arbitrary labels for neighbors of u).
                # c=3, d=5 (arbitrary labels for neighbors of v).
                # In Q3:
                # 0-1, 0-2, 0-4.
                # 1-3, 1-5.
                # 2 is connected to 0, 6, 7.
                # 4 is connected to 0, 6, 7.
                # So 2 and 4 share neighbors 6, 7.
                # 3 is connected to 1, 6, 7.
                # 5 is connected to 1, 6, 7.
                # So 3 and 5 share neighbors 6, 7.
                # And 2 is connected to 3? No, 2-6-3 is path length 2.
                # 2 and 3 are not adjacent.
                # 2 and 5 are not adjacent.
                # 4 and 3 are not adjacent.
                # 4 and 5 are not adjacent.
                
                # So we need to find e, f such that:
                # e, f are adjacent to a, b, c, d appropriately.
                # Specifically:
                # a (2) is adjacent to e, f.
                # b (4) is adjacent to e, f.
                # c (3) is adjacent to e, f.
                # d (5) is adjacent to e, f.
                # And e-f is an edge? In Q3, 6-7 is an edge.
                # So e and f must be adjacent.
                # And e, f must be distinct from u, v, a, b, c, d.
                
                # Candidates for e, f:
                # Must be neighbors of a, b, c, d in G.
                # common_ab = set(neighbors of a) & set(neighbors of b)
                # common_cd = set(neighbors of c) & set(neighbors of d)
                # e, f must be in common_ab AND common_cd.
                # And e, f must be adjacent in G.
                
                common_ab = set(x for x in range(n) if adj[a, x] == 1) & set(x for x in range(n) if adj[b, x] == 1)
                common_cd = set(x for x in range(n) if adj[c, x] == 1) & set(x for x in range(n) if adj[d, x] == 1)
                
                candidates = common_ab & common_cd
                # Exclude u, v, a, b, c, d
                candidates -= {u, v, a, b, c, d}
                
                cand_list = list(candidates)
                for e, f in itertools.combinations(cand_list, 2):
                    if adj[e, f] == 1:
                        # Check remaining edges of Q3:
                        # a-e, a-f, b-e, b-f, c-e, c-f, d-e, d-f
                        # We already know a,b,c,d are neighbors of e,f? 
                        # e,f in common_ab means a,b adj to e,f.
                        # e,f in common_cd means c,d adj to e,f.
                        # So yes.
                        # Also need to ensure no extra edges that break the "simple graph" structure?
                        # The definition is "subgraph isomorphic to Q3". 
                        # It doesn't require induced. So extra edges are fine.
                        # We just need the 12 edges of Q3 to be present.
                        return True
        return False

    # Greedy construction
    # Order edges by some heuristic. 
    # Random order or sorted by degree?
    # Let's try random restarts with different edge orderings.
    
    best_edges = 0
    best_graph = None
    best_adj = None
    
    # We will try a few random permutations of edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle edges
        edges_to_try = all_edges.copy()
        np.random.shuffle(edges_to_try)
        
        adj = np.zeros((n, n), dtype=np.int8)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
                
            # Check if adding (u,v) creates Q3
            if not contains_q3_with_edge(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
        
        current_edges = np.sum(adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u, v] == 1]
            best_adj = adj.copy()
            
        # If we hit a high score, we can stop or try to optimize further
        if best_edges >= 45: # Just a guess, K3,13 is 39.
             pass

    # Fallback to seed if nothing better found
    if best_graph is None or best_edges <= 39:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_graph

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    g = construct()
    print(f"Found graph with {len(g)} edges")
