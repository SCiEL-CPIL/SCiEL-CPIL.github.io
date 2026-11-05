import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with K4,12
    L = list(range(4))
    R = list(range(4, 16))
    edges = [[u, v] for u in L for v in R]
    
    # Helper to check if a specific set of 8 vertices contains Q3
    # And to find a Q3 in the current graph
    
    # We represent the graph as a set of frozensets for fast lookup
    edge_set = set(frozenset(e) for e in edges)
    
    def has_q3():
        # Iterate over all subsets of 8 vertices
        # This is C(16,8) = 12870 subsets.
        # For each subset, check if it contains Q3.
        # Checking if a graph on 8 vertices contains Q3 is fast.
        
        # Optimization: Only check subsets that could contain Q3.
        # Q3 is bipartite 4+4.
        # So the 8 vertices must be partitionable into two sets of 4 with no internal edges?
        # No, the graph G is not necessarily bipartite.
        # But Q3 is bipartite.
        # So the 8 vertices must induce a bipartite subgraph?
        # Not necessarily, but the Q3 subgraph must be bipartite.
        
        # Simple check:
        # Iterate all 8-subsets.
        # For each, build the induced subgraph.
        # Check if Q3 is a subgraph.
        
        import itertools
        
        vertices = list(range(16))
        for subset in itertools.combinations(vertices, 8):
            # Check if induced subgraph on subset contains Q3
            # Map subset vertices to 0..7
            # Check if Q3 is a subgraph
            
            # Fast check: Count edges? Q3 has 12 edges.
            # If induced subgraph has < 12 edges, skip.
            sub_edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if frozenset((subset[i], subset[j])) in edge_set:
                        sub_edges.append((i, j))
            
            if len(sub_edges) < 12:
                continue
                
            # Check if Q3 is a subgraph of this 8-vertex graph
            # Q3 structure:
            # 0-1, 0-2, 0-4, 1-3, 2-3, 4-5, 4-6, 5-7, 6-7, 1-5, 2-6, 3-7
            # (Standard Q3 edges)
            
            q3_edges = [(0,1), (0,2), (0,4), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (1,5), (2,6), (3,7)]
            
            # Backtracking search for embedding
            # Or simply check if there exists a bijection f: 0..7 -> 0..7
            # such that for all (u,v) in q3_edges, (f(u), f(v)) in sub_edges
            
            # Since 8! is 40320, and we do this for 12870 subsets,
            # total operations ~ 500 million. Might be too slow in Python.
            
            # Optimization:
            # Use the fact that Q3 is 3-regular.
            # The degrees in the induced subgraph must be >= 3 for the vertices in the Q3?
            # No, the induced subgraph can have higher degrees.
            
            # Better: Just return the first Q3 found.
            
            # We can speed up the subgraph check by using bitmasks or numpy.
            # But let's try a simpler heuristic first.
            
            # If we find a Q3, return it.
            # We need to find a Q3 to remove an edge.
            
            # Let's try to find a Q3 by picking 4 vertices from L and 4 from R.
            # Since G starts as K4,12, any 4 from L and 4 from R form a K4,4.
            # K4,4 contains Q3.
            # So we can just pick any 4 from R.
            # And find a Q3 in that K4,4.
            
            # This is much faster!
            
            pass # Logic inside construct
            
        return None

    # Greedy Removal
    # We know G contains Q3.
    # Find one.
    # Remove an edge.
    # Repeat.
    
    # Since G is a subgraph of K4,12, any Q3 must use all 4 vertices of L.
    # So we only need to check subsets of R of size 4.
    # There are C(12,4) = 495 such subsets.
    # For each, the induced subgraph is a subgraph of K4,4.
    # We can check if it contains Q3.
    
    def find_q3():
        import itertools
        R_vertices = list(range(4, 16))
        for R_sub in itertools.combinations(R_vertices, 4):
            # Induced subgraph on L U R_sub
            # L is 0,1,2,3
            # R_sub is 4 vertices
            
            # Check if this 8-vertex graph contains Q3
            # We can hardcode the Q3 check for K4,4 subgraphs.
            
            # If it contains Q3, return the edges of that Q3.
            
            # For simplicity, assume it contains Q3 if it has enough edges.
            # Actually, we can just pick a specific Q3 embedding in K4,4
            # and check if its edges are in G.
            
            # Embedding of Q3 in K4,4 (L={0,1,2,3}, R=R_sub):
            # L: 0,1,2,3
            # R: r0,r1,r2,r3
            
            # Q3 edges in this embedding:
            # 0-r0, 0-r1, 0-r2 (Wait, degree of 0 in Q3 is 3)
            # Let's use a standard Q3 embedding.
            # L vertices: 0,1,2,3
            # R vertices: 4,5,6,7 (mapped to R_sub)
            
            # Edges:
            # 0-4, 0-5, 0-6
            # 1-4, 1-5, 1-7
            # 2-4, 2-6, 2-7
            # 3-5, 3-6, 3-7
            
            # Let's verify this is Q3.
            # Degrees:
            # 0: 4,5,6 (3)
            # 1: 4,5,7 (3)
            # 2: 4,6,7 (3)
            # 3: 5,6,7 (3)
            # 4: 0,1,2 (3)
            # 5: 0,1,3 (3)
            # 6: 0,2,3 (3)
            # 7: 1,2,3 (3)
            # All degrees 3. Bipartite. Connected.
            # Yes, this is Q3.
            
            r0, r1, r2, r3 = R_sub
            candidate_edges = [
                (0, r0), (0, r1), (0, r2),
                (1, r0), (1, r1), (1, r3),
                (2, r0), (2, r2), (2, r3),
                (3, r1), (3, r2), (3, r3)
            ]
            
            # Check if all these edges are in G
            if all(frozenset(e) in edge_set for e in candidate_edges):
                return candidate_edges
                
        return None

    # Main loop
    while time.time() < deadline:
        q3_edges = find_q3()
        if q3_edges is None:
            break
            
        # Remove one edge from q3_edges
        # Randomly pick one to avoid getting stuck in a bad pattern?
        # Or pick the one with highest degree?
        # Let's pick randomly.
        random_edge = random.choice(q3_edges)
        edge_set.remove(frozenset(random_edge))
        
    # Reconstruct list
    return [list(e) for e in edge_set]
