"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 140
    n = 16

    # Q3: 8 vertices, 12 edges
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:
                q3_edges.append((u, v))
    assert len(q3_edges) == 12

    q3_adj = {v: set() for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)

    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120

    # Precompute all Q3 embeddings as frozensets of K16 edge indices
    # For each 8-subset of vertices, check all bijections to Q3 vertices
    # This is expensive but done once at startup
    q3_embeddings = []  # list of frozensets of edge indices
    
    # More efficient: for each subset of 8 vertices, try to find Q3 embeddings
    # Use the bipartite structure of Q3
    # Q3 has parts: even = {0,3,5,6}, odd = {1,2,4,7}
    
    # Actually, let's use a smarter approach:
    # For each pair of vertices (u,v) that could be an edge in Q3,
    # and for each Q3 edge position, try to extend.
    
    # Better: precompute all valid Q3 edge sets by iterating over all
    # 8-vertex subsets and checking if they induce a Q3 subgraph.
    
    # Even better: use the fact that Q3 is determined by its bipartition.
    # A Q3 subgraph on vertices {v0,...,v7} with bipartition (A,B) where |A|=|B|=4
    # has edges only between A and B, each vertex has degree 3, and the
    # adjacency matrix (4x4) has exactly 3 ones in each row and column.
    
    # The 4x4 0-1 matrix with row/col sums = 3 is the complement of a permutation matrix.
    # There are 4! = 24 such matrices.
    
    # So: for each 8-subset S of {0..15}, for each bipartition (A,B) with |A|=|B|=4,
    # for each of the 24 "complement of permutation" adjacency patterns,
    # check if the corresponding edges all exist in K16 (they always do since it's K16).
    # But we need to check if the graph contains this as a SUBGRAPH, meaning
    # all 12 edges of the Q3 pattern must be present in our graph.
    
    # Wait, I'm overcomplicating. Let me just precompute all Q3 edge sets.
    
    # For each combination of 8 vertices from 16:
    from itertools import combinations as comb
    import numpy as np
    
    # Precompute: for each 8-vertex subset, generate all possible Q3 edge sets
    # that are subgraphs of K16 (i.e., all edges between the 8 vertices)
    
    # Number of 8-subsets: C(16,8) = 12870
    # For each, number of Q3 embeddings: 8! / 48 (automorphisms of Q3) = 40320/48 = 840
    # Total: 12870 * 840 = ~10.8 million - too many to store all
    
    # Alternative approach: don't precompute all embeddings. Instead, use a fast
    # check function that uses bitsets.
    
    # Use integer bitsets for adjacency
    # adj[i] is a bitmask of vertices adjacent to i
    
    # Fast Q3 check: given a set of edges (as bitmask over 120 edges),
    # check if any Q3 embedding is fully contained.
    
    # Precompute for each Q3 edge position (a,b) in Q3, the list of K16 edge
    # indices that could correspond to it under some embedding.
    
    # Actually, the most efficient approach:
    # 1. Represent the current graph as a 16x16 adjacency matrix (bitmasks)
    # 2. For each 8-vertex subset, check if it contains a Q3 subgraph
    #    by checking all 24 bipartite Q3 patterns
    
    # Let's use numpy for speed
    
    # Precompute all Q3 "skeletons": for a fixed ordering of 8 vertices,
    # which pairs are edges
    # Q3 vertices: 0,1,2,3,4,5,6,7
    # Edge if XOR is a power of 2
    
    # For a given 8-tuple of distinct K16 vertices (v0,...,v7),
    # the Q3 subgraph exists if for all (a,b) in q3_edges,
    # edge (v_a, v_b) is in the graph.
    
    # To check efficiently: use the bipartite structure.
    # Q3 bipartition: A={0,3,5,6}, B={1,2,4,7}
    # Each a in A connects to exactly 3 vertices in B.
    
    # Precompute: for each a in A, the set of b in B that a connects to
    q3_A = [0, 3, 5, 6]
    q3_B = [1, 2, 4, 7]
    q3_AB = {}
    for a in q3_A:
        q3_AB[a] = [b for b in q3_B if (a ^ b) & ((a ^ b) - 1) == 0]
    
    # For a candidate 8-vertex set with bipartition (X, Y) where |X|=|Y|=4,
    # a Q3 exists if there's a bijection f: X->A, g: Y->B such that
    # for all a in A, b in q3_AB[a]: edge (f^{-1}(a), g^{-1}(b)) exists.
    
    # This is equivalent to: the 4x4 bipartite adjacency matrix between X and Y
    # (where M[i][j] = 1 if edge exists) must contain a 4x4 submatrix
    # (after permuting rows and cols) that is the complement of a permutation matrix.
    
    # i.e., M must have a 4x4 submatrix with exactly 3 ones in each row and col.
    # Equivalently, the complement (1-M) must contain a permutation matrix.
    
    # So: for each 8-subset S, for each bipartition (X,Y) of S with |X|=|Y|=4,
    # build the 4x4 matrix M where M[i][j] = 1 if edge(X[i], Y[j]) exists.
    # Check if (1-M) contains a permutation matrix, i.e., if there's a perfect
    # matching in the graph defined by 0s in M.
    
    # This can be done with a 4x4 Hungarian-like check or just try all 24 permutations.
    
    # Let's precompute all 24 permutations of 4 elements
    perms4 = list(permutations(range(4)))
    
    def check_q3_in_subset(X, Y, adj_bitmasks):
        """Check if the bipartite graph between X and Y contains a Q3 subgraph.
        X, Y are lists of 4 vertices. adj_bitmasks[i] is bitmask of neighbors of i."""
        # Build 4x4 matrix: M[i][j] = 1 if edge(X[i], Y[j]) exists
        # Q3 exists iff there's a permutation p such that for all i,
        # the "missing" edges form a permutation, i.e., M[i][p[i]] = 0 for all i
        # AND the full structure matches Q3.
        
        # Actually, for Q3: the bipartite adjacency must be exactly the complement
        # of a permutation matrix. So we need:
        # There exist permutations p (for rows) and q (for cols) such that
        # M[p[i]][q[j]] = 1 - (i==j) for all i,j.
        # Equivalently: after permuting rows by p and cols by q,
        # M becomes J - I (all ones except diagonal).
        
        # So: for some row perm p and col perm q, 
        # M[p[i]][q[j]] = 1 if i!=j, 0 if i==j.
        
        # Check: for each of 24 row perms p, for each of 24 col perms q:
        #   for all i,j: M[p[i]][q[j]] == (1 if i!=j else 0)
        
        # Precompute M
        M = [[0]*4 for _ in range(4)]
        for i in range(4):
            xi = X[i]
            mask = adj_bitmasks[xi]
            for j in range(4):
                yj = Y[j]
                if mask & (1 << yj):
                    M[i][j] = 1
        
        for p in perms4:
            for q in perms4:
                ok = True
                for i in range(4):
                    for j in range(4):
                        expected = 1 if i != j else 0
                        if M[p[i]][q[j]] != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
        return False
    
    # Even faster: precompute for each 8-subset and bipartition whether Q3 exists
    # But that's 12870 * C(8,4)/2 = 12870 * 35 = 450450 checks, each with 576 perm checks.
    # 450450 * 576 * 16 operations ~ 4 billion - too slow in Python.
    
    # Better approach: use the adjacency matrix directly with numpy.
    
    # Let me try a different strategy: greedy edge addition with fast Q3 check.
    # The Q3 check for a new edge (u,v):
    # A Q3 containing edge (u,v) must have u and v as adjacent Q3 vertices.
    # In Q3, edge (a,b) where a^b is a power of 2.
    # WLOG, map u->0, v->1 (since 0^1=1, a power of 2).
    # Then we need to assign the remaining 6 Q3 vertices {2,3,4,5,6,7} to
    # 6 of the remaining 14 K16 vertices.
    
    # For each assignment, check if all 12 Q3 edges are present.
    # But 14P6 = 14*13*12*11*10*9 = 2162160 assignments - too many.
    
    # Use constraints: 
    # Q3 vertex 0 connects to 1, 2, 4 (since 0^1=1, 0^2=2, 0^4=4)
    # Q3 vertex 1 connects to 0, 3, 5 (since 1^0=1, 1^3=2, 1^5=4)
    # So if u->0, v->1:
    #   u must connect to the images of 2 and 4 in K16
    #   v must connect to the images of 3 and 5 in K16
    
    # Let me use backtracking with strong pruning.
    
    def has_q3_fast(adj_bitmasks, u, v):
        """Check if adding edge (u,v) creates a Q3.
        adj_bitmasks[i] is bitmask of neighbors of i (NOT including the new edge (u,v) yet)."""
        # Map u->0, v->1 in Q3 (0^1=1, valid edge)
        # Also try u->1, v->0
        for mapping in [(u, v), (v, u)]:
            qv_to_kv = {0: mapping[0], 1: mapping[1]}
            used = {mapping[0], mapping[1]}
            
            # Q3 neighbors:
            # 0: {1, 2, 4}
            # 1: {0, 3, 5}
            # 2: {0, 3, 6}
            # 3: {1, 2, 7}
            # 4: {0, 5, 6}
            # 5: {1, 4, 7}
            # 6: {2, 4}
            # 7: {3, 5}
            
            # Vertex 2 must be a neighbor of 0 (i.e., mapping[0]) in K16
            # Vertex 4 must be a neighbor of 0 in K16
            # Vertex 3 must be a neighbor of 1 (i.e., mapping[1]) in K16
            # Vertex 5 must be a neighbor of 1 in K16
            
            # Candidates for Q3 vertex 2: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 4: neighbors of mapping[0] not in used
            # Candidates for Q3 vertex 3: neighbors of mapping[1] not in used
            # Candidates for Q3 vertex 5: neighbors of mapping[1] not in used
            
            mask0 = adj_bitmasks[mapping[0]]
            mask1 = adj_bitmasks[mapping[1]]
            
            # Remove used vertices
            mask0 &= ~(1 << mapping[0])
            mask0 &= ~(1 << mapping[1])
            mask1 &= ~(1 << mapping[0])
            mask1 &= ~(1 << mapping[1])
            
            # Get candidate lists
            cands2 = [i for i in range(n) if mask0 & (1 << i)]
            cands4 = [i for i in range(n) if mask0 & (1 << i)]
            cands3 = [i for i in range(n) if mask1 & (1 << i)]
            cands5 = [i for i in range(n) if mask1 & (1 << i)]
            
            # Try all combinations
            for c2 in cands2:
                if c2 in used:
                    continue
                for c4 in cands4:
                    if c4 == c2 or c4 in used:
                        continue
                    for c3 in cands3:
                        if c3 in used or c3 == c2 or c3 == c4:
                            continue
                        for c5 in cands5:
                            if c5 in used or c5 == c2 or c5 == c3 or c5 == c4:
                                continue
                            
                            # Now we have 6 vertices assigned: 0,1,2,3,4,5
                            # Need to assign 6 and 7
                            # Q3 vertex 6 connects to 2 and 4
                            # Q3 vertex 7 connects to 3 and 5
                            
                            # c6 must be neighbor of both c2 and c4
                            mask_c2 = adj_bitmasks[c2]
                            mask_c4 = adj_bitmasks[c4]
                            mask_c6 = mask_c2 & mask_c4
                            mask_c6 &= ~(1 << mapping[0])
                            mask_c6 &= ~(1 << mapping[1])
                            mask_c6 &= ~(1 << c2)
                            mask_c6 &= ~(1 << c3)
                            mask_c6 &= ~(1 << c4)
                            mask_c6 &= ~(1 << c5)
                            
                            cands6 = [i for i in range(n) if mask_c6 & (1 << i)]
                            
                            if not cands6:
                                continue
                            
                            # c7 must be neighbor of both c3 and c5
                            mask_c3 = adj_bitmasks[c3]
                            mask_c5 = adj_bitmasks[c5]
                            mask_c7 = mask_c3 & mask_c5
                            mask_c7 &= ~(1 << mapping[0])
                            mask_c7 &= ~(1 << mapping[1])
                            mask_c7 &= ~(1 << c2)
                            mask_c7 &= ~(1 << c3)
                            mask_c7 &= ~(1 << c4)
                            mask_c7 &= ~(1 << c5)
                            
                            cands7 = [i for i in range(n) if mask_c7 & (1 << i)]
                            
                            if not cands7:
                                continue
                            
                            for c6 in cands6:
                                for c7 in cands7:
                                    if c7 == c6:
                                        continue
                                    # Check all Q3 edges
                                    # Q3 edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    # Already checked: (0,1) is the new edge, (0,2),(0,4),(1,3),(1,5) by construction
                                    # Need to check: (2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
                                    
                                    # (2,3): c2-c3
                                    if not (adj_bitmasks[c2] & (1 << c3)):
                                        continue
                                    # (2,6): c2-c6
                                    if not (adj_bitmasks[c2] & (1 << c6)):
                                        continue
                                    # (3,7): c3-c7
                                    if not (adj_bitmasks[c3] & (1 << c7)):
                                        continue
                                    # (4,5): c4-c5
                                    if not (adj_bitmasks[c4] & (1 << c5)):
                                        continue
                                    # (4,6): c4-c6
                                    if not (adj_bitmasks[c4] & (1 << c6)):
                                        continue
                                    # (5,7): c5-c7
                                    if not (adj_bitmasks[c5] & (1 << c7)):
                                        continue
                                    # (6,7): c6-c7
                                    if not (adj_bitmasks[c6] & (1 << c7)):
                                        continue
                                    
                                    return True
        return False
    
    # Build initial graph: K_{3,13}
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    best_adj = adj[:]
    best_edges = set(edges)
    best_count = len(edges)
    
    random.seed(42)
    
    # Greedy: try to add edges in random order
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                edges.add((a, b))
        
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 15:
                break
            u, v = e
            if has_q3_fast(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]
    
    # Local search: try removing one edge and adding another
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            # Try each edge removal
            for e in list(edges):
                if time.time() > deadline - 10:
                    break
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add a different edge
                missing = [ne for ne in all_edges if ne not in edges]
                random.shuffle(missing)
                
                for ne in missing:
                    if time.time() > deadline - 10:
                        break
                    nu, nv = ne
                    if has_q3_fast(adj, nu, nv):
                        continue
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break  # Only try one addition per removal
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
