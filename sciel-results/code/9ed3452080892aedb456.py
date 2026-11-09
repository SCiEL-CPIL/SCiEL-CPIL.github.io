"""Optimized Q3-free graph construction on 16 vertices."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 16
    
    # Precompute all Q3 embeddings as sets of 12 edges
    # Q3 vertices are 0..7, edges connect vertices differing by 1 bit
    q3_vertex_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_vertex_edges.append((i, j))
    
    # All 12 edges of Q3
    q3_edges = q3_vertex_edges
    
    # Precompute: for each edge (a,b) in Q3, what are the structural requirements
    # If we map (a,b) -> (u,v), then:
    # - N(a) \ {b} maps to N(u) \ {v}: need 2 neighbors x,y
    # - N(b) \ {a} maps to N(v) \ {u}: need 2 neighbors z,w  
    # - N(x) ∩ N(y) must contain p (common neighbor, image of the vertex adjacent to both x,y but not a,b)
    # - N(z) ∩ N(w) must contain q (common neighbor, image of the vertex adjacent to both z,w but not a,b)
    # - p and q must be adjacent
    
    # For Q3 with vertices 0-7:
    # 0:1,2,4  1:0,3,5  2:0,6  3:1,7  4:0,6  5:1,7  6:2,4,7  7:3,5,6
    # If edge is (0,1): 
    #   N(0)\{1} = {2,4}, N(1)\{0} = {3,5}
    #   common(2,4) = {6}, common(3,5) = {7}, and 6-7 is edge
    # If edge is (0,2):
    #   N(0)\{2} = {1,4}, N(2)\{0} = {6}
    #   Wait, N(2) = {0,6}, so N(2)\{0} = {6}. Only one neighbor.
    #   But in Q3, vertex 2 has degree 2. So this structure is different.
    #   Actually for Q3 detection, the edge (u,v) can map to any edge in Q3.
    #   The structure depends on which Q3 edge we're matching.
    
    # Let's precompute for each of the 12 Q3 edges the pattern:
    # pattern[(a,b)] = (neighbors_of_a_except_b, neighbors_of_b_except_a, 
    #                    common_neighbor_of_n_a, common_neighbor_of_n_b)
    # where n_a are the 2 neighbors of a (except b), n_b are the 2 neighbors of b (except a)
    
    q3_adj = {i: set() for i in range(8)}
    for a, b in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    # For each edge in Q3, determine the "witness" structure
    # Given edge (a,b) in Q3:
    # Na = q3_adj[a] - {b}  (should have size 2)
    # Nb = q3_adj[b] - {a}  (should have size 2)
    # p = the unique vertex adjacent to all of Na (and not a,b)
    # q = the unique vertex adjacent to all of Nb (and not a,b)
    # Check p-q is an edge
    
    q3_patterns = {}
    for a, b in q3_edges:
        Na = q3_adj[a] - {b}
        Nb = q3_adj[b] - {a}
        if len(Na) != 2 or len(Nb) != 2:
            continue
        # Find common neighbor of Na
        common_a = set.intersection(*[q3_adj[x] for x in Na]) - {a, b}
        common_b = set.intersection(*[q3_adj[x] for x in Nb]) - {a, b}
        if len(common_a) == 1 and len(common_b) == 1:
            p = list(common_a)[0]
            q = list(common_b)[0]
            if p in q3_adj[q]:  # p-q must be edge
                q3_patterns[(a, b)] = (Na, Nb, p, q)
                q3_patterns[(b, a)] = (Nb, Na, q, p)  # symmetric
    
    def check_q3_with_bitmask(adj_bm, u, v):
        """Check if adding edge (u,v) creates a Q3 using bitmask adjacency.
        adj_bm[i] is a bitmask where bit j is set if edge (i,j) exists."""
        # Try all 12 Q3 edge mappings
        # For each (a,b) in Q3, try mapping a->u, b->v
        for (a, b), (Na, Nb, p_idx, q_idx) in q3_patterns.items():
            if (a, b) not in q3_patterns:
                continue
            Na_list = sorted(Na)
            Nb_list = sorted(Nb)
            
            # Try all assignments of Na to neighbors of u, Nb to neighbors of v
            # Na has 2 elements, Nb has 2 elements
            # u's neighbors (excluding v): candidates for Na
            # v's neighbors (excluding u): candidates for Nb
            
            # Get neighbors of u and v
            nu = adj_bm[u] & ~(1 << v)  # neighbors of u excluding v
            nv = adj_bm[v] & ~(1 << u)  # neighbors of v excluding u
            
            # Na must be subset of nu (2 specific vertices)
            # Nb must be subset of nv (2 specific vertices)
            
            # For each permutation of assigning Na vertices to actual vertices
            # Na = {x1, x2} where x1, x2 are the 2 neighbors of a in Q3
            # We need to find actual vertices X1, X2 in nu such that:
            #   X1, X2 are adjacent to u (they are, since in nu)
            #   X1 and X2 have a common neighbor P (image of p_idx)
            #   Similarly for Nb: Z1, Z2 in nv with common neighbor Q
            #   P and Q are adjacent
            #   All 8 vertices distinct: u, v, X1, X2, Z1, Z2, P, Q
            
            x1, x2 = Na_list
            z1, z2 = Nb_list
            
            # Iterate over pairs in nu for X1, X2
            nu_list = [i for i in range(n) if (nu >> i) & 1]
            nv_list = [i for i in range(n) if (nv >> i) & 1]
            
            forbidden_base = 1 << u | 1 << v
            
            for i in range(len(nu_list)):
                X1 = nu_list[i]
                for j in range(i+1, len(nu_list)):
                    X2 = nu_list[j]
                    
                    # Find common neighbors of X1 and X2 (excluding u, v, X1, X2)
                    common_x = (adj_bm[X1] & adj_bm[X2]) & ~(forbidden_base | (1 << X1) | (1 << X2))
                    
                    if common_x == 0:
                        continue
                    
                    # For each pair in nv for Z1, Z2
                    for k in range(len(nv_list)):
                        Z1 = nv_list[k]
                        if Z1 == X1 or Z1 == X2:
                            continue
                        for l in range(k+1, len(nv_list)):
                            Z2 = nv_list[l]
                            if Z2 == X1 or Z2 == X2 or Z2 == Z1:
                                continue
                            
                            # Find common neighbors of Z1 and Z2
                            common_z = (adj_bm[Z1] & adj_bm[Z2]) & ~(forbidden_base | (1 << Z1) | (1 << Z2))
                            
                            if common_z == 0:
                                continue
                            
                            # P must be in common_x, Q in common_z
                            # P != Q, P-Q edge, P,Q not in {u,v,X1,X2,Z1,Z2}
                            forbidden_pq = forbidden_base | (1 << X1) | (1 << X2) | (1 << Z1) | (1 << Z2)
                            
                            # Get actual P candidates
                            P_cands = [i for i in range(n) if (common_x >> i) & 1 and not (forbidden_pq >> i) & 1]
                            Q_cands = [i for i in range(n) if (common_z >> i) & 1 and not (forbidden_pq >> i) & 1]
                            
                            for P in P_cands:
                                for Q in Q_cands:
                                    if P == Q:
                                        continue
                                    # Check P-Q is an edge
                                    if (adj_bm[P] >> Q) & 1:
                                        return False
            # End of Q3 edge mapping
        return True
    
    def build_adj_from_edges(edges):
        adj_bm = [0] * n
        for u, v in edges:
            adj_bm[u] |= (1 << v)
            adj_bm[v] |= (1 << u)
        return adj_bm
    
    def count_edges(adj_bm):
        count = 0
        for i in range(n):
            count += bin(adj_bm[i]).count('1')
        return count // 2
    
    best_edges = []
    best_count = 0
    
    # Strategy: Start with dense base graph, then greedily add edges
    # Base: K_{3,13} plus some edges within parts
    
    def try_configuration(part_a_size, part_a_vertices, trial_idx):
        nonlocal best_edges, best_count
        
        if time.time() > deadline - 15:
            return
        
        part_b = [v for v in range(n) if v not in part_a_vertices]
        
        # Build initial graph: all cross edges between A and B
        adj_bm = [0] * n
        edges = []
        
        for a in part_a_vertices:
            for b in part_b:
                adj_bm[a] |= (1 << b)
                adj_bm[b] |= (1 << a)
                edges.append((a, b))
        
        # Add edges within A (complete graph on A)
        for i in range(len(part_a_vertices)):
            for j in range(i+1, len(part_a_vertices)):
                a1 = part_a_vertices[i]
                a2 = part_a_vertices[j]
                adj_bm[a1] |= (1 << a2)
                adj_bm[a2] |= (1 << a1)
                edges.append((a1, a2))
        
        # Candidate edges: within B
        candidate_edges = []
        for i in range(len(part_b)):
            for j in range(i+1, len(part_b)):
                candidate_edges.append((part_b[i], part_b[j]))
        
        # Shuffle candidates
        random.shuffle(candidate_edges)
        
        # Greedy add
        for u, v in candidate_edges:
            if time.time() > deadline - 5:
                break
            if (adj_bm[u] >> v) & 1:
                continue
            # Try adding
            adj_bm[u] |= (1 << v)
            adj_bm[v] |= (1 << u)
            if check_q3_with_bitmask(adj_bm, u, v):
                edges.append((u, v))
            else:
                # Revert
                adj_bm[u] &= ~(1 << v)
                adj_bm[v] &= ~(1 << u)
        
        # Local search polish: try to add rejected edges
        current_edge_set = set(edges)
        rejected = [(u, v) for u, v in candidate_edges if (u, v) not in current_edge_set and (v, u) not in current_edge_set]
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            random.shuffle(rejected)
            for u, v in rejected:
                if time.time() > deadline - 3:
                    break
                if (adj_bm[u] >> v) & 1:
                    continue
                adj_bm[u] |= (1 << v)
                adj_bm[v] |= (1 << u)
                if check_q3_with_bitmask(adj_bm, u, v):
                    edges.append((u, v))
                    rejected.remove((u, v))
                    improved = True
                else:
                    adj_bm[u] &= ~(1 << v)
                    adj_bm[v] &= ~(1 << u)
        
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = list(edges)
    
    # Try different configurations
    # Config 1: A = {0,1,2}, B = {3..15}
    for trial in range(40):
        if time.time() > deadline - 20:
            break
        try_configuration(3, [0, 1, 2], trial)
    
    # Config 2: A = {0,1,2,3}, B = {4..15}
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        try_configuration(4, [0, 1, 2, 3], trial)
    
    # Config 3: A = {0,1}, B = {2..15}
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        try_configuration(2, [0, 1], trial)
    
    # Final polish on best solution
    if best_edges and time.time() < deadline - 2:
        adj_bm = build_adj_from_edges(best_edges)
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        # Try adding any missing edge
        all_pairs = list(combinations(range(n), 2))
        missing = [(u, v) for u, v in all_pairs if (u, v) not in edge_set and (v, u) not in edge_set]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 1:
                break
            adj_bm[u] |= (1 << v)
            adj_bm[v] |= (1 << u)
            if check_q3_with_bitmask(adj_bm, u, v):
                best_edges.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
            else:
                adj_bm[u] &= ~(1 << v)
                adj_bm[v] &= ~(1 << u)
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
