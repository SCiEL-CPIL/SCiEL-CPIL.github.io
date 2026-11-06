import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Q3 structure
    # Vertices 0..7 (binary 000..111)
    # Edges if Hamming distance 1
    q3_adj = [[False]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if i != j and bin(i ^ j).count('1') == 1:
                q3_adj[i][j] = True
                q3_adj[j][i] = True
                
    # Bipartition of Q3
    # Even parity: 0(000), 3(011), 5(101), 6(110)
    # Odd parity: 1(001), 2(010), 4(100), 7(111)
    q3_even = [0, 3, 5, 6]
    q3_odd = [1, 2, 4, 7]
    
    # For each vertex in even part, list neighbors in odd part
    # 0: 1, 2, 4
    # 3: 1, 2, 7
    # 5: 1, 4, 7
    # 6: 2, 4, 7
    # Map even index (0..3) to list of odd indices (0..3)
    # q3_odd[0]=1, q3_odd[1]=2, q3_odd[2]=4, q3_odd[3]=7
    # 0 (even 0) -> neighbors 1(odd 0), 2(odd 1), 4(odd 2) -> [0, 1, 2]
    # 3 (even 1) -> neighbors 1(odd 0), 2(odd 1), 7(odd 3) -> [0, 1, 3]
    # 5 (even 2) -> neighbors 1(odd 0), 4(odd 2), 7(odd 3) -> [0, 2, 3]
    # 6 (even 3) -> neighbors 2(odd 1), 4(odd 2), 7(odd 3) -> [1, 2, 3]
    q3_even_neighbors = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    
    def has_q3(adj):
        """
        Check if the graph defined by adjacency matrix adj contains a Q3 subgraph.
        adj is a numpy boolean array of size n x n.
        """
        for s in subsets_8:
            # s is a tuple of 8 vertices
            
            # Quick check: count edges in induced subgraph on s
            # If < 12, cannot contain Q3
            edge_count = 0
            for i in range(8):
                ui = s[i]
                for j in range(i+1, 8):
                    uj = s[j]
                    if adj[ui][uj]:
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Try all partitions of s into two sets of 4
            # There are C(8,4) = 70 partitions.
            # For each, check if one side can map to q3_even and the other to q3_odd
            # such that edges exist.
            
            s_list = list(s)
            
            # Precompute adjacency bits for the 8 vertices within s
            # local_adj[i][j] is True if s[i] and s[j] are connected
            local_adj = [[False]*8 for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    if i != j:
                        local_adj[i][j] = adj[s[i]][s[j]]
                        
            for x_combo in itertools.combinations(range(8), 4):
                x_set = set(x_combo)
                y_set = set(range(8)) - x_set
                
                # x_set maps to q3_even, y_set maps to q3_odd
                # OR x_set maps to q3_odd, y_set maps to q3_even
                
                # Case 1: x_set -> q3_even, y_set -> q3_odd
                # We need a bijection f: x_set -> range(4) such that for all i in x_set, j in y_set:
                # if j is in q3_even_neighbors[f(i)], then local_adj[i][j] must be True.
                # Wait, q3_even_neighbors are indices into q3_odd.
                # Let's say x_set vertices are assigned to positions 0,1,2,3 of q3_even.
                # y_set vertices are assigned to positions 0,1,2,3 of q3_odd.
                # This is a matching problem.
                # Actually, we can just iterate over permutations of y_set for a fixed assignment of x_set?
                # No, we can iterate over permutations of x_set (4! = 24) and check if a matching exists in y?
                # Or iterate over both? 24 * 24 = 576 checks per partition.
                # 70 partitions * 576 = 40320 checks per 8-set.
                # 12870 8-sets * 40320 = ~500 million operations. Too slow in Python.
                
                # Optimization:
                # For a fixed partition (X, Y), we can check if there exists a mapping.
                # Let's fix the mapping of X to q3_even positions (0,1,2,3).
                # There are 24 permutations.
                # For a given permutation p of X (assigning x[0] to pos 0, etc.),
                # we require that for each x[k] in X, its neighbors in Y (in the graph)
                # must include the images of the q3_odd neighbors of k.
                # Specifically, let N_q3(k) be the set of odd indices connected to even index k.
                # The vertices in Y corresponding to N_q3(k) MUST be connected to x[k] in the graph.
                # This implies that the set of neighbors of x[k] in Y must be a SUPERSET of the set of Y-vertices mapped to N_q3(k).
                # Since it's a bijection, the Y-vertices mapped to N_q3(k) are specific vertices.
                # This suggests we can prune.
                
                # Let's try a slightly different approach for speed:
                # Iterate over permutations of X (24).
                # For each, determine which Y vertices are "forced" to be neighbors of each X vertex.
                # Actually, let's just use a simple backtracking or bitmasks.
                
                # Let's represent the adjacency between X and Y as a 4x4 boolean matrix M.
                # M[i][j] is True if x[i] and y[j] are connected.
                # We want to know if there is a permutation sigma of {0,1,2,3} (mapping X indices to q3_even positions)
                # and a permutation tau of {0,1,2,3} (mapping Y indices to q3_odd positions)
                # such that for all i, j:
                # if q3_even_neighbors[sigma(i)] contains tau(j), then M[i][j] must be True.
                
                # This is equivalent to:
                # For all i, j: M[i][j] must be True whenever tau(j) is in q3_even_neighbors[sigma(i)].
                
                # Let's iterate over sigma (24).
                # For a fixed sigma, we need to find tau.
                # For each j (Y index), let S_j = { i in X | tau(j) is in q3_even_neighbors[sigma(i)] }.
                # Then for all i in S_j, M[i][j] must be True.
                # This constraint on tau is: for each j, tau(j) cannot be in the set of odd indices
                # that are NOT neighbors of some i where M[i][j] is False.
                
                # Let's define Forbidden[j] as the set of odd indices k such that
                # there exists i in X with M[i][j] == False AND k in q3_even_neighbors[sigma(i)].
                # Then tau(j) must NOT be in Forbidden[j].
                
                # So we need a perfect matching in a bipartite graph where Y indices are on one side,
                # odd indices (0..3) on the other, and edge (j, k) exists if k NOT in Forbidden[j].
                
                # This is much faster.
                
                # Build M (4x4)
                # x_indices = sorted(x_set) -- wait, we need consistent indexing.
                # Let's just use the list x_list = [s[i] for i in x_combo]
                # y_list = [s[i] for i in y_set]
                
                x_list = [s[i] for i in x_combo]
                y_list = [s[i] for i in y_set]
                
                # M[i][j] = adj[x_list[i]][y_list[j]]
                M = [[adj[x_list[i]][y_list[j]] for j in range(4)] for i in range(4)]
                
                # Iterate over sigma: permutation of 0,1,2,3
                # sigma[i] is the q3_even position assigned to x_list[i]
                for sigma in itertools.permutations(range(4)):
                    # Calculate Forbidden[j] for each j in 0..3
                    # Forbidden[j] is a bitmask of odd indices (0..3)
                    forbidden = [0] * 4 # bitmask
                    for j in range(4):
                        mask = 0
                        for i in range(4):
                            if not M[i][j]:
                                # x_list[i] is NOT connected to y_list[j]
                                # So y_list[j] cannot be mapped to any odd neighbor of x_list[i]'s q3 position
                                q3_pos = sigma[i]
                                # q3_even_neighbors[q3_pos] is a list of odd indices
                                for k in q3_even_neighbors[q3_pos]:
                                    mask |= (1 << k)
                        forbidden[j] = mask
                    
                    # Now check if there is a perfect matching from Y indices (0..3) to Odd indices (0..3)
                    # such that tau(j) is not in forbidden[j].
                    # Use bitmask DP or simple backtracking (4! = 24)
                    # Since it's small, backtracking is fine.
                    
                    def match(used_mask, j):
                        if j == 4:
                            return True
                        for k in range(4):
                            if not (used_mask & (1 << k)):
                                if not (forbidden[j] & (1 << k)):
                                    if match(used_mask | (1 << k), j + 1):
                                        return True
                        return False
                        
                    if match(0, 0):
                        return True
                        
                # Case 2: x_set -> q3_odd, y_set -> q3_even
                # Symmetric.
                # M[i][j] = adj[x_list[i]][y_list[j]]
                # Now X maps to odd, Y maps to even.
                # sigma: permutation of X to odd positions.
                # tau: permutation of Y to even positions.
                # Condition: if q3_odd_neighbors[sigma(i)] contains tau(j), then M[i][j] must be True.
                # Wait, we need q3_odd_neighbors.
                # q3_odd positions: 0(1), 1(2), 2(4), 3(7)
                # Neighbors in even:
                # 1 (odd 0) -> 0(even 0), 3(even 1), 5(even 2) -> [0, 1, 2]
                # 2 (odd 1) -> 0(even 0), 3(even 1), 6(even 3) -> [0, 1, 3]
                # 4 (odd 2) -> 0(even 0), 5(even 2), 6(even 3) -> [0, 2, 3]
                # 7 (odd 3) -> 3(even 1), 5(even 2), 6(even 3) -> [1, 2, 3]
                
                q3_odd_neighbors = [
                    [0, 1, 2],
                    [0, 1, 3],
                    [0, 2, 3],
                    [1, 2, 3]
                ]
                
                for sigma in itertools.permutations(range(4)):
                    forbidden = [0] * 4
                    for j in range(4):
                        mask = 0
                        for i in range(4):
                            if not M[i][j]:
                                q3_pos = sigma[i]
                                for k in q3_odd_neighbors[q3_pos]:
                                    mask |= (1 << k)
                        forbidden[j] = mask
                        
                    if match(0, 0):
                        return True
                        
        return False

    # Initial Graph: K_3,13 + K_3(A)
    # A = {0,1,2}, B = {3..15}
    adj = np.zeros((n, n), dtype=bool)
    current_edges = set()
    
    # Add A-B edges
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
            current_edges.add((a, b))
            
    # Add A-A edges (Triangle)
    a_edges = [(0,1), (1,2), (0,2)]
    for u, v in a_edges:
        adj[u][v] = True
        adj[v][u] = True
        current_edges.add((u, v))
        
    # Verify initial graph is Q3-free (should be)
    # has_q3(adj) # Skip for speed, we know it's free
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local Search
    b_vertices = list(range(3, 16))
    b_edges = [(b_vertices[i], b_vertices[j]) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    
    # We will try to add B-B edges greedily with random restarts
    # Since has_q3 is expensive, we should be careful.
    
    # Optimization: Only check has_q3 if we actually add an edge?
    # Yes.
    
    # To speed up has_q3, we can cache results? No, graph changes.
    # We can prune subsets_8?
    # If we add an edge (u,v), only subsets containing both u and v can become Q3.
    # But has_q3 checks all subsets.
    # We can modify has_q3 to take a list of "candidate subsets" to check.
    # When adding edge (u,v), the new Q3s must contain u and v.
    # So we only need to check subsets of 8 vertices that contain u and v.
    # Number of such subsets: C(14, 6) = 3003.
    # This is much smaller than 12870.
    
    def has_q3_restricted(adj, u, v):
        """
        Check if there is a Q3 containing edge (u,v).
        """
        # Generate all 8-vertex subsets containing u and v
        # Choose 6 from the remaining 14
        remaining = [x for x in range(n) if x != u and x != v]
        for combo_6 in itertools.combinations(remaining, 6):
            s = tuple(sorted((u, v) + tuple(combo_6)))
            
            # Check if s contains a Q3
            # We can reuse the logic from has_q3 but for a single subset
            
            # Count edges in s
            edge_count = 0
            for i in range(8):
                ui = s[i]
                for j in range(i+1, 8):
                    uj = s[j]
                    if adj[ui][uj]:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            s_list = list(s)
            
            for x_combo in itertools.combinations(range(8), 4):
                x_set = set(x_combo)
                y_set = set(range(8)) - x_set
                
                x_list = [s[i] for i in x_combo]
                y_list = [s[i] for i in y_set]
                
                M = [[adj[x_list[i]][y_list[j]] for j in range(4)] for i in range(4)]
                
                # Case 1: X -> Even, Y -> Odd
                for sigma in itertools.permutations(range(4)):
                    forbidden = [0] * 4
                    for j in range(4):
                        mask = 0
                        for i in range(4):
                            if not M[i][j]:
                                q3_pos = sigma[i]
                                for k in q3_even_neighbors[q3_pos]:
                                    mask |= (1 << k)
                        forbidden[j] = mask
                        
                    def match(used_mask, j):
                        if j == 4:
                            return True
                        for k in range(4):
                            if not (used_mask & (1 << k)):
                                if not (forbidden[j] & (1 << k)):
                                    if match(used_mask | (1 << k), j + 1):
                                        return True
                        return False
                        
                    if match(0, 0):
                        return True
                        
                # Case 2: X -> Odd, Y -> Even
                for sigma in itertools.permutations(range(4)):
                    forbidden = [0] * 4
                    for j in range(4):
                        mask = 0
                        for i in range(4):
                            if not M[i][j]:
                                q3_pos = sigma[i]
                                for k in q3_odd_neighbors[q3_pos]:
                                    mask |= (1 << k)
                        forbidden[j] = mask
                        
                    if match(0, 0):
                        return True
                        
        return False

    # Local search loop
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Reset to best known or initial?
        # Let's try starting from the base graph (K3,13 + K3) and adding edges
        adj = np.zeros((n, n), dtype=bool)
        current_edges = set()
        
        for a in range(3):
            for b in range(3, 16):
                adj[a][b] = True
                adj[b][a] = True
                current_edges.add((a, b))
                
        for u, v in a_edges:
            adj[u][v] = True
            adj[v][u] = True
            current_edges.add((u, v))
            
        # Shuffle B-B edges
        b_edges_shuffled = b_edges[:]
        random.shuffle(b_edges_shuffled)
        
        for u, v in b_edges_shuffled:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
                
            # Try adding edge
            adj[u][v] = True
            adj[v][u] = True
            current_edges.add((min(u,v), max(u,v)))
            
            # Check if this creates a Q3
            # Only check subsets containing u and v
            if has_q3_restricted(adj, u, v):
                # Revert
                adj[u][v] = False
                adj[v][u] = False
                current_edges.remove((min(u,v), max(u,v)))
            else:
                # Keep
                pass
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
