import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Vertices 0..15
    n = 16
    all_vertices = list(range(n))
    
    # Precompute 8-vertex subsets
    subsets_8 = list(combinations(all_vertices, 8))
    
    # Precompute bipartitions for a set of 8 vertices
    # We need to map local indices 0..7 to global indices
    # For a given subset S, we need all partitions of S into U, V of size 4
    # There are 35 such partitions.
    # We can precompute the local indices for these partitions.
    # local_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    # Partitions of [0..7] into two sets of 4.
    local_partitions = []
    for u in combinations(range(8), 4):
        v = tuple(x for x in range(8) if x not in u)
        # Avoid duplicates (U,V) and (V,U)
        # We can just check if min(u) < min(v) or something, 
        # or just store all and deduplicate by sorted tuple of tuples
        local_partitions.append((u, v))
    # Deduplicate
    unique_local_partitions = set()
    for u, v in local_partitions:
        # Canonical form: sort the two sets, then sort the pair
        set_u = frozenset(u)
        set_v = frozenset(v)
        if set_u > set_v:
            set_u, set_v = set_v, set_u
        unique_local_partitions.add((set_u, set_v))
    
    # Convert to lists for faster access
    final_local_partitions = []
    for su, sv in unique_local_partitions:
        # We need indices, not just sets, to build the matrix
        # Actually, we just need to know which local indices are in U and which in V
        u_list = [i for i in range(8) if i in su]
        v_list = [i for i in range(8) if i in sv]
        final_local_partitions.append((u_list, v_list))
        
    # There should be 35 partitions
    
    def contains_q3(adj_matrix):
        """
        adj_matrix: 16x16 numpy array or list of lists with 0/1
        """
        for S in subsets_8:
            # Build 8x8 submatrix
            # S is a tuple of 8 global indices
            # local_indices 0..7 correspond to S[0]..S[7]
            
            # Optimization: If the induced subgraph has fewer than 12 edges, it cannot contain Q3
            # Count edges in induced subgraph
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj_matrix[S[i]][S[j]]:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            # Check bipartitions
            for u_list, v_list in final_local_partitions:
                # u_list and v_list are local indices
                # Build 4x4 bipartite matrix M
                # M[i][j] = adj_matrix[S[u_list[i]]][S[v_list[j]]]
                
                # We need to check if 0s in M form a matching
                # 0s are where M[i][j] == 0
                
                # Row zero counts
                row_zeros = [0]*4
                col_zeros = [0]*4
                valid = True
                for i in range(4):
                    for j in range(4):
                        if adj_matrix[S[u_list[i]]][S[v_list[j]]] == 0:
                            row_zeros[i] += 1
                            col_zeros[j] += 1
                            if row_zeros[i] > 1 or col_zeros[j] > 1:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # 0s form a matching, so Q3 is present
                    return True
        return False

    # Initial graph: K3,13
    # Vertices 0,1,2 connected to 3..15
    adj = [[0]*n for _ in range(n)]
    edges = []
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = 1
            adj[b][a] = 1
            edges.append([a, b])
            
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    
    # List of all possible edges not in initial graph
    potential_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] == 0:
                potential_edges.append((u, v))
                
    # Greedy search with multiple restarts
    # We will try to add edges in random order
    # If an edge causes Q3, we skip it
    
    # To be more effective, we can sort potential edges by some heuristic?
    # Or just random shuffle for each restart.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.shuffle(potential_edges)
        
        current_adj = [row[:] for row in best_adj]
        current_edges = best_edges
        
        for u, v in potential_edges:
            if current_adj[u][v] == 1:
                continue
            # Try adding edge
            current_adj[u][v] = 1
            current_adj[v][u] = 1
            
            if not contains_q3(current_adj):
                current_edges += 1
                # Keep this edge
            else:
                # Revert
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in current_adj]
            
        # If we reached a high number, maybe stop early?
        # Max possible is 120. Q3-free max is likely around 50-60.
        if best_edges > 55:
             # Try a few more, but if we hit 60, we can be more aggressive
             pass

    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] == 1:
                result_edges.append([u, v])
                
    return result_edges
