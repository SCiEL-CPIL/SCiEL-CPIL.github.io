"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse.csgraph as csgraph

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # 1. Compute Minimum Spanning Tree (MST)
    # Ensure the matrix is suitable for MST (symmetric, non-negative, 0 diagonal)
    # distance_matrix is already Euclidean, so it satisfies these.
    mst = csgraph.minimum_spanning_tree(distance_matrix)
    
    # 2. Extract edges from MST
    # mst is a sparse matrix. Convert to CSR for easy iteration over non-zeros.
    mst_csr = mst.tocsr()
    adj = [[] for _ in range(n)]
    
    # Iterate over non-zero elements in the upper triangle to avoid duplicates
    # Or just iterate all and add both directions.
    for i in range(n):
        start_idx = mst_csr.indptr[i]
        end_idx = mst_csr.indptr[i + 1]
        for k in range(start_idx, end_idx):
            j = mst_csr.indices[k]
            w = mst_csr.data[k]
            if w > 0: # Ignore self-loops or zero weights
                adj[i].append((j, w))
                # Since MST is undirected in this context, we add both ways
                # But csgraph.minimum_spanning_tree returns a directed graph representation?
                # Actually, it returns a sparse matrix where (i,j) is set if edge exists.
                # For undirected MST, usually (i,j) and (j,i) are both set?
                # Let's check scipy docs. "The minimum spanning tree is returned as a sparse matrix."
                # If input is symmetric, output is symmetric?
                # To be safe, we treat it as undirected.
                # If only one direction is present, we might miss edges.
                # Let's force symmetry in adjacency list construction by checking both or just trusting the symmetric input.
                # Given input is symmetric, MST should be symmetric.
                pass

    # Re-verify symmetry handling.
    # If mst[i,j] > 0, add edge i-j.
    # If mst[j,i] > 0, add edge j-i.
    # If both are present, we add duplicate to adjacency list?
    # Let's use a set of edges or just check if i < j.
    
    adj = [[] for _ in range(n)]
    # Iterate over all i, j
    # For efficiency with N=50, double loop is fine.
    mst_dense = mst.toarray()
    for i in range(n):
        for j in range(i + 1, n):
            if mst_dense[i, j] > 0:
                w = mst_dense[i, j]
                adj[i].append((j, w))
                adj[j].append((i, w))
                
    # 3. Compute Bottleneck Distances (Minimax Path)
    # bottleneck[i][j] = max edge weight on path from i to j in MST
    bottleneck = np.zeros((n, n))
    
    for start in range(n):
        # DFS from start
        visited = [False] * n
        stack = [(start, 0.0)]
        visited[start] = True
        
        while stack:
            node, max_w = stack.pop()
            bottleneck[start, node] = max_w
            for neighbor, w in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    new_max_w = max(max_w, w)
                    stack.append((neighbor, new_max_w))
                    
    # 4. Construct Heuristic
    # Combine local distance (distance_matrix) and global structure (bottleneck)
    # h_ij = 1/(d_ij + eps) + 1/(B_ij + eps)
    eps = 1e-9
    local_heur = 1.0 / (distance_matrix + eps)
    global_heur = 1.0 / (bottleneck + eps)
    
    # Normalize or scale? 
    # The ACO probability is proportional to pheromone * heuristic.
    # The relative scale matters. 
    # Adding them directly is a reasonable combination.
    
    H = local_heur + global_heur
    
    # Handle diagonal: distance is 0, bottleneck is 0.
    # H_ii = 1/eps + 1/eps = 2/eps.
    # This is very large. 
    # Ants don't pick i==j, so this shouldn't affect the outcome 
    # unless the ACO implementation has a bug. 
    # However, to be strictly compliant with "entries that are not positive become 1e-9", 
    # these are positive. 
    # But large diagonal values might cause numerical issues if not masked. 
    # Standard practice: set diagonal to 0.
    # If I set diagonal to 0, it becomes 1e-9 in the evaluator.
    # This is safer.
    np.fill_diagonal(H, 0.0)
    
    # Ensure no non-finite values
    H[~np.isfinite(H)] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
