import numpy as np

def construct():
    n = 31
    # Initialize empty adjacency matrix
    M = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for _ in range(91):
        best_edge = None
        best_cost = float('inf')
        
        for u, v in all_edges:
            if M[u, v]:
                continue
            
            # Calculate cost: number of C4s added
            # S_u = neighbors of u, S_v = neighbors of v
            # Using M rows
            # M[u, :] is S_u, M[v, :] is S_v
            
            # We need to count C4s formed by adding (u,v)
            # As derived:
            # A = S_u \ K, B = S_v \ K, K = S_u \cap S_v
            
            su = M[u, :]
            sv = M[v, :]
            
            # K is where su and sv are both 1
            k_mask = su & sv
            k_indices = np.where(k_mask)[0]
            
            # A indices: su is 1, sv is 0
            a_mask = su & (~sv)
            a_indices = np.where(a_mask)[0]
            
            # B indices: sv is 1, su is 0
            b_mask = sv & (~su)
            b_indices = np.where(b_mask)[0]
            
            cost = 0
            
            # 1. Edges between A and B
            if len(a_indices) > 0 and len(b_indices) > 0:
                # Submatrix M[A, B]
                cost += np.sum(M[np.ix_(a_indices, b_indices)])
                
            # 2. Edges between A and K
            if len(a_indices) > 0 and len(k_indices) > 0:
                cost += np.sum(M[np.ix_(a_indices, k_indices)])
                
            # 3. Edges between K and B
            if len(k_indices) > 0 and len(b_indices) > 0:
                cost += np.sum(M[np.ix_(k_indices, b_indices)])
                
            # 4. Edges within K (counted twice in double sum, so divide by 2)
            if len(k_indices) > 0:
                cost += np.sum(M[np.ix_(k_indices, k_indices)]) // 2
                
            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        u, v = best_edge
        M[u, v] = 1
        M[v, u] = 1
        edges.append([u, v])
        
    return edges
