import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the known rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Flatten the tensor dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u * dim_v * dim_w
    
    # Pre-flatten the rank-1 tensors
    # C_r is the vectorized form of u_r \otimes v_r \otimes w_r
    C = np.zeros((total_dim, rank))
    for r in range(rank):
        C[:, r] = np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r]).flatten()
        
    # Target tensor T
    T = np.zeros(total_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j) * dim_v * dim_w + (j * p + k) * dim_w + (k * n + i)] = 1.0

    # Iteratively remove redundant columns
    # We check if column r is in the span of the others
    # This is equivalent to checking if the rank of C_without_r is the same as rank of [C_without_r | T - C_r]
    # Or solving C_without_r x = T - C_r
    
    current_rank = rank
    U_cur = U_np
    V_cur = V_np
    W_cur = W_np
    C_cur = C
    
    # Optimization: We can use the fact that T = sum C_r
    # So T - C_r = sum_{r' != r} C_{r'}
    # We just need to check if C_r is in the span of {C_{r'} | r' != r}
    # This is equivalent to checking if the rank of the matrix formed by all C's is the same as the rank of the matrix formed by all C's except C_r.
    # Since we know T is in the span of all C's, if C_r is redundant, the rank doesn't drop.
    # Actually, if C_r is redundant, then rank(C) == rank(C_without_r).
    
    # We will use a simpler check: 
    # Compute the SVD of C_cur. If the smallest singular value is close to 0, there is a dependency.
    # But we want to identify WHICH column is redundant.
    
    # Efficient check for a specific column r:
    # Solve C_without_r x = C_r. If residual is small, r is redundant.
    
    def is_redundant(r, C_mat, current_rank):
        # C_mat shape: (total_dim, current_rank)
        # Create mask
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        C_rem = C_mat[:, mask]
        
        # Solve C_rem x = C_mat[:, r]
        # Use lstsq
        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(C_rem, C_mat[:, r], rcond=None)
            # Check residual
            if residuals.size > 0:
                if residuals[0] < 1e-9:
                    return True
            else:
                # Exact solution
                return np.allclose(C_rem @ x, C_mat[:, r], atol=1e-9)
        except np.linalg.LinAlgError:
            return False
        return False

    # Loop to remove columns
    # We restart the index after each removal to catch new dependencies
    r = 0
    while r < current_rank:
        if time.time() > deadline:
            break
            
        if is_redundant(r, C_cur, current_rank):
            # Remove column r
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            U_cur = U_cur[:, mask]
            V_cur = V_cur[:, mask]
            W_cur = W_cur[:, mask]
            C_cur = C_cur[:, mask]
            current_rank -= 1
            # Do not increment r, check the same index again (which is now the next column)
        else:
            r += 1
            
    # Convert back to lists
    U_out = U_cur.tolist()
    V_out = V_cur.tolist()
    W_out = W_cur.tolist()
    
    # Ensure entries are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
                
    return {"U": U_out, "V": V_out, "W": W_out}
