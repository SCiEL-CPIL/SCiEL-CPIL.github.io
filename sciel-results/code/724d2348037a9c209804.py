import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col
        
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)
        
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1
        
        return U, V, W
    
    U, V, W = build_rank44()
    
    # Verify the decomposition is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U, V, W, n, m, p):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0.0
                    for r in range(rank):
                        coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_np, V_np, W_np, n, m, p):
        # Fallback: schoolbook rank-48
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: check for linearly dependent terms
    # A term r is linearly dependent if its "tensor vector" T_r = U[:,r] \otimes V[:,r] \otimes W[:,r]
    # can be expressed as a linear combination of other terms.
    # We check this by forming the matrix of all tensor vectors and finding its rank.
    
    if time.time() < deadline - 10:
        # Build the tensor matrix: each column r is the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # But that's 12*16*12 = 2304 entries per column, 44 columns.
        # Instead, we can check linear dependence more efficiently.
        # 
        # Key insight: if term r is a linear combination of other terms, then for ALL (i,j,k):
        #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{r' != r} c_{r'} * U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
        #
        # This is a system of n*m*p = 48 equations in (rank-1) unknowns for each candidate r.
        # We can try to find if any term is redundant.
        
        rank = 44
        # Try to find redundant terms by checking if removing term r still allows
        # the decomposition to hold (i.e., the remaining terms can still produce the identity tensor).
        
        # Efficient approach: build the 48 x 44 matrix where entry (ijk, r) = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # The tensor is the sum over r of these columns. If the columns are linearly dependent
        # in a way that the sum still equals the target, we can remove redundant ones.
        
        # Actually, the simplest check: if the 44 column vectors in R^48 are linearly dependent,
        # it doesn't necessarily mean we can remove one (the sum constraint matters).
        # But if a column is the zero vector, we can remove it.
        
        # Let's check: are any columns of the "contribution matrix" zero?
        # contribution[ijk, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        contrib = np.zeros((n*m*p, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        contrib[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    idx += 1
        
        # Check for zero columns
        col_norms = np.linalg.norm(contrib, axis=0)
        zero_cols = [r for r in range(rank) if col_norms[r] < 1e-12]
        
        # Check linear dependence of columns
        # If rank of contrib matrix < 44, some columns are linearly dependent
        matrix_rank = np.linalg.matrix_rank(contrib, tol=1e-10)
        
        if matrix_rank < rank:
            # Some columns are linearly dependent. Try to find a minimal subset.
            # Use greedy: try removing each column and check if the remaining still sum to target.
            target = np.ones(n*m*p)
            
            best_indices = list(range(rank))
            # Try removing one column at a time
            for r_remove in range(rank):
                if time.time() > deadline - 5:
                    break
                remaining = [r for r in range(rank) if r != r_remove]
                # Check if sum of remaining columns still equals target
                if len(remaining) < 1:
                    continue
                sub = contrib[:, remaining]
                col_sums = np.sum(sub, axis=1)
                if np.allclose(col_sums, target, atol=1e-9):
                    best_indices = remaining
                    break
            
            if len(best_indices) < rank:
                new_rank = len(best_indices)
                U_new = [[0]*new_rank for _ in range(dim_u)]
                V_new = [[0]*new_rank for _ in range(dim_v)]
                W_new = [[0]*new_rank for _ in range(dim_w)]
                for new_r, old_r in enumerate(best_indices):
                    U_new[:, new_r] = U_np[:, old_r].tolist()
                    V_new[:, new_r] = V_np[:, old_r].tolist()
                    W_new[:, new_r] = W_np[:, old_r].tolist()
                
                # Verify the reduced decomposition
                U_chk = np.array(U_new, dtype=float)
                V_chk = np.array(V_new, dtype=float)
                W_chk = np.array(W_new, dtype=float)
                if verify_decomp(U_chk, V_chk, W_chk, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    return {"U": U, "V": W if False else W, "W": W}  # careful with keys
    
    return {"U": U, "V": V, "W": W}
