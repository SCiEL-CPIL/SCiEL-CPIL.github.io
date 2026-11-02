import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    
    # Build the rank-50 decomposition (from current code)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for linear dependency using SVD on the tensor flattening
    # The tensor T has shape (n*m, m*p, p*n)
    # We want to check if the 50 rank-1 terms are linearly independent
    # Flatten the tensor to a matrix of shape (n*m*m*p, p*n) or similar
    # Actually, the standard way is to check the rank of the matricization
    # Or check if the vectors u_c ⊗ v_c ⊗ w_c are linearly independent
    # A simpler check: form the matrix M where M[i,j,k,c] = U[i,c]*V[j,c]*W[k,c]
    # This is too large. Instead, we can check the rank of the 3-mode matricizations
    # Or, more directly, check if the 50 columns of the "tensor matrix" are independent
    # The tensor can be seen as a linear map from (m*p) to (n*m) x (p*n)
    # Or from (n*m) x (m*p) to (p*n)
    
    # Let's use the SVD approach on a smaller matrix
    # Form the matrix K of size (n*m * m*p) x rank
    # K[i*m*p + j, c] = U[i,c] * V[j,c]
    # Then the tensor is T[i,j,k] = sum_c K[i,j,c] * W[k,c]
    # This doesn't directly give us the rank of the tensor decomposition
    # We need to check if the 50 vectors (u_c, v_c, w_c) are linearly independent
    # in the space of tensors
    
    # A practical approach: check if the 50 rank-1 tensors are linearly independent
    # by forming a matrix where each column is a flattened rank-1 tensor
    # The flattened tensor has size n*m*m*p*p*n = 6*27*18 = 2916
    # This is manageable
    
    mat = np.zeros((n*m*m*p*p*n, rank))
    for c in range(rank):
        # Flatten u_c ⊗ v_c ⊗ w_c
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    mat[(i*m*p + j)*p*n + k, c] = U_np[i,c] * V_np[j,c] * W_np[k,c]
    
    # Compute SVD to find the rank
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        # Find the rank (number of significant singular values)
        tol = s[0] * 50 * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There's a linear dependency
            # The null space is given by the last (rank - num_rank) rows of Vt
            # We can drop the column corresponding to the largest component in the null space
            null_vec = Vt[-1]  # Last row of Vt is in the null space
            # Find the column with the largest absolute value in null_vec
            drop_col = np.argmax(np.abs(null_vec))
            
            # Remove that column
            U_new = np.delete(U_np, drop_col, axis=1)
            V_new = np.delete(V_np, drop_col, axis=1)
            W_new = np.delete(W_np, drop_col, axis=1)
            
            # Verify
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n*m):
                ai = i // m
                am = i % m
                for j in range(m*p):
                    bj = j // p
                    bm = j % p
                    if am == bj:
                        k = bm * n + ai
                        T[i, j, k] = 1.0
            
            # Check residual
            residual = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = np.dot(U_new[i,:], V_new[j,:] * W_new[k,:])
                        residual += (val - T[i,j,k])**2
            
            if residual < 1e-8:
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
    except:
        pass
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
