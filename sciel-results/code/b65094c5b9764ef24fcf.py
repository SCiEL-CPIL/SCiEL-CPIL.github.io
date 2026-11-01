import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 7
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all i,j,k
    dim_A = n * m  # 8
    dim_B = m * p  # 28
    dim_C = p * n  # 14
    
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # Reshape T to a matrix and use SVD to find approximate low-rank structure
    # T as a 3-tensor: T[a,b,c]. Reshape to (dim_A, dim_B*dim_C)
    T_mat = T.reshape(dim_A, dim_B * dim_C)
    
    # Try ranks from 50 down to see if we can get exact with half-integers
    # First, let's check what the numerical rank suggests
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # The singular values will tell us about the numerical rank
    # For the schoolbook tensor, there are 56 non-zero singular values (all = 1)
    # Let's check
    print("Singular values:", s_vals[:60])
    
    # Try a direct approach: use the structure of the problem
    # For each i (row of A), we compute a 1x7 row: C[i,:] = A[i,:] @ B
    # A[i,:] is 1x4, B is 4x7
    # This is a rank-4 operation (the row vector times B)
    # But we can use the fact that 4 = 2*2 and apply a 2-level decomposition
    
    # Let me try: decompose the 4-dimensional inner product using a 3-term Strassen-like method?
    # Actually, the rank of the 4x4x1 tensor (inner product) is 4, not reducible below 4.
    # But the full 2x4x7 tensor might have global structure.
    
    # Let's try rank 50 decomposition with SVD and check if it's exact
    best_rank = 56
    best_U, best_V, best_W = None, None, None
    
    for rank in range(50, 57):
        U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        if rank > len(s_vals):
            break
        # Truncated SVD
        U_r = U_svd[:, :rank] / np.sqrt(s_vals[:rank])
        V_r = Vt_svd[:rank, :] / np.sqrt(s_vals[:rank])
        # Reconstruct T from U_r and V_r
        # T[a,b,c] = sum_r U_r[a,r] * V_r[r, b*dim_C + c]
        # We need T = U @ V where U is (dim_A, rank), V is (rank, dim_B*dim_C)
        # Then V reshaped to (rank, dim_B, dim_C)
        V_reshaped = V_r.reshape(rank, dim_B, dim_C)
        # Verify: T_recon[a,b,c] = sum_r U_r[a,r] * V_reshaped[r,b,c]
        T_recon = np.einsum('ar,rcb->abc', U_r, V_r.reshape(rank, dim_B, dim_C).transpose(0,2,1))
        # Wait, let me be more careful
        # V_r has shape (rank, dim_B*dim_C). Index is r, (b*dim_C + c)
        # So V_reshaped[r, b, c] = V_r[r, b*dim_C + c]
        V_3d = V_r.reshape(rank, dim_B, dim_C)
        T_recon = np.einsum('ar,rb c->abc', U_r, V_3d)
        
        if np.allclose(T_recon, T, atol=1e-10):
            best_rank = rank
            # Now try to find half-integer decomposition
            break
    
    # For now, return schoolbook if we can't find better
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
