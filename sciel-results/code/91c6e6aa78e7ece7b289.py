"""Program for rank-minimal exact tensor decomposition of 6x6 matrix multiplication."""
import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # Use Strassen's algorithm with 2x2 blocks
    # 6x6 = 3x3 blocks of 2x2
    # Each 2x2 multiplication uses 7 scalar multiplications
    # Total rank = 3*3*3*7 = 189
    
    rank = 189
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for ib in range(3):  # block row of A
        for jb in range(3):  # block col of C
            for kb in range(3):  # block col of A / block row of B
                # A block at (ib, kb): rows 2*ib..2*ib+1, cols 2*kb..2*kb+1
                # B block at (kb, jb): rows 2*kb..2*kb+1, cols 2*jb..2*jb+1
                # C block at (ib, jb): rows 2*ib..2*ib+1, cols 2*jb..2*jb+1
                
                def u_idx(ar, ac):
                    return (2*ib + ar) * m + (2*kb + ac)
                
                def v_idx(br, bc):
                    return (2*kb + br) * p + (2*jb + bc)
                
                def w_idx(cr, cc):
                    # W[k*n+i] where C[i,k], so i=cr, k=cc
                    return (2*jb + cc) * n + (2*ib + cr)
                
                # M1 = (A11 + A22) * (B11 + B22)
                U[u_idx(0,0), col] = 1
                U[u_idx(1,1), col] = 1
                V[v_idx(0,0), col] = 1
                V[v_idx(1,1), col] = 1
                W[w_idx(0,0), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M2 = (A21 + A22) * B11
                U[u_idx(1,0), col] = 1
                U[u_idx(1,1), col] = 1
                V[v_idx(0,0), col] = 1
                W[w_idx(1,0), col] = 1
                W[w_idx(1,1), col] = -1
                col += 1
                
                # M3 = A11 * (B12 - B22)
                U[u_idx(0,0), col] = 1
                V[v_idx(0,1), col] = 1
                V[v_idx(1,1), col] = -1
                W[w_idx(0,1), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M4 = A22 * (B21 - B11)
                U[u_idx(1,1), col] = 1
                V[v_idx(1,0), col] = 1
                V[v_idx(0,0), col] = -1
                W[w_idx(0,0), col] = 1
                W[w_idx(1,0), col] = 1
                col += 1
                
                # M5 = (A11 + A12) * B22
                U[u_idx(0,0), col] = 1
                U[u_idx(0,1), col] = 1
                V[v_idx(1,1), col] = 1
                W[w_idx(0,0), col] = -1
                W[w_idx(0,1), col] = 1
                col += 1
                
                # M6 = (A21 - A11) * (B11 + B12)
                U[u_idx(1,0), col] = 1
                U[u_idx(0,0), col] = -1
                V[v_idx(0,0), col] = 1
                V[v_idx(0,1), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M7 = (A12 - A22) * (B21 + B22)
                U[u_idx(0,1), col] = 1
                U[u_idx(1,1), col] = -1
                V[v_idx(1,0), col] = 1
                V[v_idx(1,1), col] = 1
                W[w_idx(0,0), col] = 1
                col += 1
    
    # Verify the decomposition
    # For random A, B, check that sum_r U[:,r] * V[:,r] * W[:,r] gives C = A*B
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Flatten A, B, C
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        C_computed = np.zeros((p, n))
        for r in range(rank):
            prod = np.dot(U[:, r], A_flat) * np.dot(V[:, r], B_flat)
            # W[:, r] is indexed as W[k*n+i] = coefficient for C[i,k]
            w_col = W[:, r]
            for idx in range(p * n):
                i = idx % n
                k = idx // n
                C_computed[k, i] += prod * w_col[idx]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Decomposition is incorrect
            raise ValueError("Decomposition verification failed")
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
