"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check shape
    if U.shape != (n*m, U.shape[1]) or V.shape != (m*p, V.shape[1]) or W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] should be decomposable as
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For matrix multiplication 2x2 * 2x3, the tensor is:
    # T[i*m+j, j2*p+k, k2*n+i2] = 1 if i==i2, j==j2, k==k2, else 0
    # Wait, let me think about this more carefully.
    # C = A @ B, where A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form: C[i,k] = sum_{i',j',j'',k'} A[i',j'] * B[j'',k'] * T[i',j',j'',k',i,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need T[i',j',j'',k',i,k] = 1 iff i'=i, j'=j, j''=j, k'=k, else 0
    # In the decomposition: T[a,b,c] where a=i'*m+j', b=j''*p+k', c=k'*n+i (trace convention)
    # Actually the trace convention is W[k*n+i, r]
    # So C[i,k] = sum_r (sum_{i',j'} A[i',j'] * U[i'*m+j', r]) * (sum_{j'',k'} V[j''*p+k', r] * B[j'',k']) * W[k*n+i, r]
    # = sum_r W[k*n+i, r] * (sum_{i',j'} A[i',j'] U[i'*m+j',r]) * (sum_{j'',k'} V[j''*p+k',r] B[j'',k'])
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Let's build the full tensor and check
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    # Check entry (i*m+j, j2*p+k, k*n+i) in the decomposed tensor
                    # Actually, the tensor index is (a, b, c) where:
                    # a = i'*m+j' (input A index)
                    # b = j''*p+k' (input B index)  
                    # c = k*n+i (output C index, trace convention)
                    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # Should equal 1 if a's (i',j') = (i,j) AND b's (j'',k') = (j,k) ... 
                    # Wait, I need to be more careful.
                    pass
    
    # Simpler check: for each r, the outer product structure
    # Actually, let's just verify by computing C from A,B using the decomposition
    # and comparing to direct multiplication for random inputs.
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_direct = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            A_vec = U[:, r]  # shape (n*m,)
            B_vec = V[:, r]  # shape (m*p,)
            W_vec = W[:, r]  # shape (p*n,)
            A_mat = A_vec.reshape(n, m)
            B_mat = B_vec.reshape(m, p)
            scalar_A = np.sum(A * A_mat)  # <A, A_r>
            scalar_B = np.sum(B * B_mat)  # <B, B_r>
            W_mat = W_vec.reshape(p, n)  # trace convention: W[k*n+i] -> W_mat[k,i]
            C_decomp += scalar_A * scalar_B * W_mat.T  # W_mat.T[i,k]
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _try_rank_11():
    """Construct a rank-11 decomposition: Strassen (7) + 4 terms for 3rd column."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Index conventions:
    # U[i*m+j, r] for A[i,j]
    # V[j*p+k, r] for B[j,k]
    # W[k*n+i, r] for C[i,k] (trace convention)
    # So C[i,k] -> W index = k*n+i
    # C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3, C[0,2]->4, C[1,2]->5
    
    # Strassen for 2x2x2 (columns 0,1 of B):
    # C[0,0] = M1 - M3 + M7
    # C[0,1] = M3 - M5 - M7
    # C[1,0] = M2 + M4 - M6
    # C[1,1] = -M1 + M4 + M5 + M6
    
    # M1 = A[0,0] * (B[0,0]+B[1,1])
    U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A[0,0] * (B[1,0]+B[1,1])
    U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = 1
    W[0, 2] = -1; W[1, 2] = 1
    
    # M4 = A[1,1] * (B[0,0]+B[0,1])
    U[3, 3] = 1; V[0, 3] = 1; V[1, 3] = 1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 5] = 1; U[3, 5] = -1; V[2, 5] = 1; V[3, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    
    # M7 = (A[0,0]-A[1,0]) * (B[0,0]+B[0,1])
    U[0, 6] = 1; U[2, 6] = -1; V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    # Third column (k=2): B indices j*3+2: B[0,2]->2, B[1,2]->5
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # M8 = A[0,0]*B[0,2]
    U[0, 7] = 1; V[2, 7] = 1
    W[4, 7] = 1
    
    # M9 = A[0,1]*B[1,2]
    U[1, 8] = 1; V[5, 8] = 1
    W[4, 8] = 1
    
    # M10 = A[1,0]*B[0,2]
    U[2, 9] = 1; V[2, 9] = 1
    W[5, 9] = 1
    
    # M11 = A[1,1]*B[1,2]
    U[3, 10] = 1; V[5, 10] = 1
    W[5, 10] = 1
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank 11 first (Strassen + 4 terms for 3rd column)
    r11 = _try_rank_11()
    if r11 is not None:
        return r11
    
    # If that fails, use schoolbook (rank 12)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
