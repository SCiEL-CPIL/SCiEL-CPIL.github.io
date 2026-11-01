import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 2, 8
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with known rank-28 solution
    best = _get_strassen_28()
    best_rank = 28
    
    # Try to search for lower rank decompositions
    # Strategy: try rank 27, 26, ... down to some limit
    # Use a simple approach: start from rank-28 and try to zero out entries
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 20), -1):
        if time.time() > deadline - 10:
            break
        
        result = _search_low_rank(target_rank, n, m, p)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best

def _get_strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    num_blocks = p // 2  # 4 blocks
    
    for t in range(num_blocks):
        for r in range(7):
            # Fill U
            U[0, col] = U_s[r][0]
            U[1, col] = U_s[r][1]
            U[2, col] = U_s[r][2]
            U[3, col] = U_s[r][3]
            
            # Fill V
            v00 = 0 * p + 2 * t
            v01 = 0 * p + (2 * t + 1)
            v10 = 1 * p + 2 * t
            v11 = 1 * p + (2 * t + 1)
            
            V[v00, col] = V_s[r][0]
            V[v01, col] = V_s[r][1]
            V[v10, col] = V_s[r][2]
            V[v11, col] = V_s[r][3]
            
            # Fill W
            w00 = (2 * t) * n + 0
            w01 = (2 * t + 1) * n + 0
            w10 = (2 * t) * n + 1
            w11 = (2 * t + 1) * n + 1
            
            W[w00, col] = W_s[r][0]
            W[w01, col] = W_s[r][1]
            W[w10, col] = W_s[r][2]
            W[w11, col] = W_s[r][3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # The tensor T[i,j,k] = delta_{i,k} (for A[i,j]*B[j,k] -> C[i,k])
    # The decomposition says: T[i,j,k] = sum_r U[i*? , r] * V[j*?, r] * W[k*?, r]
    # With the convention: U indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs
    # W[k*n+i, r] where k is row of C, i is column... wait let me re-read.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is the row index of C (0..p-1) and i is... 
    # Actually for matrix multiplication A(m×n) * B(n×p) -> C(m×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Here n=2 (cols of A), m=2 (rows of A), p=8 (cols of B)
    # So A is 2×2, B is 2×8, C is 2×8
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..1, k in 0..7
    
    # U has shape (n*m, rank) = (4, rank): indexed by (i,j) where i=row of A, j=col of A
    # V has shape (m*p, rank) = (16, rank): indexed by (j,k) where j=row of B, k=col of B  
    # Wait, m=2 is rows of A = cols of B? No...
    
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # n=2, m=2, p=8
    # U: (4, rank) - this should correspond to A entries: A has m×n = 2×2 = 4 entries
    # V: (16, rank) - this should correspond to B entries: B has n×p = 2×8 = 16 entries
    # W: (16, rank) - this should correspond to C entries: C has m×p = 2×8 = 16 entries
    
    # So U[i*m + j, r] for A[i,j]? Or U[j*n + i, r]?
    # Given the W convention W[k*n+i, r], by analogy:
    # U[i*n + j, r] for A[i,j] where i in 0..m-1, j in 0..n-1
    # V[j*n + k, r] for B[j,k] where j in 0..n-1, k in 0..p-1  
    # W[k*n + i, r] for C[i,k] where i in 0..m-1, k in 0..p-1
    
    # Wait, W[k*n+i] with n=2: W[k*2+i] for k in 0..7, i in 0..1
    # That gives indices 0..15, which matches p*n = 8*2 = 16. Good.
    
    # The tensor equation: for all i,j,k:
    # delta_{i==k?} no... C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 if this term contributes, i.e., T[i,j,k] is the coefficient
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each triple (i,j,k) has coefficient 1.
    
    # The decomposition: T[i,j,k] = sum_r U[i*n+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify:
    for i in range(m):
        for j in range(n):
            for k in range(p):
                expected = 1  # coefficient in matrix multiplication
                actual = 0.0
                for r in range(rank):
                    u_idx = i * n + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    actual += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _search_low_rank(target_rank, n, m, p):
    """Try to find a decomposition of the given rank. Returns dict or None."""
    # This is a hard problem. Let's try a heuristic approach.
    # Start from rank-28 and try to zero out one column.
    
    base = _get_strassen_28()
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    
    # Try removing each column and see if we can fix it
    # This is very naive but let's try
    for col_to_remove in range(28):
        U_test = np.delete(U, col_to_remove, axis=1)
        V_test = np.delete(V, col_to_remove, axis=1)
        W_test = np.delete(W, col_to_remove, axis=1)
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
    
    return None
