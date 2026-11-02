"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    Returns a 3D array of shape (n*m, m*p, p*n) where:
    T[a, b, c] = 1 if (i,j,k) maps to (a=i*m+j, b=j*p+k, c=k*n+i)
    Actually, the tensor for matrix mult is: T[i,j,k] = 1
    U index: i*m + j (row of A, col of A)
    V index: j*p + k (row of B, col of B)
    W index: k*n + i (row of C, col of C)
    
    The tensor T has entries T[i,j,k] = 1 for all valid i,j,k.
    As a 3D tensor of shape (n*m, m*p, p*n):
    T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor
    T_recon = np.zeros_like(T)
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(T_recon, T, atol=tol)


def _strassen_block_2x2():
    """Strassen's 7-term decomposition for 2x2 matrix multiplication.
    Returns U, V, W matrices for the 2x2x2 tensor."""
    # Strassen's formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A indices: (i,j) where i=row, j=col, so A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: (j,k) where j=row, k=col, so B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: (i,k) where i=row, k=col, so C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # U: shape (4, 7) for A part (2x2 = 4 elements)
    # V: shape (4, 7) for B part
    # W: shape (4, 7) for C part
    
    # U columns (one per Strassen term), rows indexed by (i,j) -> i*2+j
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1; V[3, 2] = -1  # B12, B22
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1; V[0, 3] = -1  # B21, B11
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    
    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C12 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C21 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _embed_2x2_into_2x4x4(U2, V2, W2, n=2, m=4, p=4):
    """Embed a 2x2x2 Strassen decomposition into a 2x4x4 decomposition.
    
    For 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    We can split A into two 2x2 blocks: A[:, 0:2] and A[:, 2:4]
    B into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    C into two 2x2 blocks: C[:, 0:2] and C[:, 2:4]
    
    C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    So we have 4 independent 2x2x2 multiplications.
    Each uses 7 rank-1 terms, so total rank = 4 * 7 = 28.
    """
    # U2: (4, 7) where row index = i*2+j for i in {0,1}, j in {0,1}
    # For 2x4x4:
    # U has shape (n*m, rank) = (8, rank)
    # Row index for A[i,j]: i*m + j = i*4 + j
    # V has shape (m*p, rank) = (16, rank)  
    # Row index for B[j,k]: j*p + k = j*4 + k
    # W has shape (p*n, rank) = (8, rank)
    # Row index for C[k,i]: k*n + i = k*2 + i
    
    # The four 2x2x2 sub-problems:
    # 1. C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]: i in {0,1}, j in {0,1}, k in {0,1}
    #    A indices: i*4 + j for j in {0,1}
    #    B indices: j*4 + k for j in {0,1}, k in {0,1}
    #    C indices: k*2 + i for k in {0,1}
    
    # 2. C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]: i in {0,1}, j in {2,3}, k in {0,1}
    #    A indices: i*4 + j for j in {2,3}
    #    B indices: j*4 + k for j in {2,3}, k in {0,1}
    #    C indices: k*2 + i for k in {0,1}
    
    # 3. C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]: i in {0,1}, j in {0,1}, k in {2,3}
    #    A indices: i*4 + j for j in {0,1}
    #    B indices: j*4 + k for j in {0,1}, k in {2,3}
    #    C indices: k*2 + i for k in {2,3}
    
    # 4. C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]: i in {0,1}, j in {2,3}, k in {2,3}
    #    A indices: i*4 + j for j in {2,3}
    #    B indices: j*4 + k for j in {2,3}, k in {2,3}
    #    C indices: k*2 + i for k in {2,3}
    
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # For each of the 4 sub-problems, embed the 2x2x2 Strassen decomposition
    # Sub-problem s: A_cols = {s%2 * 2, s%2 * 2 + 1}, B_rows = {s%2 * 2, s%2 * 2 + 1}, B_cols = {(s//2) * 2, (s//2) * 2 + 1}
    # Wait, let me re-index:
    # Sub-problem 0: A cols {0,1}, B rows {0,1}, B cols {0,1} -> C cols {0,1}
    # Sub-problem 1: A cols {2,3}, B rows {2,3}, B cols {0,1} -> C cols {0,1}
    # Sub-problem 2: A cols {0,1}, B rows {0,1}, B cols {2,3} -> C cols {2,3}
    # Sub-problem 3: A cols {2,3}, B rows {2,3}, B cols {2,3} -> C cols {2,3}
    
    for s in range(4):
        a_col_offset = (s % 2) * 2  # 0 or 2
        b_row_offset = (s % 2) * 2  # 0 or 2
        b_col_offset = (s // 2) * 2  # 0 or 2
        c_col_offset = (s // 2) * 2  # 0 or 2
        
        for t in range(7):
            col = s * 7 + t
            # U: for each (i, j_local) in 2x2, map to global
            for i in range(2):
                for j_local in range(2):
                    u_idx = i * 2 + j_local  # index in U2
                    u_val = U2[u_idx, t]
                    if u_val != 0:
                        global_j = a_col_offset + j_local
                        U[i * 4 + global_j, col] = u_val
            
            # V: for each (j_local, k_local) in 2x2, map to global
            for j_local in range(2):
                for k_local in range(2):
                    v_idx = j_local * 2 + k_local  # index in V2
                    v_val = V2[v_idx, t]
                    if v_val != 0:
                        global_j = b_row_offset + j_local
                        global_k = b_col_offset + k_local
                        V[global_j * 4 + global_k, col] = v_val
            
            # W: for each (k_local, i) in 2x2, map to global
            for k_local in range(2):
                for i in range(2):
                    w_idx = k_local * 2 + i  # index in W2
                    w_val = W2[w_idx, t]
                    if w_val != 0:
                        global_k = b_col_offset + k_local
                        W[global_k * 2 + i, col] = w_val
    
    return U, V, W


def _try_greedy_rank1(T, n_terms, time_limit=10):
    """Try to find a rank-1 decomposition greedily. Returns None if not found."""
    start_time = time.time()
    n, m, p = T.shape
    U = np.zeros((n, n_terms))
    V = np.zeros((m, n_terms))
    W = np.zeros((p, n_terms))
    residual = T.copy()
    
    for t in range(n_terms):
        if time.time() - start_time > time_limit:
            return None
        
        # Find the entry with maximum absolute value in residual
        idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        val = residual[idx]
        if abs(val) < 1e-10:
            break
        
        i, j, k = idx
        # Greedy: set U[i,t] = 1, V[j,t] = 1, W[k,t] = val
        # But this doesn't work well for exact decomposition
        # Better: try to find a rank-1 component that best approximates residual
        # For exact decomposition with integer entries, this is hard
        
        # Simple greedy: place a 1 at the max position
        U[i, t] = 1
        V[j, t] = 1
        W[k, t] = val
        residual -= np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
    
    if np.allclose(residual, 0, atol=1e-9):
        return U, V, W
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank-28 decomposition
    U2, V2, W2 = _strassen_block_2x2()
    U, V, W = _embed_2x2_into_2x4x4(U2, V2, W2, n, m, p)
    
    # Convert to lists with integer entries
    U_list = [[int(round(U[i, c])) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(round(V[i, c])) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(round(W[i, c])) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U_list, V_list, W_list, T):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = U.shape[1]
    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to improve: attempt to reduce rank by checking if any columns are redundant
    # or by trying different block structures
    
    # Try 2x4x4 as two independent 2x2x4 problems (splitting B into two 2x4 halves)
    # Actually, let's try a different embedding: split A into 1x4 rows
    # A is 2x4. We can think of it as two 1x4 vectors.
    # C[:, :] = A[0, :] * B + A[1, :] * B shifted... no, that's not right.
    # C[i, k] = sum_j A[i,j] B[j,k]
    # This is inherently a 2x4x4 problem.
    
    # Try to see if rank 27 is possible by attempting a linear algebra approach
    # The tensor has 32 entries (all 1s in the relevant positions).
    # Actually, the tensor T has shape (8, 16, 8) but only 32 non-zero entries.
    
    # For now, return the rank-28 Strassen-based solution
    return best_result

# EVOLVE-BLOCK-END
