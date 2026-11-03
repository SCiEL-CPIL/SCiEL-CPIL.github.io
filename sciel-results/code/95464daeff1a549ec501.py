"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _strassen_2x2x2():
    """Return U, V, W matrices for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = a00(b01-b11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = a11(b10-b00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _hybrid_2x2x3():
    """Return U, V, W for 2x2x3: Strassen on cols 0-1 + trivial on col 2 (rank 11)."""
    U1, V1, W1 = _strassen_2x2x2()
    U = np.zeros((4, 11), dtype=int)
    V = np.zeros((6, 11), dtype=int)
    W = np.zeros((6, 11), dtype=int)
    for c in range(7):
        U[:, c] = U1[:, c]
        V[0, c] = V1[0, c]  # j=0,k=0
        V[1, c] = V1[1, c]  # j=0,k=1
        V[3, c] = V1[2, c]  # j=1,k=0
        V[4, c] = V1[3, c]  # j=1,k=1
        W[0, c] = W1[0, c]  # k=0,i=0
        W[1, c] = W1[1, c]  # k=0,i=1
        W[2, c] = W1[2, c]  # k=1,i=0
        W[3, c] = W1[3, c]  # k=1,i=1
    # Trivial for column 2
    U[0,7]=1; V[2,7]=1; W[4,7]=1   # A[0,0]*B[0,2] -> C[0,2]
    U[1,8]=1; V[5,8]=1; W[4,8]=1   # A[0,1]*B[1,2] -> C[0,2]
    U[2,9]=1; V[2,9]=1; W[5,9]=1   # A[1,0]*B[0,2] -> C[1,2]
    U[3,10]=1; V[5,10]=1; W[5,10]=1 # A[1,1]*B[1,2] -> C[1,2]
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two tensor decompositions."""
    r1, r2 = U1.shape[1], U2.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    rank = r1 * r2
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(m1):
                        for j2 in range(m2):
                            u1 = U1[i1*m1+j1, c1]
                            u2 = U2[i2*m2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*m1*n2 + i2*m2 + j1*n2 + j2  # (i1*n2+i2)*m2 + (j1*n2+j2)
                                row = (i1*n2+i2)*m2 + (j1*n2+j2)
                                U[row, c] = u1 * u2
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            v1 = V1[j1*p1+k1, c1]
                            v2 = V2[j2*p2+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = (j1*p1+j2)*p2 + k1*p2 + k2  # wrong, let me redo
                                # j = j1*p1 + j2? No. j in B is the shared index.
                                # B is m x p, so B[j,k]. j = j1*m1_factor... 
                                # Actually for Kronecker: B[j1*m1+j2, k1*p1+k2]... no
                                # Let me think again. A is n1*n2 x m1*m2, B is m1*m2 x p1*p2
                                # A[i1*n2+i2, j1*m2+j2] * B[j1*m2+j2, k1*p2+k2]
                                # U row: i*m+j where i=i1*n2+i2, j=j1*m2+j2
                                # V row: j*p+k where j=j1*m2+j2, k=k1*p2+k2
                                # W row: k*n+i where k=k1*p2+k2, i=i1*n2+i2
                                row_v = (j1*m2+j2)*p2 + k1*p2 + k2
                                V[row_v, c] = v1 * v2
            for k1 in range(p1):
                for k2 in range(p2):
                    for i1 in range(n1):
                        for i2 in range(n2):
                            w1 = W1[k1*n1+i1, c1]
                            w2 = W2[k2*n2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row_w = (k1*p2+k2)*n2 + i1*n2 + i2
                                W[row_w, c] = w1 * w2
    return U, V, W, rank

def _verify(U_mat, V_mat, W_mat, n, m, p):
    """Verify the decomposition with random matrices."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    C_computed = np.zeros((n, p))
    for r in range(U_mat.shape[1]):
        alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _find_and_remove_dependencies(U, V, W, n, m, p, deadline):
    """
    Try to find linear dependencies among the rank-1 terms and remove redundant ones.
    Uses the tensor coefficient matrix approach: build T_mat where each column is
    the vectorized rank-1 tensor, then find the null space.
    """
    rank = U.shape[1]
    tensor_size = n * m * p  # 96 for 4x4x6
    
    # Build the tensor coefficient matrix
    T_mat = np.zeros((tensor_size, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_mat[i*m*p + j*p + k, r] = float(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
    
    # Check rank
    try:
        rank_T = np.linalg.matrix_rank(T_mat, tol=1e-10)
    except:
        return U, V, W, rank
    
    if rank_T >= rank:
        return U, V, W, rank  # No dependencies
    
    # Find null space vectors - these give linear combinations that sum to zero
    # Use SVD to find null space
    try:
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=True)
        # Null space is the last (rank - rank_T) rows of Vt
        null_dim = rank - rank_T
        if null_dim <= 0:
            return U, V, W, rank
        null_space = Vt[rank_T:, :]  # shape (null_dim, rank)
    except:
        return U, V, W, rank
    
    # For each null space vector, try to remove the term with the largest coefficient
    # We need to find integer (or rational) approximations
    # Strategy: for each null vector, find the index with max |coeff|, and check if
    # we can eliminate that term by adjusting others
    
    # Simpler approach: greedy elimination
    # Try removing each term one at a time and check if the remaining terms still span
    # the same tensor space
    
    current_rank = rank
    active = list(range(rank))
    
    for _ in range(20):  # Try up to 20 eliminations
        if time.time() > deadline:
            break
        
        # Find the term whose removal causes the least loss
        best_remove = -1
        best_rank_after = current_rank
        best_null_vec = None
        
        for idx in active:
            if time.time() > deadline:
                break
            # Check if removing term idx still allows reconstruction
            # This is expensive, so use a heuristic: check if the column is in
            # the span of others
            remaining = [i for i in active if i != idx]
            if len(remaining) == 0:
                continue
            T_remaining = T_mat[:, remaining]
            try:
                r_rem = np.linalg.matrix_rank(T_remaining, tol=1e-10)
            except:
                continue
            if r_rem >= rank_T:
                # Term idx is redundant
                best_remove = idx
                best_rank_after = len(remaining)
                break
        
        if best_remove == -1:
            break
        
        # Remove the term
        active.remove(best_remove)
        current_rank = len(active)
    
    if current_rank < rank:
        U_new = U[:, active]
        V_new = V[:, active]
        W_new = W[:, active]
        return U_new, V_new, W_new, current_rank
    
    return U, V, W, rank

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product structure with null-space elimination
    - 4x4x6 = (2x2x2) ⊗ (2x2x3)
    - 2x2x2 uses Strassen (rank 7)
    - 2x2x3 uses Strassen for first 2 cols (rank 7) + trivial for 3rd col (rank 4) = rank 11
    - Total rank = 7 * 11 = 77
    - Then apply structured null-space elimination to reduce rank
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 105.0
    
    # Build the Kronecker product decomposition
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _hybrid_2x2x3()
    
    U, V, W, rank = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 3)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction via null-space elimination
    U, V, W, rank = _find_and_remove_dependencies(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Rebuild from scratch if something went wrong
        U1, V1, W1 = _strassen_2x2x2()
        U2, V2, W2 = _hybrid_2x2x3()
        U, V, W, rank = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
