"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if U, V, W give an exact decomposition of matrix multiplication."""
    rank = U.shape[1]
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Build the full tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And check that T[i, j, k] == 1 if i,j,k are the "multiplication" indices, else 0.
    # Actually, the tensor for matrix multiplication C = A * B is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T is such that <T, A ⊗ B ⊗ C> = trace(C^T (A B))
    # In the trace convention, T[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k, and 0 otherwise.
    # Wait, let me re-read the contract.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So the bilinear form is sum_{i,j,k,r} A[i,j] * B[j,k] * W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # This should equal sum_{i,k} C[i,k] * C[i,k] where C = A*B.
    # Actually, it equals sum_{i,j,k} A[i,j] B[j,k] (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]).
    # This must equal sum_{i,k} (sum_j A[i,j] B[j,k]) * C[i,k].
    # So the tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] must equal
    # the Kronecker delta for the multiplication: T[i,j,k] = 1 if we are multiplying A[i,j]*B[j,k] into C[i,k].
    # Actually, the standard matrix multiplication tensor has entries:
    # T[i, j, k] = 1 for all i, j, k, but it's not a tensor in that sense.
    # Let me think again.
    # The multiplication map is mu: M_{n,m} ⊗ M_{m,p} -> M_{n,p}.
    # mu(A ⊗ B) = A * B.
    # In coordinates: mu(A ⊗ B)[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T is in M_{n,m}^* ⊗ M_{m,p}^* ⊗ M_{n,p}^* such that
    # <T, A ⊗ B ⊗ C> = trace(C^T mu(A ⊗ B)) = sum_{i,k} C[i,k] sum_j A[i,j] B[j,k].
    # = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k, and 0 otherwise?
    # No, T is a 3-way tensor with indices (i*m+j), (j*p+k), (k*n+i).
    # Actually, the tensor is T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i.
    # T[a, b, c] = 1 if a, b, c are consistent (i.e., the j in a and b are the same, and the i in a and c are the same, and the k in b and c are the same), else 0.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k.
    # And the decomposition is: T = sum_r U[:, r] ⊗ V[:, r] ⊗ W[:, r].
    # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    # And for all a, b, c where the indices are inconsistent, the sum should be 0.
    # Let's verify with the schoolbook decomposition.
    # U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1.
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if r = (i,j,k), else 0.
    # So for consistent indices, the sum is 1. For inconsistent indices, e.g., a = i1*m+j1, b = j2*p+k2, c = k3*n+i3 where j1 != j2, then no r has U[a, r]=1, V[b, r]=1, W[c, r]=1 simultaneously, so the sum is 0.
    # Good.
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    return np.allclose(T_computed, T_expected)

def _strassen_2x2x2(U, V, W, i_off, j_off, k_off, rank_off):
    """Add Strassen's 7-term decomposition for a 2x2x2 sub-problem."""
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices: i_global = i_off + i, j_global = j_off + j, k_global = k_off + k
    # U index: (i_off+i)*2 + (j_off+j) ... wait, no.
    # U is of shape (n*m, rank). The row index is i*m + j.
    # But here, we are dealing with a sub-problem of size 2x2x2.
    # The local i, j, k are 0 or 1.
    # The global i, j, k are i_off+i, j_off+j, k_off+k.
    # But the global U row index is (i_off+i)*m + (j_off+j).
    # Similarly, V row index is (j_off+j)*p + (k_off+k).
    # W row index is (k_off+k)*n + (i_off+i).
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The 7 terms correspond to the 7 rank-1 components.
    # For each term, we have U, V, W vectors of length 2 (local i, j) and 2 (local j, k) and 2 (local k, i).
    
    # Let's define the local U, V, W for each of the 7 terms.
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U1: [1, 1] (for i=0,1, j=0,1? No, U is indexed by (i, j).)
    # Wait, U is of shape (n*m, rank). For a 2x2 sub-problem, the local U is 2x2.
    # The entries of U for term t are U[i, j] for i, j in {0,1}.
    # Similarly, V[j, k] for j, k in {0,1}, and W[k, i] for k, i in {0,1}.
    
    # Term 1: (A11+A22)(B11+B22)
    # U1 = [[1, 0], [0, 1]] (A11 and A22)
    # V1 = [[1, 1], [0, 0]] (B11 and B12? No, B11 and B22? Wait.)
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # So U1 = [[1, 1], [1, 1]]? No.
    # U1[i, j] is the coefficient of A[i, j].
    # For M1, we have A11 and A22. So U1 = [[1, 0], [0, 1]].
    # V1[j, k] is the coefficient of B[j, k].
    # For M1, we have B11 and B22. So V1 = [[1, 0], [0, 1]].
    # W1[k, i] is the coefficient of C[i, k] (since W[k*n+i, r] is the coefficient of C[i, k]).
    # M1 contributes to C11, C12, C21, C22? No, M1 = (A11+A22)(B11+B22) is a scalar.
    # It contributes to C11, C12, C21, C22 with coefficients 1, 1, 1, 1? No.
    # Actually, M1 is one of the 7 products. The final C is a linear combination of the 7 M's.
    # The tensor decomposition has 7 rank-1 terms. Each term t has U_t, V_t, W_t.
    # The contribution to T[i, j, k] from term t is U_t[i, j] * V_t[j, k] * W_t[k, i].
    # And the sum over t of these must equal 1 for all i, j, k.
    
    # Let's look up the standard Strassen tensor decomposition.
    # The 7 terms are:
    # 1. (A11+A22)(B11+B22) -> U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[1,1]]? No.
    # Actually, the standard decomposition is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # The corresponding U, V, W for each M:
    # M1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[1,1]] (since M1 contributes to all C's with coeff 1? No.)
    # Let's derive it.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for M1:
    # W1[0,0] = 1 (C11), W1[0,1] = 1 (C12? No, M1 doesn't contribute to C12.)
    # Wait, M1 = (A11+A22)(B11+B22). This is a scalar.
    # In the tensor decomposition, W1[k, i] is the coefficient such that M1 * W1[k, i] contributes to C[i, k].
    # So for M1, it contributes to C11, C12, C21, C22? No, M1 is just one product.
    # The final C is a linear combination of the M's.
    # So W_t[k, i] is the coefficient of M_t in C[i, k].
    
    # M1: C11: 1, C12: 0, C21: 0, C22: 1. So W1 = [[1, 0], [0, 1]].
    # M2: C21: 1, C22: -1. So W2 = [[0, 0], [1, -1]].
    # M3: C12: 1, C22: 1. So W3 = [[1, 0], [0, 1]].
    # M4: C11: 1, C21: 1. So W4 = [[1, 0], [1, 0]].
    # M5: C11: -1, C12: 1. So W5 = [[-1, 1], [0, 0]].
    # M6: C22: 1. So W6 = [[0, 0], [0, 1]].
    # M7: C11: 1. So W7 = [[1, 0], [0, 0]].
    
    # Now U and V:
    # M1 = (A11+A22)(B11+B22). U1 = [[1, 0], [0, 1]], V1 = [[1, 0], [0, 1]].
    # M2 = (A21+A22)B11. U2 = [[0, 0], [1, 1]], V2 = [[1, 0], [0, 0]].
    # M3 = A11(B12-B22). U3 = [[1, 0], [0, 0]], V3 = [[0, 1], [0, -1]].
    # M4 = A22(B21-B11). U4 = [[0, 0], [0, 1]], V4 = [[-1, 0], [1, 0]].
    # M5 = (A11+A12)B22. U5 = [[1, 1], [0, 0]], V5 = [[0, 0], [0, 1]].
    # M6 = (A21-A11)(B11+B12). U6 = [[-1, 0], [1, 0]], V6 = [[1, 1], [0, 0]].
    # M7 = (A12-A22)(B21+B22). U7 = [[0, 1], [0, -1]], V7 = [[0, 1], [1, 1]].
    
    # Let's verify one: M1.
    # U1 * V1 * W1: for i=0, j=0, k=0: U1[0,0]*V1[0,0]*W1[0,0] = 1*1*1 = 1.
    # For i=0, j=0, k=1: U1[0,0]*V1[0,1]*W1[1,0] = 1*0*1 = 0.
    # For i=0, j=1, k=0: U1[0,1]*V1[1,0]*W1[0,0] = 0*0*1 = 0.
    # For i=1, j=0, k=0: U1[1,0]*V1[0,0]*W1[0,1] = 0*1*0 = 0.
    # Wait, W1[0,1] is 0. So M1 only contributes to (0,0,0) and (1,0,1)?
    # No, W1 = [[1, 0], [0, 1]]. So W1[k, i]:
    # k=0, i=0: 1. k=0, i=1: 0. k=1, i=0: 0. k=1, i=1: 1.
    # So M1 contributes to (i=0, j=0, k=0) with 1*1*1=1.
    # (i=0, j=1, k=0): U1[0,1]=0, so 0.
    # (i=1, j=0, k=0): U1[1,0]=0, so 0.
    # (i=1, j=1, k=0): U1[1,1]=1, V1[1,0]=0, so 0.
    # (i=0, j=0, k=1): U1[0,0]=1, V1[0,1]=0, so 0.
    # (i=1, j=0, k=1): U1[1,0]=0, so 0.
    # (i=0, j=1, k=1): U1[0,1]=0, so 0.
    # (i=1, j=1, k=1): U1[1,1]=1, V1[1,1]=1, W1[1,1]=1, so 1.
    # So M1 contributes 1 to (0,0,0) and (1,1,1).
    
    # Let's check the sum over all 7 M's for (0,0,0):
    # M1: 1
    # M2: U2[0,0]=0, so 0.
    # M3: U3[0,0]=1, V3[0,0]=0, so 0.
    # M4: U4[0,0]=0, so 0.
    # M5: U5[0,0]=1, V5[0,0]=0, so 0.
    # M6: U6[0,0]=-1, V6[0,0]=1, W6[0,0]=0, so 0.
    # M7: U7[0,0]=0, so 0.
    # Sum = 1. Good.
    
    # For (0,0,1):
    # M1: 0
    # M2: U2[0,0]=0, so 0.
    # M3: U3[0,0]=1, V3[0,1]=1, W3[1,0]=0, so 0.
    # M4: U4[0,0]=0, so 0.
    # M5: U5[0,0]=1, V5[0,1]=0, so 0.
    # M6: U6[0,0]=-1, V6[0,1]=1, W6[1,0]=0, so 0.
    # M7: U7[0,0]=0, so 0.
    # Sum = 0. Good.
    
    # This seems correct. I will implement this.
    
    terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),  # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),  # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[1, 0], [0, 1]]),  # M3
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),  # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),  # M5
        ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),  # M6
        ([[0, 1], [0, -1]], [[0, 1], [1, 1]], [[1, 0], [0, 0]]),  # M7
    ]
    
    n, m, p = DIMENSIONS
    for t in range(7):
        U_t, V_t, W_t = terms[t]
        col = rank_off + t
        for i in range(2):
            for j in range(2):
                U[(i_off + i) * m + (j_off + j)][col] = U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * p + (k_off + k)][col] = V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[(k_off + k) * n + (i_off + i)][col] = W_t[k][i]

def _schoolbook(U, V, W, i_off, j_off, k_off, rank_off, i_len, j_len, k_len):
    """Add schoolbook decomposition for a sub-problem."""
    n, m, p = DIMENSIONS
    col = rank_off
    for i in range(i_len):
        for j in range(j_len):
            for k in range(k_len):
                U[(i_off + i) * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + (k_off + k)][col] = 1
                W[(k_off + k) * n + (i_off + i)][col] = 1
                col += 1
    return col - rank_off

def construct():
    n, m, p = DIMENSIONS
    # Try rank 84 (schoolbook)
    # Try rank 78 (strassen on 2x2x2 blocks)
    # Let's try to construct a rank 78 decomposition.
    # Split A into A_top (1x4) and A_bot (2x4).
    # Split B into B_left (4x2) and B_right (4x5)? No, B is 4x7.
    # Let's split B into B1 (4x2) and B2 (4x5).
    # Then C = A_top * B1 + A_top * B2 + A_bot * B1 + A_bot * B2.
    # A_top * B1: 1x4 * 4x2 -> 1x2. Rank 1*4*2 = 8.
    # A_top * B2: 1x4 * 4x5 -> 1x5. Rank 1*4*5 = 20.
    # A_bot * B1: 2x4 * 4x2 -> 2x2. Rank 2*4*2 = 16. Use Strassen? No, Strassen is for 2x2*2x2.
    # Here it's 2x4 * 4x2. We can split B1 into B1a (2x2) and B1b (2x2).
    # Then A_bot * B1a: 2x4 * 2x2? No, B1 is 4x2.
    # Let's split the middle dimension j into 2 blocks of 2.
    # A_top: 1x4 -> split into A_top1 (1x2), A_top2 (1x2).
    # A_bot: 2x4 -> split into A_bot1 (2x2), A_bot2 (2x2).
    # B: 4x7 -> split into B1 (2x7), B2 (2x7).
    # Then C = A_top1*B1 + A_top2*B2 + A_bot1*B1 + A_bot2*B2.
    # A_top1*B1: 1x2 * 2x7 -> 1x7. Rank 1*2*7 = 14.
    # A_top2*B2: 1x2 * 2x7 -> 1x7. Rank 14.
    # A_bot1*B1: 2x2 * 2x7 -> 2x7. Rank 2*2*7 = 28.
    # A_bot2*B2: 2x2 * 2x7 -> 2x7. Rank 28.
    # Total rank: 14+14+28+28 = 84. No improvement.
    
    # To use Strassen, we need 2x2 * 2x2.
    # So we need to split B into 2x2 blocks.
    # B is 4x7. We can split it into 2x2, 2x2, 2x2, 2x1.
    # A is 3x4. Split into 3x2, 3x2.
    # Then:
    # A1 (3x2) * B1 (2x2) -> 3x2. Rank 3*2*2 = 12.
    # A1 (3x2) * B2 (2x2) -> 3x2. Rank 12.
    # A1 (3x2) * B3 (2x2) -> 3x2. Rank 12.
    # A1 (3x2) * B4 (2x1) -> 3x1. Rank 3*2*1 = 6.
    # A2 (3x2) * B1 (2x2) -> 3x2. Rank 12.
    # A2 (3x2) * B2 (2x2) -> 3x2. Rank 12.
    # A2 (3x2) * B3 (2x2) -> 3x2. Rank 12.
    # A2 (3x2) * B4 (2x1) -> 3x1. Rank 6.
    # Total: 12*6 + 6*2 = 72 + 12 = 84.
    
    # It seems hard to beat 84 with simple block decompositions.
    # Let's just return the schoolbook decomposition for now.
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
# EVOLVE-BLOCK-END
