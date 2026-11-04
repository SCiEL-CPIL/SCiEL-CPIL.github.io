import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _strassen_4x4x4_rank17():
    """
    Rank-17 decomposition for 4x4x4 matrix multiplication.
    Returns U, V, W of shape (16, 17) each.
    U index: i*4 + j (i,j in 0..3)
    V index: j*4 + k (j,k in 0..3)  
    W index: k*4 + i (k,i in 0..3)
    """
    # The rank-17 decomposition for 4x4x4 based on Strassen's work
    # This uses the hierarchical Strassen approach: 4x4 = 2x2 blocks of 2x2
    # Each 2x2 uses 7 multiplications, but with sharing between blocks
    
    # Actually, let me use the known explicit rank-17 decomposition
    # The 17 terms correspond to specific linear combinations
    
    # For 4x4x4, we can view it as multiplying 4x4 matrices
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # But each of these involves 2x2 multiplications, which need 7 terms each
    # That gives 7*7 = 49 terms, not 17.
    
    # The rank-17 result is more subtle. Let me use a different approach.
    # Actually, the known lower bound for 4x4x4 is 17, and it was proven achievable.
    
    # Let me try a direct construction using the fact that we can share
    # computations across the 2x2 block structure.
    
    # Alternative: use the 2x2x2 Strassen recursively but with optimization
    # For now, let me implement the Kronecker product approach which gives rank 49
    # and see if we can find a better decomposition through search.
    
    # For the purpose of this program, let me use the rank-49 Kronecker approach
    # and add the 5th dimension terms, giving rank 51.
    # Then I'll add a search phase to try to find lower rank.
    
    return None  # Placeholder - we'll use the Kronecker approach below

def _strassen_factors_2x2x2():
    """Returns U, V, W of shape (4, 7) for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A11(B12-B22) -> C12 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: A22(B21-B11) -> C21 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5: (A11+A12)B22 -> C12 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C21 += M6, C22 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C12 += M7, C22 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _kron_4x4x4():
    """
    Kronecker product of 2x2x2 Strassen to get 4x4x4 decomposition.
    Returns U, V, W of shape (16, 49).
    """
    U_s, V_s, W_s = _strassen_factors_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U: i = 2*i1 + i2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i*4 + 0, c] = U_s[i1, c1] * U_s[i2, c2]  # This is wrong indexing
                    
    # Let me redo this properly
    # U index: row = i*4 + j where i,j are the 2x2 block indices
    # Wait, for 4x4 matrix multiplication, U has shape (4*4, rank) = (16, rank)
    # U[i*4 + j, c] corresponds to A[i,j]
    # V[j*4 + k, c] corresponds to B[j,k]
    # W[k*4 + i, c] corresponds to C[k,i]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i*4 + j, c] = U_s[i1*2 + j1, c1] * U_s[i2*2 + j2, c2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V[j*4 + k, c] = V_s[j1*2 + k1, c1] * V_s[j2*2 + k2, c2]
                                    W[k*4 + i, c] = W_s[k1*2 + i1, c1] * W_s[k2*2 + i2, c2]
    
    return U, V, W

def construct():
    deadline = time.time() + 120 - 10
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the 4x4x4 core using Kronecker product of Strassen 2x2x2
    U_444, V_444, W_444 = _kron_4x4x4()
    
    # Now extend to 4x5x5
    # A is 4x5: A[i,j] for i in 0..3, j in 0..4
    # B is 5x5: B[j,k] for j in 0..4, k in 0..4
    # C is 5x4: C[k,i] for k in 0..4, i in 0..3
    
    # The tensor T[i,j,k] = 1 for all i in 0..3, j in 0..4, k in 0..4
    # We can decompose as:
    # T = T_core (4x4x4 part) + T_extra (involving j=4 or k=4)
    
    # T_core: i in 0..3, j in 0..3, k in 0..3
    # T_extra: terms where j=4 or k=4 (but not both, to avoid double counting)
    #   - j=4, k in 0..3: 4x1x4 = 16 terms
    #   - j in 0..3, k=4: 4x4x1 = 16 terms  
    #   - j=4, k=4: 4x1x1 = 4 terms
    # Total extra: 16 + 16 + 4 = 36 terms
    # But we can be smarter with rank-1 decompositions
    
    # Rank-1 decomposition for the extra terms:
    # For j=4: A[i,4] * B[4,k] * C[k,i] for all i,k
    # This is a rank-1 term: U[i*5+4, c] = 1 for all i, V[4*5+k, c] = 1 for all k, W[k*4+i, c] = 1 for all i,k
    # Wait, that's not right. Let me think more carefully.
    
    # Actually, the term where j=4 contributes:
    # sum_k A[i,4] * B[4,k] * C[k,i]
    # This can be done with rank-1: U[i*5+4, c] = 1 for all i, V[4*5+k, c] = 1 for all k, W[k*4+i, c] = 1 for all i,k
    # But W[k*4+i, c] = 1 for all i,k means W is all ones, which is a valid rank-1 term.
    
    # Similarly for k=4:
    # sum_j A[i,j] * B[j,4] * C[4,i]
    # Rank-1: U[i*5+j, c] = 1 for all i,j, V[j*5+4, c] = 1 for all j, W[4*4+i, c] = 1 for all i
    
    # And for j=4, k=4:
    # A[i,4] * B[4,4] * C[4,i]
    # This is already covered by the two rank-1 terms above (it's the intersection)
    # But we need to be careful not to double count.
    
    # Let me use a different decomposition:
    # Term 1: j=4, all k (rank 1): U[i*5+4, c1] = 1, V[4*5+k, c1] = 1, W[k*4+i, c1] = 1
    # Term 2: k=4, j in 0..3 (rank 1): U[i*5+j, c2] = 1 for j in 0..3, V[j*5+4, c2] = 1 for j in 0..3, W[4*4+i, c2] = 1
    # Term 3: j=4, k=4 (rank 1): U[i*5+4, c3] = 1, V[4*5+4, c3] = 1, W[4*4+i, c3] = 1
    
    # But wait, Term 1 already includes j=4, k=4. So if I use Term 1 and Term 2, I'm double counting the j=4,k=4 term.
    # Solution: Use Term 1 for j=4 (all k), and Term 2 for k=4, j in 0..3 only.
    
    # So:
    # c1: j=4, all k in 0..4
    # c2: k=4, j in 0..3
    # Total extra rank: 2
    
    rank = 49 + 2  # 51
    U = np.zeros((n*m, rank))  # (20, 51)
    V = np.zeros((m*p, rank))  # (25, 51)
    W = np.zeros((p*n, rank))  # (20, 51)
    
    # Copy the 4x4x4 core (indices i in 0..3, j in 0..3, k in 0..3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j, 0:49] = U_444[i*4 + j, 0:49]
                V[j*5 + k, 0:49] = V_444[j*4 + k, 0:49]
                W[k*4 + i, 0:49] = W_444[k*4 + i, 0:49]
    
    # Term 1: j=4, all k in 0..4
    c1 = 49
    for i in range(4):
        U[i*5 + 4, c1] = 1.0
    for k in range(5):
        V[4*5 + k, c1] = 1.0
    for k in range(5):
        for i in range(4):
            W[k*4 + i, c1] = 1.0
    
    # Term 2: k=4, j in 0..3
    c2 = 50
    for i in range(4):
        for j in range(4):
            U[i*5 + j, c2] = 1.0
    for j in range(4):
        V[j*5 + 4, c2] = 1.0
    for i in range(4):
        W[4*4 + i, c2] = 1.0
    
    # Verify
    T = _make_tensor(n, m, p)
    if _verify_decomposition(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_fb = np.zeros((n*m, rank_schoolbook))
        V_fb = np.zeros((m*p, rank_schoolbook))
        W_fb = np.zeros((p*n, rank_schoolbook))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, c] = 1
                    V_fb[j*p + k, c] = 1
                    W_fb[k*n + i, c] = 1
                    c += 1
        return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
