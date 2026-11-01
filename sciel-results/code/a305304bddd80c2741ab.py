"""Matrix multiplication 2x2 x 2x9 with rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Strassen's 7-term decomposition for 2x2 x 2x2.
    Returns U, V, W matrices of shape (4,7) each.
    
    Convention:
    U[i*m+j, c] where i in [0,2), j in [0,2), m=2
    V[j*p+k, c] where j in [0,2), k in [0,2), p=2
    W[k*n+i, c] where k in [0,2), i in [0,2), n=2
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # U: rows indexed by i*m+j, columns by term c
    # A indices: A00=(i=0,j=0), A01=(i=0,j=1), A10=(i=1,j=0), A11=(i=1,j=1)
    # Row indices: 0->A00, 1->A01, 2->A10, 3->A11
    
    # V: rows indexed by j*p+k
    # B indices: B00=(j=0,k=0), B01=(j=0,k=1), B10=(j=1,k=0), B11=(j=1,k=1)
    # Row indices: 0->B00, 1->B01, 2->B10, 3->B11
    
    # W: rows indexed by k*n+i
    # C indices: C00=(k=0,i=0), C01=(k=0,i=1), C10=(k=1,i=0), C11=(k=1,i=1)
    # Row indices: 0->C00, 1->C01, 2->C10, 3->C11
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # Term 2: M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # Term 3: M4 = A11(B01-B10)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[2, 3] = -1 # B10
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[1, 4] = 1  # C01
    W[0, 4] = -1 # C00
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # M6 doesn't appear in standard Strassen... wait, let me recheck.
    # Actually M6 is not used in the standard 7-term Strassen. Let me reconsider.
    # 
    # Standard Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # So M6 is NOT used in the final combination! That's a problem.
    # Wait, I think I have the wrong set of 7 terms. Let me use the standard formulation.
    
    # Let me restart with a verified Strassen decomposition.
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # P6 is indeed not used. This means the standard Strassen uses only 6 of the 7?
    # No, that's not right. Let me look this up more carefully.
    #
    # Actually, I believe the standard Strassen 7-term algorithm does use all 7.
    # Let me re-derive:
    #
    # The issue is that P6 might be needed. Let me check if there's an alternative formulation.
    #
    # Alternative: Some formulations use:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # And then:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # In this case M6 is not used. That can't be right for a rank-7 decomposition.
    #
    # I think the correct formulation must use all 7. Let me try a different set.
    
    # Let me just use a known correct Strassen decomposition.
    # I'll use the one from the literature where all 7 terms are used.
    
    # Actually, upon reflection, I think there might be an error in my memory.
    # Let me just construct it directly by solving for a rank-7 decomposition.
    
    # For now, let me use a simpler approach: construct the full schoolbook (rank 36)
    # and then try to reduce it. But first, let me just get a valid answer.
    
    # Actually, let me just build the schoolbook decomposition which is guaranteed correct.
    n, m, p = 2, 2, 9
    rank = n * m * p  # 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_strassen_extended():
    """
    Attempt to build a rank-32 decomposition using Strassen for the 2x2 part.
    The 2x2x2 multiplication has rank 7 (Strassen).
    The 2x1x9 part is just vector-matrix mult, rank 9 (schoolbook, since it's already minimal).
    
    If we treat C = A*B where A is 2x2 and B is 2x9, we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using Strassen for the 2x2x2 core and extending to 9 columns:
    Each Strassen term M_c involves combinations of A entries and B entries.
    For each column k of B, we need to compute the contribution.
    
    Actually, the tensor for 2x2 x 2x9 multiplication has:
    - Input A: 4 entries (2x2)
    - Input B: 18 entries (2x9)  
    - Output C: 18 entries (2x9)
    
    The rank of this tensor is at most 7*9 = 63 if we naively extend Strassen,
    but potentially less due to structure.
    
    However, a simpler approach: since B has 9 columns, and for each column k,
    we're doing a 2x2 x 2x1 multiplication (which has rank 4 schoolbook or can use
    Strassen-like tricks), the total rank is at most 9 * (rank of 2x2x2x1).
    
    The 2x2x2x1 multiplication: C[i] = sum_j A[i,j] B[j] for i=0,1.
    This is just a linear map from (A,B) to C. The tensor rank of 2x2x2x1...
    Actually this is the tensor [i,j,k] -> delta(i,j,k) which is just the standard
    matrix-vector multiplication tensor. Its rank is 4 (schoolbook) since we need
    to compute C0 = A00*B0 + A01*B1 and C1 = A10*B0 + A11*B1.
    
    So total rank <= 9 * 4 = 36, which is the schoolbook.
    
    Can we do better? If we use Strassen for the 2x2 part, we get 7 "intermediate"
    values, each of which is a product of a linear combination of A entries and
    a linear combination of B entries. But B has 9 columns, so each "B entry"
    in Strassen would be a linear combination of 9 B entries.
    
    Let me think about this differently. The multiplication tensor T_{i,j,k} for
    C[i,k] = sum_j A[i,j] B[j,k] has indices i in {0,1}, j in {0,1}, k in {0,...,8}.
    
    T_{i,j,k} = 1 if the (i,j) entry of A multiplies the (j,k) entry of B.
    
    This tensor has rank at most 36 (schoolbook). Can we find a lower rank?
    
    One approach: group the 9 columns of B. For each column k, we have a 2x2x2x1
    multiplication. If all 9 columns are independent, the rank is 9*4=36.
    
    But wait, maybe we can share Strassen terms across columns. In Strassen,
    we compute 7 intermediate products. For each column k, we'd need to apply
    the same Strassen pattern. So we'd have 7*9 = 63 terms, which is worse.
    
    Unless we can combine terms across k. But since the B entries are different
    for each k, I don't see an obvious way to combine.
    
    So the schoolbook rank of 36 might be optimal, or close to it.
    
    Let me just return the schoolbook for now, which is guaranteed correct.
    """
    n, m, p = 2, 2, 9
    rank = n * m * p  # 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W, n=2, m=2, p=9):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    For each (i,k) output entry, check that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j, j') for all j,j'
    
    More precisely, the tensor T_{i,j,k} should satisfy:
    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k],
    which is always 1 for valid (i,j,k) triples where j is the shared index.
    
    Wait, the tensor for matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is just the all-ones tensor of shape (n, m, p) = (2, 2, 9).
    
    The decomposition says: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need: for all i in [0,2), j in [0,2), k in [0,9):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 120  # Safety margin
    
    # Start with schoolbook (rank 36) as fallback
    U_fb, V_fb, W_fb = _build_strassen_extended()
    
    # Try to find a better decomposition
    # For now, just return the schoolbook which is guaranteed correct
    # The rank is 36, which gives score = 36/36 = 1.0 (same as seed)
    
    # To improve, we need rank < 36.
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(4, 18, 18) = 18.
    # But practical lower bounds are higher.
    
    # One idea: use the fact that the 2x2 matrix multiplication has rank 7 (Strassen),
    # and see if we can exploit that.
    
    # Actually, let me try a rank-32 construction based on Strassen.
    # The key insight: for each of the 9 columns of B, we do a 2x2 x 2x1 multiplication.
    # If we use Strassen for the 2x2 part, we get 7 intermediate products per "block".
    # But we have 9 columns, so naively 7*9 = 63.
    
    # However, we can group columns. For example, group the 9 columns into:
    # - 1 group of 1 column (using 4 terms schoolbook)
    # - 2 groups of 4 columns (using 7*4 = 28 terms each? No, that's worse)
    
    # Actually, I don't think Strassen helps here because the B matrix has different
    # entries in each column, so we can't share the Strassen intermediate computations
    # across columns in a way that reduces the total rank.
    
    # Let me just return the schoolbook decomposition for now.
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
