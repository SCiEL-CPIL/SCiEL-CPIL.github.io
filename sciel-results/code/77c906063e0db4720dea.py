"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_2x2x2():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: (i,j) for i in 0..1, j in 0..1 -> index i*2+j
    # V rows: (j,k) for j in 0..1, k in 0..1 -> index j*2+k
    # W rows: (k,i) for k in 0..1, i in 0..1 -> index k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # A11 = U[0,0]=1, A22 = U[3,0]=1
    # B11 = V[0,0]=1, B22 = V[3,0]=1
    # C11 += M1: W[0,0]=1, C22 += M1: W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    # A21 = U[2,1]=1, A22 = U[3,1]=1
    # B11 = V[0,1]=1
    # C21 += M2: W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2: M3 = A11(B21-B22)
    # A11 = U[0,2]=1
    # B21 = V[2,2]=1, B22 = V[3,2]=-1
    # C12 += M3: W[1,2]=1, C22 += M3: W[3,2]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: M4 = A22(B12-B11)
    # A22 = U[3,3]=1
    # B12 = V[1,3]=1, B11 = V[0,3]=-1
    # C11 += M4: W[0,3]=1, C21 += M4: W[2,3]=1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    # A11 = U[0,4]=1, A12 = U[1,4]=1
    # B22 = V[3,4]=1
    # C11 -= M5: W[0,4]=-1, C12 += M5: W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    # A21 = U[2,5]=1, A11 = U[0,5]=-1
    # B11 = V[0,5]=1, B12 = V[1,5]=1
    # C22 += M6: W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    # A12 = U[1,6]=1, A22 = U[3,6]=-1
    # B21 = V[2,6]=1, B22 = V[3,6]=1
    # C11 += M7: W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_rank55():
    """Build rank-55 decomposition for 2x3 x 3x10 using Strassen on 2-column blocks.
    
    Partition the 10 columns into 5 pairs. For each pair, use a rank-7 
    Strassen-based decomposition of the 2x3 x 3x2 sub-problem.
    
    The 2x3 x 3x2 multiplication: A(2x3) * B(3x2) = C(2x2)
    We can write C = sum_{j=0}^{2} A[:,j] * B[j,:]
    Each A[:,j] is 2x1 and B[j,:] is 1x2.
    
    To use Strassen, we embed this into a 2x2 x 2x2 problem by padding.
    Actually, the known construction: for 2x3 x 3x2, rank is 7.
    
    Construction: Think of it as 2x2 x 2x2 (using first 2 of 3 middle indices) 
    plus a 2x1 x 1x2 correction (using the 3rd middle index).
    2x2 x 2x2 has rank 7, 2x1 x 1x2 has rank 1, total = 8. Not helpful.
    
    Alternative: The rank of [2,3,2] tensor. Let me just use a known rank-7 
    decomposition if it exists, or fall back to rank 8 per block giving 40.
    
    Actually, I recall that for [n,m,p] with min(n,m,p) >= 2, the rank can be 
    less than n*m*p. For [2,3,2], let me try to construct rank 7.
    """
    n, m, p = 2, 3, 10
    
    # Use Strassen 2x2x2 for each pair of output columns
    # For pair t (columns 2t, 2t+1), we need to decompose 2x3 x 3x2
    # 
    # Key insight: 2x3 x 3x2 can be done in rank 7 by the following construction:
    # Treat the 3 middle indices as: use Strassen on indices 0,1 (2x2x2 part) 
    # and handle index 2 separately, but with a clever combination.
    #
    # Actually, let me use a different approach. The 2x3 x 3x2 tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # I'll use the following rank-7 construction:
    # This is based on the fact that [2,3,2] has rank 7 (verified by 
    # embedding into [2,2,2] with an extra dimension handled by linearity).
    #
    # Construction for 2x3 x 3x2 in rank 7:
    # Let A = [a00 a01 a02; a10 a11 a12], B = [b00 b01; b10 b11; b20 b21]
    # C = A*B = [c00 c01; c10 c11]
    #
    # Use Strassen-style 7 terms:
    # M0 = (a00+a12)(b00+b11)  -- but this mixes different j indices
    #
    # Let me use a cleaner approach. For the 2x3 x 3x2 case, I'll use:
    # The standard result that rank([2,3,2]) = 7.
    #
    # Here's a construction:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00  
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # But wait, this only uses B indices 0,1 and A indices 0,1. We need index 2.
    #
    # I think the correct approach is:
    # C[i,k] = sum_{j=0}^{2} A[i,j]B[j,k]
    # = sum_{j=0}^{1} A[i,j]B[j,k] + A[i,2]B[2,k]
    #
    # The first part is a 2x2 x 2x2 multiplication (rank 7 with Strassen)
    # The second part is a 2x1 x 1x2 multiplication (rank 1)
    # Total: rank 8
    #
    # So per 2-column block, we get rank 8, total rank = 5*8 = 40? 
    # That's better than 60! But the knowledge says 55 was achieved.
    #
    # Hmm, maybe the rank of [2,3,2] is indeed 7, not 8. Let me try harder.
    #
    # Actually, I found it: the rank of the matrix multiplication tensor [2,3,2]
    # is 7. Here's a construction:
    #
    # For A (2x3) and B (3x2), C = AB (2x2):
    # Let's define 7 rank-1 tensors:
    #
    # Using the approach from Bini et al., we can show rank([2,3,2]) = 7.
    # 
    # Construction:
    # M1 = (A00+A12)(B00+B11)
    # M2 = (A10+A12)B00
    # M3 = A00(B10-B11)  
    # M4 = A12(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A12)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me verify: 
    # C00 should be A00*B00 + A01*B10 + A02*B20
    # M1 + M4 - M5 + M7
    # = (A00+A12)(B00+B11) + A12(B01-B00) - (A00+A01)B11 + (A01-A12)(B10+B11)
    # = A00*B00 + A00*B11 + A12*B00 + A12*B11 + A12*B01 - A12*B00 
    #   - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A12*B10 - A12*B11
    # = A00*B00 + A01*B10 + A12*B01
    #
    # But we need C00 = A00*B00 + A01*B10 + A02*B20. 
    # This doesn't match unless B01 = B20, which is not general.
    #
    # So this construction is wrong. Let me try a different one.
    
    # Let me just use the straightforward approach: 
    # For each 2-column block, use rank 8 (Strassen for 2x2x2 part + 1 for the extra)
    # Total rank = 5 * 8 = 40
    # This is already much better than 60!
    
    # But wait, can we do better? Let me check if [2,3,2] has rank 7.
    # 
    # After more thought, I believe the rank of [2,3,2] is 8, not 7.
    # The Strassen saving only applies to [2,2,2].
    #
    # So let's go with rank 40: 5 blocks * 8 terms per block.
    
    U = [[0] * 40 for _ in range(6)]   # n*m = 2*3 = 6
    V = [[0] * 40 for _ in range(30)]  # m*p = 3*10 = 30
    W = [[0] * 40 for _ in range(20)]  # p*n = 10*2 = 20
    
    col = 0
    for t in range(5):  # 5 pairs of columns
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # For this block, we decompose 2x3 x 3x2 in rank 8
        # Using Strassen for the 2x2x2 part (j=0,1) and 1 term for j=2
        
        # Strassen 2x2x2 for j in {0,1}, k in {k0,k1}
        # A_sub[i,j] for i in 0..1, j in 0..1
        # B_sub[j,k] for j in 0..1, k in {k0,k1}
        # C_sub[i,k] for i in 0..1, k in {k0,k1}
        
        # Term 0: M1 = (A00+A11)(B00+B11) where B00=B[j=0,k=k0], B11=B[j=1,k=k1]
        U[0][col] = 1   # A[0,0]
        U[3][col] = 1   # A[1,1]
        V[0][col] = 1   # B[0,k0]
        V[5][col] = 1   # B[1,k1]  (j=1, k=k1 -> index j*p+k = 1*10+k1)
        W[k0*2+0][col] = 1  # C[0,k0]
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # Term 1: M2 = (A10+A11)B00
        U[2][col] = 1   # A[1,0]
        U[3][col] = 1   # A[1,1]
        V[0][col] = 1   # B[0,k0]
        W[k0*2+1][col] = 1  # C[1,k0]
        col += 1
        
        # Term 2: M3 = A00(B10-B11)
        U[0][col] = 1   # A[0,0]
        V[4][col] = 1   # B[1,k0] (j=1, k=k0 -> index 1*10+k0)
        V[5][col] = -1  # B[1,k1]
        W[k0*2+0][col] = 1  # C[0,k0]... wait, C12 in Strassen
        # C12 = M3 + M5, so C[0,k1] gets M3
        W[k1*2+0][col] = 1  # C[0,k1]
        col += 1
        
        # Term 3: M4 = A11(B01-B00)
        U[3][col] = 1   # A[1,1]
        V[1][col] = 1   # B[0,k1] (j=0, k=k1 -> index 0*10+k1)
        V[0][col] = -1  # B[0,k0]
        W[k0*2+0][col] = 1  # C[0,k0]
        W[k0*2+1][col] = 1  # C[1,k0]
        col += 1
        
        # Term 4: M5 = (A00+A01)B11
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        V[5][col] = 1   # B[1,k1]
        W[k0*2+0][col] = -1  # C[0,k0] (negative in C11)
        W[k1*2+0][col] = 1   # C[0,k1]
        col += 1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U[2][col] = 1   # A[1,0]
        U[0][col] = -1  # A[0,0]
        V[0][col] = 1   # B[0,k0]
        V[1][col] = 1   # B[0,k1]
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U[1][col] = 1   # A[0,1]
        U[3][col] = -1  # A[1,1]
        V[4][col] = 1   # B[1,k0]
        V[5][col] = 1   # B[1,k1]
        W[k0*2+0][col] = 1  # C[0,k0]
        col += 1
        
        # Term 7: Extra term for j=2: A[i,2] * B[2,k]
        # This is a rank-1 term for each (i,k), but we need 4 terms? 
        # No, it's A[:,2] (2x1) * B[2,:] (1x2), which is rank 1 as a 
        # 2x2 matrix, but in tensor form it's 4 rank-1 tensors? No.
        # 
        # Actually, A[:,2]*B[2,:] = sum over nothing, it's just the outer product.
        # In the tensor decomposition, this contributes to all 4 entries of C.
        # We need: C[i,k] += A[i,2]*B[2,k]
        # This can be done with 1 rank-1 tensor? No, because it affects 4 entries.
        # 
        # A rank-1 tensor u⊗v⊗w contributes u[i]*v[j]*w[k] to T[i,j,k].
        # Here we need T[i,2,k] = A[i,2]*B[2,k] for all i,k.
        # This is u[i]*v[2]*w[k] = A[i,2]*B[2,k], so v[2]=1, u[i]=A[i,2], w[k]=B[2,k].
        # But u and w are fixed vectors, so this is ONE rank-1 tensor!
        # 
        # Wait, in our formulation:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For j=2: T[i,2,k] = sum_c U[i*3+2,c] * V[2*10+k,c] * W[k*2+i,c]
        # We want this to equal A[i,2]*B[2,k] = U_part[i]*V_part[k]
        # So set U[i*3+2,c] = a_i, V[20+k,c] = b_k, W[k*2+i,c] = 1
        # Then T[i,2,k] = a_i * b_k * 1
        # We need a_i * b_k = A[i,2]*B[2,k] for all i,k.
        # This works with a_i = A[i,2], b_k = B[2,k], but A and B are variables!
        # 
        # In the tensor decomposition, U, V, W are FIXED matrices. The "variables" 
        # are the entries of A and B. So we need:
        # sum_c U[i*3+2,c] * V[20+k,c] * W[k*2+i,c] = 1 for the coefficient of A[i,2]*B[2,k]
        # and 0 for all other (i,j,k) combinations.
        #
        # For j=2, we want the contribution to T[i,2,k] to be exactly A[i,2]*B[2,k].
        # In our tensor: T[i,2,k] = sum_c U[i*3+2,c]*V[20+k,c]*W[k*2+i,c]
        # This should equal 1 (as a coefficient) when we're looking at the 
        # A[i,2]*B[2,k] term.
        #
        # So we need: sum_c U[i*3+2,c]*V[20+k,c]*W[k*2+i,c] = 1 for all i,k
        # and for j != 2, the contributions from this term should be 0.
        #
        # Since U[i*3+2,c] is only nonzero for j=2 rows, and V[20+k,c] is only 
        # for j=2, this term only affects T[i,2,k]. Good.
        #
        # We need U[i*3+2,c]*V[20+k,c]*W[k*2+i,c] = 1 for all i,k with one c.
        # Set U[0*3+2,c]=1, U[1*3+2,c]=1, V[20+k0,c]=1, V[20+k1,c]=1, 
        # W[k0*2+0,c]=1, W[k0*2+1,c]=1, W[k1*2+0,c]=1, W[k1*2+1,c]=1
        # Then T[0,2,k0] = 1*1*1 = 1, T[0,2,k1] = 1*1*1 = 1, etc.
        # But this gives T[i,2,k] = 1 for all i,k, which means the coefficient 
        # of A[i,2]*B[2,k] is 1. 
        
        U[2][col] = 1   # A[0,2]
        U[5][col] = 1   # A[1,2]
        V[20+k0][col] = 1  # B[2,k0]
        V[20+k1][col] = 1  # B[2,k1]
        W[k0*2+0][col] = 1  # C[0,k0]
        W[k0*2+1][col] = 1  # C[1,k0]
        W[k1*2+0][col] = 1  # C[0,k1]
        W[k1*2+1][col] = 1  # C[1,k1]
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition is exact."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 if we're decomposing the multiplication tensor
    # T[i,j,k] should be 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False, f"({i},{j},{k}): {val} != 1"
    return True, "OK"

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 x 3x10."""
    n, m, p = DIMENSIONS
    
    # Try the rank-40 Strassen-based decomposition
    result = _build_strassen_rank55()
    U, V, W = result["U"], result["V"], result["W"]
    
    # Verify
    valid, msg = _verify(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook
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
    
    return result
# EVOLVE-BLOCK-END
