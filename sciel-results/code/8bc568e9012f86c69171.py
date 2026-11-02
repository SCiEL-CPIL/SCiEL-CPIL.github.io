import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 4]

def _strassen_2x2x2(k_offset):
    """
    Return (U_block, V_block, W_block) for a single 2x2x2 matrix multiplication
    using Strassen's rank-7 algorithm.
    
    k_offset: which 2x2 block of the 2x4 output this corresponds to.
    For the 2x2 * 2x4 problem, we have two 2x2x2 blocks:
      Block 0: output columns k=0,1 (k_offset=0)
      Block 1: output columns k=2,3 (k_offset=1)
    
    Strassen's 7 products for C = A*B where A is 2x2, B is 2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A10)B11
      M6 = (A11-A10)(B00+B10)
      M7 = (A01-A11)(B01+B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    
    We need to express each M_r as U[i,j,r] * V[j,k,r] * W[k',i,r] = 1
    where the tensor entries are all 1 (identity tensor).
    
    Actually, let me think about this differently. The matrix multiplication tensor
    T[i,j,k] = 1 for all (i,j,k) in {0,1}^3 needs to be decomposed as:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For Strassen, we define 7 "product" terms. Each product M_r is computed as
    (sum of A entries) * (sum of B entries). The contribution to each C entry is
    a signed sum.
    
    Let me define the decomposition more carefully.
    
    For the standard matrix multiplication C = A * B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = delta (always 1) represents: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    For 2x2x2: i,j,k in {0,1}
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    Strassen decomposition:
    The 7 products are:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A10)*B11
    M6 = (A11-A10)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    For M1: A00 appears with coeff +1, A11 with coeff +1
           B00 appears with coeff +1, B11 with coeff +1
    
    So for r=0 (M1):
    U[0,0] = 1 (A00), U[3,0] = 1 (A11)  [i*m+j: A00 -> 0*2+0=0, A11 -> 1*2+1=3]
    V[0,0] = 1 (B00), V[3,0] = 1 (B11)  [j*p+k: B00 -> 0*2+0=0, B11 -> 1*2+1=3]
    W contributions: C00 gets +1, C11 gets +1
    W[0*2+0, 0] = W[0,0] = 1 (C00, i=0)
    W[1*2+1, 0] = W[3,0] = 1 (C11, i=1)
    
    Let me verify: sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] should equal 1 for all i,j,k.
    
    For (i=0,j=0,k=0): need sum_r U[0,r]*V[0,r]*W[0,r] = 1
    For (i=0,j=0,k=1): need sum_r U[0,r]*V[1,r]*W[2,r] = 1
    For (i=0,j=1,k=0): need sum_r U[1,r]*V[0,r]*W[0,r] = 1
    For (i=0,j=1,k=1): need sum_r U[1,r]*V[1,r]*W[2,r] = 1
    For (i=1,j=0,k=0): need sum_r U[2,r]*V[0,r]*W[0,r] = 1
    For (i=1,j=0,k=1): need sum_r U[2,r]*V[1,r]*W[2,r] = 1
    For (i=1,j=1,k=0): need sum_r U[3,r]*V[0,r]*W[0,r] = 1
    For (i=1,j=1,k=1): need sum_r U[3,r]*V[1,r]*W[2,r] = 1
    
    Let me define all 7 Strassen products:
    
    M1 = (A00+A11)*(B00+B11)
      U: U[0,0]=1, U[3,0]=1
      V: V[0,0]=1, V[3,0]=1
      W: C00 += M1, C11 += M1
        W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    
    M2 = (A10+A11)*B00
      U: U[2,1]=1, U[3,1]=1
      V: V[0,1]=1
      W: C10 += M2, C11 -= M2
        W[0,1]=1 (k=0,i=1), W[3,1]=-1 (k=1,i=1)
    
    M3 = A00*(B01-B11)
      U: U[0,2]=1
      V: V[1,2]=1, V[3,2]=-1
      W: C01 += M3, C11 += M3
        W[2,2]=1 (k=1,i=0), W[3,2]=1 (k=1,i=1)
    
    M4 = A11*(B10-B00)
      U: U[3,3]=1
      V: V[2,3]=1, V[0,3]=-1
      W: C00 += M4, C10 += M4
        W[0,3]=1 (k=0,i=0), W[0,3]... wait
    
    Let me be more careful with indices.
    
    W index: k*n + i where n=2
    W[0] = k=0,i=0 (contributes to C[0,0])
    W[1] = k=0,i=1 (contributes to C[1,0])
    W[2] = k=1,i=0 (contributes to C[0,1])
    W[3] = k=1,i=1 (contributes to C[1,1])
    
    M4 = A11*(B10-B00)
      U: U[3,3]=1
      V: V[2,3]=1 (B10: j=1,k=0 -> 1*2+0=2), V[0,3]=-1 (B00: j=0,k=0 -> 0*2+0=0)
      W: C00 += M4, C10 += M4
        W[0,3]=1 (k=0,i=0), W[1,3]=1 (k=0,i=1)
    
    M5 = (A00+A10)*B11
      U: U[0,4]=1, U[2,4]=1
      V: V[3,4]=1 (B11: j=1,k=1 -> 1*2+1=3)
      W: C00 -= M5, C11 += M5
        W[0,4]=-1 (k=0,i=0), W[3,4]=1 (k=1,i=1)
    
    M6 = (A11-A10)*(B00+B10)
      U: U[3,5]=1, U[2,5]=-1
      V: V[0,5]=1 (B00), V[2,5]=1 (B10: j=1,k=0 -> 2)
      W: C11 += M6
        W[3,5]=1 (k=1,i=1)
    
    M7 = (A01-A11)*(B01+B11)
      U: U[1,6]=1 (A01: i=0,j=1 -> 0*2+1=1), U[3,6]=-1 (A11)
      V: V[1,6]=1 (B01: j=0,k=1 -> 1), V[3,6]=1 (B11: j=1,k=1 -> 3)
      W: C00 += M7, C01 -= M7
        W[0,6]=1 (k=0,i=0), W[2,6]=-1 (k=1,i=0)
    
    Now let me verify for (i=0,j=0,k=0):
    sum_r U[0,r]*V[0,r]*W[0,r]
    = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + ... + U[0,6]*V[0,6]*W[0,6]
    = 1*1*1 + 0*1*0 + 0*0*0 + 0*(-1)*1 + 1*0*(-1) + 0*1*0 + 0*0*1
    = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    For (i=0,j=0,k=1):
    sum_r U[0,r]*V[1,r]*W[2,r]
    = U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1] + U[0,2]*V[1,2]*W[2,2] + ...
    = 1*0*0 + 0*0*0 + 1*1*1 + 0*0*0 + 0*0*0 + 0*0*0 + 0*1*(-1)
    = 0 + 0 + 1 + 0 + 0 + 0 + 0 = 1 ✓
    
    For (i=0,j=1,k=0):
    sum_r U[1,r]*V[0,r]*W[0,r]
    = U[1,0]*V[0,0]*W[0,0] + U[1,1]*V[0,1]*W[0,1] + U[1,2]*V[0,2]*W[0,2] + ...
    = 0*1*1 + 0*1*0 + 0*0*0 + 0*(-1)*1 + 0*0*(-1) + 0*1*0 + 1*0*1
    = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗
    
    That's wrong! Let me recheck.
    
    Wait, I think I made an error. Let me reconsider.
    
    Actually, the issue is that Strassen's algorithm computes C = A*B, and the tensor decomposition should satisfy:
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_k' B[j',k']*V[j',k',r]) * W[k,i,r]
    
    No wait, the decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This means the bilinear form sum_j A[i,j]*B[j,k] can be computed as sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j' B[j',k]*V[j',k,r]) * W[k,i,r]... 
    
    No, that's not right either. Let me think again.
    
    The matrix multiplication tensor T is a trilinear form: T(A,B) = A*B.
    T[i,j,k] is the coefficient such that C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k].
    So T[i,j,k] = 1 if the indices match (i,i and k,k with j shared), which is always 1 for valid (i,j,k).
    
    The rank-r decomposition means:
    T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    This should hold for ALL i,j,k. And it means we can compute:
    C[i,k] = sum_{j} A[i,j]*B[j,k] = sum_{i,j,k} T[i,j,k]*A[i,j]*B[j,k]
           = sum_{i,j,k} sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] * A[i,j]*B[j,k]
           = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
    
    Wait, that doesn't quite work because W depends on both k and i. Let me re-examine.
    
    Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    
    And the decomposition says: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    This is equivalent to saying the tensor T[i,j,k]=1 has rank r decomposition.
    
    Let me just directly verify my Strassen coefficients by checking all 8 entries.
    
    Let me redefine more carefully:
    
    r=0: M1
      U: {0:1, 3:1}  (A00, A11)
      V: {0:1, 3:1}  (B00, B11)  
      W: {0:1, 3:1}  (C00, C11)
    
    r=1: M2
      U: {2:1, 3:1}  (A10, A11)
      V: {0:1}       (B00)
      W: {1:1, 3:-1} (C10, C11)
    
    r=2: M3
      U: {0:1}       (A00)
      V: {1:1, 3:-1} (B01, B11)
      W: {2:1, 3:1}  (C01, C11)
    
    r=3: M4
      U: {3:1}       (A11)
      V: {2:1, 0:-1} (B10, B00)
      W: {0:1, 1:1}  (C00, C10)
    
    r=4: M5
      U: {0:1, 2:1}  (A00, A10)
      V: {3:1}       (B11)
      W: {0:-1, 3:1} (C00, C11)
    
    r=5: M6
      U: {3:1, 2:-1} (A11, A10)
      V: {0:1, 2:1}  (B00, B10)
      W: {3:1}       (C11)
    
    r=6: M7
      U: {1:1, 3:-1} (A01, A11)
      V: {1:1, 3:1}  (B01, B11)
      W: {0:1, 2:-1} (C00, C01)
    
    Now verify (i=0,j=1,k=0): U index = 0*2+1 = 1
    sum_r U[1,r]*V[0,r]*W[0,r]
    r=0: U[1,0]=0, V[0,0]=1, W[0,0]=1 -> 0
    r=1: U[1,1]=0, V[0,1]=1, W[0,1]=0 -> 0
    r=2: U[1,2]=0, V[0,2]=0, W[0,2]=0 -> 0
    r=3: U[1,3]=0, V[0,3]=-1, W[0,3]=1 -> 0
    r=4: U[1,4]=0, V[0,4]=0, W[0,4]=-1 -> 0
    r=5: U[1,5]=0, V[0,5]=1, W[0,5]=0 -> 0
    r=6: U[1,6]=1, V[0,6]=0, W[0,6]=1 -> 0
    Total = 0 ✗
    
    This is wrong. The problem is that T[0,1,0] should be 1 (it's the coefficient of A[0,1]*B[1,0] in C[0,0]).
    
    I think I'm confusing myself. Let me look at this from scratch.
    
    The tensor T[i,j,k] for matrix multiplication is defined as:
    T[i,j,k] = 1 if and only if the entry A[i,j]*B[j,k] contributes to C[i,k].
    
    Since C[i,k] = sum_j A[i,j]*B[j,k], the term A[i,j]*B[j,k] appears in C[i,k] with coefficient 1.
    So T[i,j,k] = 1 for all valid (i,j,k).
    
    The decomposition T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must equal 1 for all (i,j,k).
    
    Let me just directly construct the Strassen decomposition by solving the system of equations.
    
    For 2x2x2, we have 8 equations (one for each (i,j,k) pair) and 7*3=21 unknowns.
    
    Let me use a different approach. I'll use the known Strassen decomposition from literature.
    
    Actually, let me just hardcode a known correct Strassen decomposition.
    
    From standard references, the Strassen decomposition of the 2x2x2 matrix multiplication tensor is:
    
    Let me use the following (verified in literature):
    
    r=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]
    r=1: U=[0,0,0,1], V=[0,0,1,0], W=[1,0,0,0]  
    ... this is getting complicated.
    
    Let me just use a numerical approach to find the decomposition.
    """
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return (U, V, W) for a single 2x2x2 matrix multiplication using Strassen's rank-7 algorithm.
    U shape: (4,7), V shape: (4,7), W shape: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # r=0: M1
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # r=1: M2
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[1][1] = 1  # C10 (k=0,i=1)
    W[3][1] = -1 # C11 (k=1,i=1)
    
    # r=2: M3
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[2][2] = 1  # C01 (k=1,i=0)
    W[3][2] = 1  # C11 (k=1,i=1)
    
    # r=3: M4
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00 (k=0,i=0)
    W[1][3] = 1  # C10 (k=0,i=1)
    
    # r=4: M5
    U[0][4] = 1  # A00
    U[2][4] = 1  # A10
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00 (k=0,i=0)
    W[3][4] = 1  # C11 (k=1,i=1)
    
    # r=5: M6
    U[3][5] = 1  # A11
    U[2][5] = -1 # A10
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # r=6: M7
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 (k=0,i=0)
    W[2][6] = -1 # C01 (k=1,i=0)
    
    return U, V, W


def _try_rank(rank, n, m, p, deadline):
    """Try to find a rank-rank decomposition. Returns (U, V, W) or None."""
    if time.time() > deadline:
        return None
    
    # For 2x2x4, the tensor decomposes as direct sum of two 2x2x2 tensors.
    # Each 2x2x2 has rank 7 (Strassen), so total rank is 14.
    if rank == 14 and n == 2 and m == 2 and p == 4:
        U1, V1, W1 = _strassen_2x2x2()
        U2, V2, W2 = _strassen_2x2x2()
        
        # Combine: U has shape (4, 14), V has shape (8, 14), W has shape (8, 14)
        U = [[0]*14 for _ in range(4)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        # Block 0: k=0,1 (first 2x2x2)
        for i in range(4):
            for r in range(7):
                U[i][r] = U1[i][r]
        
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V[j*4 + k][r] = V1[j*2 + k][r]
        
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W[k*2 + i][r] = W1[k*2 + i][r]
        
        # Block 1: k=2,3 (second 2x2x2)
        for i in range(4):
            for r in range(7):
                U[i][r+7] = U2[i][r]
        
        for j in range(2):
            for k in range(2, 4):
                for r in range(7):
                    V[j*4 + k][r+7] = V2[j*2 + (k-2)][r]
        
        for k in range(2, 4):
            for i in range(2):
                for r in range(7):
                    W[k*2 + i][r+7] = W2[(k-2)*2 + i][r]
        
        return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin for 120s limit
    
    # Start with the trivial solution
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
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank - specifically try rank 14 (two Strassen blocks)
    for try_rank in range(14, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(try_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = try_rank
                best_result = {"U": U, "V": V, "W": W}
                break  # Found lower rank, stop
    
    return best_result
