"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 6x6 matrix multiplication.
    
    We use a block-Strassen approach: treat the 6x6 matrices as 3x3 blocks of 2x2 submatrices,
    and apply Strassen's 7-product algorithm to each 2x2 block multiplication, then combine.
    
    The key insight: if we decompose the 6x6 multiplication into 9 block positions, each requiring
    a 2x2 matrix product, we can use Strassen's algorithm (rank 7 instead of 8) for each 2x2 product.
    
    However, a more direct approach: use the known result that the border rank of 6x6 matrix 
    multiplication is at most 189 (from applying Strassen to 2x2 blocks in a 3x3 grid).
    
    Actually, let me think more carefully. The standard schoolbook is 6*6*6 = 216.
    
    Using Strassen on 2x2 blocks: divide 6x6 into 3x3 blocks of size 2x2.
    For each of the 9 output block positions, we need to compute a sum of products of 2x2 blocks.
    
    The tensor for 6x6 matmul can be seen as: 
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    If we group indices into blocks of size 2:
    Let i = 2*a + a', j = 2*b + b', k = 2*c + c' where a,b,c in {0,1,2} and a',b',c' in {0,1}.
    
    Then C[2a+a', 2b+b'] = sum_{c=0}^{2} sum_{c'=0}^{1} A[2a+a', 2c+c'] * B[2c+c', 2b+b']
    
    This is a sum over 6 terms of products of 2x2 matrix elements.
    
    A cleaner approach: use the Kronecker product structure.
    The 6x6 matmul tensor T_{6,6,6} can be related to T_{2,2,2} ⊗ T_{3,3,3} in some sense,
    but that's not quite right either.
    
    Let me use a different strategy: direct construction using Strassen-like ideas.
    
    Actually, the most practical approach given the time constraint: 
    1. Start with the schoolbook decomposition (rank 216)
    2. Try to find a lower-rank decomposition using random initialization and gradient-free optimization
    3. Or use a known algebraic construction
    
    Let me try a block-Strassen approach more carefully:
    
    For 6x6 = (2x3)x(2x3)x(2x3) in terms of block sizes... no, let's think of it as 3x3 blocks of 2x2.
    
    The 6x6 matmul tensor has structure:
    T = sum_{i,j,k=0}^{5} e_i ⊗ e_k ⊗ e_{k,n} ⊗ e_{j,p} ... 
    
    Let me just implement a search-based approach with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 6, 6, 6
    
    # First, get the schoolbook decomposition as baseline
    rank_school = n * m * p  # 216
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Convert to lists for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try block-Strassen: divide into 3x3 blocks of 2x2
    # For each 2x2 block multiplication, Strassen uses 7 multiplications instead of 8
    # Total: 9 blocks * 7 = 63 "Strassen products", but each Strassen product involves
    # additions/subtractions of 2x2 blocks, so the actual rank contribution needs careful accounting.
    
    # Actually, let me think about this differently.
    # The matmul tensor for n x n matrices has rank r(n).
    # r(1) = 1, r(2) = 7 (Strassen), r(4) = 49 (Strassen^2), etc.
    # For n=6, we can write 6 = 4 + 2, so r(6) <= r(4) + r(2) + ... but that's not how it works.
    
    # Better: 6 = 2*3, so we can use the fact that T_{6,6,6} embeds in T_{2,2,2} ⊗ T_{3,3,3}
    # The rank of a Kronecker product of tensors is at most the product of ranks.
    # r(T_{2,2,2}) = 7, r(T_{3,3,3}) <= 27 (schoolbook) or possibly lower.
    # So r(T_{6,6,6}) <= 7 * 27 = 189.
    
    # Let's construct this: T_{6,6,6} as a sub-tensor of T_{2,2,2} ⊗ T_{3,3,3}
    
    # T_{2,2,2}: indices (i,k,j) for i,k,j in {0,1}
    # T_{3,3,3}: indices (a,c,b) for a,c,b in {0,1,2}
    # T_{6,6,6}: index i = 2*a + i', k = 2*c + k', j = 2*b + j'
    # where a,c,b in {0,1,2} and i',k',j' in {0,1}
    
    # The tensor element T_{i,j,k} = 1 if (i,k) is an input pair and j is output, else 0
    # Actually for matmul: T_{(i,k), j} ... let me be more careful.
    
    # Matmul tensor: C = A * B, C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor T has indices (i, k, j) with T[i,k,j] = 1 (it's a 0-1 tensor)
    # U has rows indexed by (i,k), V by (k,j), W by (j,i) [or per the convention W[k*n+i,r]]
    
    # Per the task: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U[i*m + k, r], V[k*p + j, r], W[j*n + i, r]
    # And the decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = T[i,k,j]
    
    # For the Kronecker product approach:
    # T_{6,6,6}[i,k,j] where i=2a+i', k=2c+k', j=2b+j'
    # = T_{2,2,2}[i',k',j'] * T_{3,3,3}[a,c,b]
    # This is true because the matmul tensor factors this way!
    
    # So we can use the Kronecker product of a rank-7 decomposition of T_{2,2,2}
    # and a rank-27 decomposition of T_{3,3,3} to get rank 7*27 = 189.
    
    # Strassen's 7-term decomposition of 2x2 matmul:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For T_{2,2,2}, the tensor elements are T[i,k,j] = 1 for all i,k,j in {0,1}
    # U_{2x2}[i*2+k, r], V_{2x2}[k*2+j, r], W_{2x2}[j*2+i, r]
    
    # Let me construct the Strassen decomposition for 2x2:
    # The 7 products correspond to:
    # r=0: M1: (a00+a11)*(b00+b11)
    #   U: rows (0,0),(1,1) get 1; V: rows (0,0),(0,1) get 1; W: ...
    #   Actually, U[i*2+k, r] contributes to A[i,k]
    #   For M1 = (A[0,0]+A[1,1])*(B[0,0]+B[0,1]):
    #   U[0*2+0, 0] = 1, U[1*2+1, 0] = 1  (i=0,k=0 and i=1,k=1)
    #   V[0*2+0, 0] = 1, V[0*2+1, 0] = 1  (k=0,j=0 and k=0,j=1)
    #   W[0*2+0, 0] = 1, W[1*2+1, 0] = 1  (j=0,i=0 and j=1,i=1)
    #   But wait, W[j*2+i, r] should give the coefficient for C[i,j]
    #   M1 contributes to C[0,0] and C[1,1] with +1, and C[0,1] and C[1,0] with -1
    #   Actually no, M1 is just a product, and its contribution to each C element is determined by the W factor.
    
    # Let me be more systematic. For the 2x2 matmul tensor:
    # T[i,k,j] = 1 for all i,k,j in {0,1}
    # We need: sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r] = 1 for all i,k,j
    
    # Strassen gives us 7 terms. Let me define them:
    
    # Term 0: M1 = (A00+A11)(B00+B01)
    #   U factor: A00 + A11 => U[0,0]=1, U[3,0]=1 (indices: i*2+k, so (0,0)->0, (1,1)->3)
    #   V factor: B00 + B01 => V[0,0]=1, V[1,0]=1 (indices: k*2+j, so (0,0)->0, (0,1)->1)
    #   W factor: contributes to C00 (+1), C01 (-1), C10 (-1), C11 (+1)
    #   W[j*2+i, 0]: (0,0)->0: +1, (1,0)->1: -1, (0,1)->2: -1, (1,1)->3: +1
    
    # Term 1: M2 = (A10+A11)B00
    #   U: A10 + A11 => U[2,1]=1, U[3,1]=1 ((1,0)->2, (1,1)->3)
    #   V: B00 => V[0,1]=1
    #   W: contributes to C10 (+1), C11 (-1)
    #   W[0,1]=0, W[1,1]=1, W[2,1]=0, W[3,1]=-1  (j*2+i: (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3)
    #   Wait, C10 means i=1,j=0, so j*2+i = 0*2+1 = 1. C11 means i=1,j=1, so j*2+i = 1*2+1 = 3.
    #   W[1,1] = 1, W[3,1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    #   U: A00 => U[0,2]=1
    #   V: B10 - B11 => V[2,2]=1, V[3,2]=-1 ((1,0)->2, (1,1)->3)
    #   W: contributes to C01 (+1), C11 (+1)
    #   C01: i=0,j=1, j*2+i=1*2+0=2. C11: i=1,j=1, j*2+i=1*2+1=3.
    #   W[2,2]=1, W[3,2]=1
    
    # Term 3: M4 = A11(B01-B00)
    #   U: A11 => U[3,3]=1
    #   V: B01 - B00 => V[1,3]=1, V[0,3]=-1
    #   W: contributes to C00 (+1), C10 (+1)
    #   C00: j*2+i=0. C10: j*2+i=1.
    #   W[0,3]=1, W[1,3]=1
    
    # Term 4: M5 = (A00+A01)B11
    #   U: A00 + A01 => U[0,4]=1, U[1,4]=1
    #   V: B11 => V[3,4]=1
    #   W: contributes to C00 (-1), C11 (-1)
    #   W[0,4]=-1, W[3,4]=-1
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    #   U: A10 - A00 => U[2,5]=1, U[0,5]=-1
    #   V: B00 + B10 => V[0,5]=1, V[2,5]=1
    #   W: contributes to C10 (+1)
    #   W[1,5]=1
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    #   U: A01 - A11 => U[1,6]=1, U[3,6]=-1
    #   V: B01 + B11 => V[1,6]=1, V[3,6]=1
    #   W: contributes to C00 (-1), C01 (+1), C11 (+1)
    #   W[0,6]=-1, W[2,6]=1, W[3,6]=1
    
    # Let me verify term 0 (M1):
    # sum over i,k,j of U[i*2+k,0]*V[k*2+j,0]*W[j*2+i,0]
    # Non-zero U: (0,0), (1,1). Non-zero V: (0,0), (0,1). Non-zero W: (0,0)=1, (1,0)=-1, (0,1)=-1, (1,1)=1
    # For i=0,k=0: U[0,0]=1. V[0*2+j,0]: j=0: V[0,0]=1, j=1: V[1,0]=1
    #   j=0: W[0*2+0,0]=W[0,0]=1. Product: 1*1*1=1. T[0,0,0] gets +1.
    #   j=1: W[1*2+0,0]=W[2,0]=-1. Product: 1*1*(-1)=-1. T[0,0,1] gets -1.
    # For i=0,k=1: U[1,0]=0. Skip.
    # For i=1,k=0: U[2,0]=0. Skip.
    # For i=1,k=1: U[3,0]=1. V[1*2+j,0]: j=0: V[2,0]=0, j=1: V[3,0]=0. All zero. Skip.
    # So M1 contributes: T[0,0,0]+=1, T[0,0,1]+=-1.
    
    # This is getting complex. Let me just build the full 6x6 decomposition using the Kronecker product.
    
    # For T_{3,3,3}, use the schoolbook decomposition (rank 27).
    # U3[a*3+c, r3] = 1 if (a,c) matches, else 0
    # V3[c*3+b, r3] = 1 if (c,b) matches, else 0
    # W3[b*3+a, r3] = 1 if (b,a) matches, else 0
    
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    r3 = 0
    for a in range(3):
        for c in range(3):
            for b in range(3):
                U3[a*3+c, r3] = 1
                V3[c*3+b, r3] = 1
                W3[b*3+a, r3] = 1
                r3 += 1
    
    # For T_{2,2,2}, use Strassen (rank 7).
    # Let me construct it properly by building the U2, V2, W2 matrices.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = -1; W2[2, 0] = -1; W2[3, 0] = 1
    
    # Term 1: M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # Term 2: M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M4
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 4: M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[3, 4] = -1
    
    # Term 5: M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[1, 5] = 1
    
    # Term 6: M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[2, 6] = 1; W2[3, 6] = 1
    
    # Now form the Kronecker product decomposition for 6x6:
    # U6[(2a+i')*6 + (2c+k'), r] = U2[i'*2+k', r2] * U3[a*3+c, r3]
    # where r = r2 * 27 + r3
    # Similarly for V and W.
    
    rank_kron = 7 * 27  # 189
    U6 = np.zeros((36, rank_kron))
    V6 = np.zeros((36, rank_kron))
    W6 = np.zeros((36, rank_kron))
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for a in range(3):
                for i in range(2):
                    for c in range(3):
                        for k in range(2):
                            idx_U6 = (2*a + i) * 6 + (2*c + k)
                            val = U2[i*2+k, r2] * U3[a*3+c, r3]
                            if val != 0:
                                U6[idx_U6, r] = val
            
            for c in range(3):
                for k in range(2):
                    for b in range(3):
                        for j in range(2):
                            idx_V6 = (2*c + k) * 6 + (2*b + j)
                            val = V2[k*2+j, r2] * V3[c*3+b, r3]
                            if val != 0:
                                V6[idx_V6, r] = val
            
            for b in range(3):
                for j in range(2):
                    for a in range(3):
                        for i in range(2):
                            idx_W6 = (2*b + j) * 6 + (2*a + i)
                            val = W2[j*2+i, r2] * W3[b*3+a, r3]
                            if val != 0:
                                W6[idx_W6, r] = val
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Check that sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1 for all i,k,j."""
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, k, j, s)
        return True, None
    
    valid, info = verify(U6, V6, W6, 6, 6, 6)
    
    if valid:
        best_U, best_V, best_W = U6, V6, W6
        best_rank = rank_kron
    
    # Check deadline
    if time.time() > deadline:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Try to improve further if time permits
    # For now, return the Kronecker product result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
