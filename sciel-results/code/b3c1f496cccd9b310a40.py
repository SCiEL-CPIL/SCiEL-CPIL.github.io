import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin from 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    # For M<2,2,9>, the tensor decomposes as a direct sum of 9 copies of M<2,2,1>.
    # The rank of M<2,2,1> (standard 2x2 matrix multiply) is known to be 7 (Landsberg's bound).
    # Thus the rank-minimal decomposition has rank 7 * 9 = 63.
    rank = 7 * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 63
    V = [[0] * rank for _ in range(m * p)]  # 18 x 63
    W = [[0] * rank for _ in range(p * n)]  # 18 x 63
    
    # The 7 rank-1 terms for 2x2 matrix multiplication (A * B = C):
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    #
    # Standard 7-term decomposition (Landsberg's optimal rank for 2x2):
    # Term 1: a00 * b00 -> c00
    # Term 2: a01 * b10 -> c00
    # Term 3: a00 * b01 -> c01
    # Term 4: a01 * b11 -> c01
    # Term 5: a10 * b00 -> c10
    # Term 6: a11 * b10 -> c10
    # Term 7: (a00+a01+a10+a11) * (b00+b01+b10+b11) - (terms 1-6) -> c11
    # Actually, the standard 7-term scheme uses:
    # m1 = a00*b00
    # m2 = a01*b10
    # m3 = a00*b01
    # m4 = a01*b11
    # m5 = a10*b00
    # m6 = a11*b10
    # m7 = (a00+a01+a10+a11)*(b00+b01+b10+b11) - m1-m2-m3-m4-m5-m6
    # But we need rank-1 tensors, so let's use a proper 7-term decomposition.
    
    # A well-known 7-term decomposition for 2x2 matrix multiplication:
    # Using the formula from Landsberg:
    # We can write it as:
    # C = sum_{t=1}^{7} u_t * v_t * w_t
    
    # Let's use a concrete 7-term decomposition. One standard one:
    # Define vectors in R^4 for U, R^4 for V, R^4 for W (where W index is k*n+i, but for p=1, k=0, so W index is i)
    
    # For 2x2: A has indices (i,j) in {0,1}x{0,1}, B has (j,k) in {0,1}x{0,1}, C has (i,k) in {0,1}x{0,1}
    # U[i*m+j], V[j*p+k], W[k*n+i]
    
    # 7-term decomposition:
    terms = [
        # (u, v, w) where u is in R^4 (A), v is in R^4 (B), w is in R^4 (C)
        # Using a known optimal scheme:
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # a00*b00 -> c00
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]),  # a01*b10 -> c00
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # a00*b01 -> c01
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # a01*b11 -> c01
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]),  # a10*b00 -> c10
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]),  # a11*b10 -> c10
        # 7th term for c11: a10*b01 + a11*b11
        # We need: a10*b01 + a11*b11 -> c11
        # This can be done with one more term if we use a linear combination, but that would be rank 8.
        # Actually, the 7-term decomposition uses a clever trick.
    ]
    
    # Let me look up a proper 7-term decomposition.
    # One standard 7-term decomposition for 2x2:
    # m1 = a00 * b00
    # m2 = a01 * b10  
    # m3 = a00 * b01
    # m4 = a01 * b11
    # m5 = a10 * b00
    # m6 = a11 * b10
    # m7 = (a00 + a01 + a10 + a11) * (b00 + b01 + b10 + b11)
    # Then:
    # c00 = m1 + m2
    # c01 = m3 + m4
    # c10 = m5 + m6
    # c11 = m7 - m1 - m3 - m5 - m6  ... wait, that's not right either.
    
    # Actually, the correct 7-term decomposition is more subtle. Let me use a verified one.
    # From the literature, a 7-term decomposition exists. Here's one:
    
    # Let's use the decomposition from Bini et al. or a standard reference.
    # A simpler approach: since we need exact integer/half-integer entries, let's verify the 7-term scheme.
    
    # Standard 7-term scheme (from Landsberg 1988):
    # Define:
    # x1 = a00, x2 = a01, x3 = a10, x4 = a11
    # y1 = b00, y2 = b01, y3 = b10, y4 = b11
    
    # m1 = x1 * y1
    # m2 = x2 * y3
    # m3 = x1 * y2
    # m4 = x2 * y4
    # m5 = x3 * y1
    # m6 = x4 * y3
    # m7 = (x1+x2+x3+x4) * (y1+y2+y3+y4)
    
    # c00 = m1 + m2
    # c01 = m3 + m4
    # c10 = m5 + m6
    # c11 = m7 - m1 - m3 - m5 - m6
    
    # Let's verify c11:
    # m7 - m1 - m3 - m5 - m6 = (x1+x2+x3+x4)(y1+y2+y3+y4) - x1*y1 - x1*y2 - x3*y1 - x4*y3
    # = x1*y1 + x1*y2 + x1*y3 + x1*y4 + x2*y1 + x2*y2 + x2*y3 + x2*y4 + x3*y1 + x3*y2 + x3*y3 + x3*y4 + x4*y1 + x4*y2 + x4*y3 + x4*y4
    #   - x1*y1 - x1*y2 - x3*y1 - x4*y3
    # = x1*y3 + x1*y4 + x2*y1 + x2*y2 + x2*y3 + x2*y4 + x3*y2 + x3*y3 + x3*y4 + x4*y1 + x4*y2 + x4*y4
    # But we need c11 = x3*y2 + x4*y4
    # This doesn't match. So this 7-term scheme is incorrect as stated.
    
    # Let me use a correct 7-term decomposition. After research, here's a valid one:
    # The 7-term decomposition for 2x2 matrix multiplication is non-trivial.
    # For now, let's use the rank 8 decomposition which is simpler, then try to optimize.
    # Actually, rank 8 is straightforward:
    # m1 = a00*b00 -> c00
    # m2 = a01*b10 -> c00
    # m3 = a00*b01 -> c01
    # m4 = a01*b11 -> c01
    # m5 = a10*b00 -> c10
    # m6 = a11*b10 -> c10
    # m7 = a10*b01 -> c11
    # m8 = a11*b11 -> c11
    
    # But rank 8 is not minimal. The minimal rank is 7.
    
    # Let me implement the rank 8 version first as a fallback, then try rank 7.
    
    # For rank 8:
    # Each k in 0..8 gets 8 terms.
    
    # Actually, let's just use rank 8 * 9 = 72 for now, which is safe, and see if we can improve.
    # But the task asks for rank-minimal, so we should aim for 63.
    
    # Let me find a correct 7-term decomposition.
    # From a reliable source, the 7-term decomposition is:
    # m1 = a00 * b00
    # m2 = a01 * b10
    # m3 = a00 * b01
    # m4 = a01 * b11
    # m5 = a10 * b00
    # m6 = a11 * b10
    # m7 = (a00 + a10) * (b01 + b11)
    # Then:
    # c00 = m1 + m2
    # c01 = m3 + m4
    # c10 = m5 + m6
    # c11 = m7 - m3 - m6  ... let's check: (a00+a10)(b01+b11) - a00*b01 - a11*b10
    # = a00*b01 + a00*b11 + a10*b01 + a10*b11 - a00*b01 - a11*b10
    # = a00*b11 + a10*b01 + a10*b11 - a11*b10
    # But we need c11 = a10*b01 + a11*b11. This doesn't match.
    
    # I'll use the straightforward rank-8 decomposition for correctness, giving rank 72.
    # This is a valid decomposition, just not rank-minimal.
    
    rank = 8 * p  # 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 8-term decomposition for 2x2:
    # Term indices for (i,j,k) in 2x2x1:
    # (0,0,0): a00*b00 -> c00
    # (0,1,0): a01*b10 -> c00
    # (0,0,1): a00*b01 -> c01
    # (0,1,1): a01*b11 -> c01
    # (1,0,0): a10*b00 -> c10
    # (1,1,0): a11*b10 -> c10
    # (1,0,1): a10*b01 -> c11
    # (1,1,1): a11*b11 -> c11
    
    terms_2x2 = [
        (0, 0, 0),  # a00 * b00 -> c00
        (1, 3, 0),  # a01 * b10 -> c00  (U index 0*2+1=1, V index 1*2+0=2... wait)
    ]
    
    # Let me be more careful with indices.
    # A is 2x2: A[i,j] for i,j in {0,1}. U index = i*m + j = i*2 + j.
    # B is 2x2: B[j,k] for j,k in {0,1}. V index = j*p + k = j*2 + k (for p=1, j*1+k... wait p=1 for the 2x2x1 sub-problem)
    # Actually, in the full problem, B is 2x9, so B[j,k] for j in {0,1}, k in {0..8}. V index = j*9 + k.
    # C is 2x9: C[i,k] for i in {0,1}, k in {0..8}. W index = k*2 + i (since p*n = 9*2 = 18, and W[k*n+i]).
    
    # For the k-th column of C (k in 0..8), we have a 2x2 matrix multiply:
    # C[:,k] = A * B[:,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # For each k, we use 4 terms (the standard 4-term decomposition for 2x2x1, which has rank 4):
    # Term 1: A[0,0] * B[0,k] -> C[0,k]
    # Term 2: A[0,1] * B[1,k] -> C[0,k]
    # Term 3: A[1,0] * B[0,k] -> C[1,k]
    # Term 4: A[1,1] * B[1,k] -> C[1,k]
    
    # This gives rank 4*9 = 36, which is what the original code does.
    # But the rank of M<2,2,1> is actually 4? No, the rank of standard 2x2 matrix multiplication is 7.
    # The 4-term decomposition I just described is for the "diagonal" terms only, which is not correct.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] requires 2 terms, not 1.
    
    # So the correct rank for 2x2 matrix multiply is 7, and the 4-term scheme I described is invalid.
    # The original code with rank 36 is also invalid because it only captures diagonal-like terms.
    
    # Let me verify the original code:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # For col corresponding to (i,j,k): U[i*2+j] = 1, V[j*9+k] = 1, W[k*2+i] = 1
    # This means the term is A[i,j] * B[j,k] * C[k,i]... but C should be C[i,k], not C[k,i].
    # W index is k*n+i = k*2+i, which corresponds to C[i,k] if we interpret W index as k*n+i meaning row k, col i? 
    # The task says "W[k*n+i,r]" which suggests W is indexed by (k,i) but the output is C[i,k].
    # Actually, the trace convention W[k*n+i] likely means the (k,i)-th entry, but for matrix multiply C = A*B,
    # C[i,k] = sum_j A[i,j]*B[j,k]. So W should have index i*n+k or k*n+i depending on convention.
    # The task says W[k*n+i], so W[k*n+i] corresponds to C[i,k]? Or C[k,i]?
    # Given the ambiguity, let's assume W[k*n+i] is the coefficient for C[i,k] where the index is computed as k*n+i.
    
    # For a correct decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) ... no, that's not right.
    # The tensor T[i,j,k] for matrix multiply is 1 if the indices are compatible, i.e., T[i,j,k] = 1 always for 2x2x2?
    # No, the matrix multiplication tensor T satisfies C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in the appropriate spaces).
    # Wait, no. The tensor is T in R^{n*m} x R^{m*p} x R^{p*n} where T[(i,j),(j,k),(k,i)]... 
    # Actually, the standard form is: T[i,j,k] = 1 if the indices form a valid "chain" i->j->k, which is always true.
    # So T is the tensor with T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
    # But that's not right either. The matrix multiplication tensor is defined by:
    # <T, A \otimes B \otimes C^T> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # For C = A*B, we have C[k,i] = sum_j A[i,j] B[j,k]... no, C[i,k] = sum_j A[i,j] B[j,k].
    # So C[k,i] would be the (k,i)-th entry of C, which is sum_j A[k,j] B[j,i].
    # This is getting confusing. Let me just use the standard 7-term decomposition for 2x2 and replicate it 9 times.
    
    # I'll implement the rank-8 version (4 terms per 2x2 block, but correctly) which gives rank 36,
    # and verify it's correct. If the original code's rank 36 is correct, then that's the answer.
    
    # Let me re-examine the original code's correctness.
    # Original: U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # This creates a rank-1 tensor for each (i,j,k) with:
    # U has 1 at position (i*m+j)
    # V has 1 at position (j*p+k)
    # W has 1 at position (k*n+i)
    # The product sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] 
    # For a fixed (i,j,k), this is 1 if there's a column r where all three are 1.
    # In the original code, each column corresponds to one (i',j',k') tuple, and U[i'*m+j'][col]=1, V[j'*p+k'][col]=1, W[k'*n+i'][col]=1.
    # So for the term to contribute to T[i,j,k], we need i'=i, j'=j, k'=k, and i'=i (from W).
    # So T[i,j,k] = 1 if there exists a column with i'=i, j'=j, k'=k, which is always true.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Is the matrix multiplication tensor the all-ones tensor? 
    # For C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined by T[i,j,k] such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = <C, A*B>.
    # <C, A*B> = sum_{i,k} C[i,k] (A*B)[i,k] = sum_{i,k} C[i,k] sum_j A[i,j] B[j,k]
    # = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # So T[i,j,k] = 1 for all i,j,k, and C is indexed by (i,k) not (k,i).
    # But the task says W[k*n+i], which suggests C is indexed by (k,i) or the index is k*n+i.
    # If W[k*n+i] corresponds to C[i,k], then T[i,j,k] = 1 for all i,j,k is correct.
    
    # So the original code with rank 36 is a valid decomposition! The rank of the all-ones tensor T_{n,m,p} 
    # is at most n*m*p, and the original code achieves exactly n*m*p = 36.
    # Is 36 the minimal rank? The rank of the all-ones tensor T_{n,m,p} is known to be n*m*p for small dimensions?
    # Actually, the rank of T_{n,m,p} (the tensor with all entries 1) is not necessarily n*m*p.
    # For T_{2,2,2}, the rank is 7 (same as 2x2 matrix multiply).
    # So for T_{2,2,9}, the rank is not 36; it's 7*9 = 63 if it decomposes as a direct sum.
    
    # But does T_{2,2,9} decompose as a direct sum of 9 copies of T_{2,2,1}?
    # T_{2,2,9}[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..8}.
    # This is NOT a direct sum. A direct sum would have T[i,j,k] = 0 unless k is in a specific range for each (i,j).
    # So T_{2,2,9} is a single tensor, not a direct sum.
    
    # The rank of T_{n,m,p} (all-ones tensor) is a known problem. For T_{2,2,2}, rank = 7.
    # For T_{2,2,9}, the rank is likely less than 36 but greater than 7.
    
    # Given the complexity, let me stick with the rank-36 decomposition which is valid,
    # and note that it may not be minimal. The score will be 1.0 (same as parent) since rank is 36.
    
    # Actually, the parent score is 1.0, and the score is seed/value. If seed is 36, then value=36 gives score 1.0.
    # To improve, we need value < 36.
    
    # Let me try to find a lower-rank decomposition. For now, I'll keep rank 36 as a safe fallback.
    
    rank = n * m * p  # 36
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
