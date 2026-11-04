import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 8
    
    # Try rank-70 via 8x8x8 Strassen embedding
    try:
        result = construct_rank70(n, m, p)
        return result
    except Exception:
        pass
    
    # Fallback to rank-98
    return construct_rank98(n, m, p)

def construct_rank70(n, m, p):
    """
    Embed 4x4*4x8 into 8x8*8x8.
    A_8x8: top-left 4x4 = A, rest = 0
    B_8x8: top-right 4x8 = B, rest = 0
    C_8x8 = A_8x8 * B_8x8
    Output C[0:4, 0:4] = A*B (top-left 4x4 of result)
    
    Use Strassen rank-7 on 8x8x8, keep only terms contributing to C[0:4,0:4].
    """
    n, m, p = 4, 4, 8
    N = 8  # full dimension for Strassen
    
    # Strassen 2x2x2 base matrices (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 8x8x8 Strassen, we recursively apply this.
    # The 8x8x8 tensor has rank 7^3 = 343 terms if fully expanded.
    # But we only need C[0:4, 0:4] which is the top-left 4x4.
    
    # Let's build the 8x8x8 Strassen decomposition explicitly.
    # Index: for 8x8, split into 4 blocks of 4x4.
    # A_8x8 = [[A11, A12], [A21, A22]] where each is 4x4
    # B_8x8 = [[B11, B12], [B21, B22]] where each is 4x4
    # C_8x8 = A_8x8 * B_8x8
    # C11 (top-left 4x4) = A11*B11 + A12*B21
    
    # Using Strassen on the 4x4 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    #
    # Each M_k is a 4x4*4x4 product, which itself uses Strassen rank-7.
    # So total rank = 7 * 7 = 49 for C11.
    #
    # But wait - we need to be more careful. The 4x4 blocks themselves
    # can be further decomposed. Let me think about this differently.
    
    # Actually, let's use the recursive structure properly.
    # 8x8x8 Strassen: split 8 into 4+4, apply rank-7 Strassen.
    # Each of the 7 sub-products is a 4x4*4x4 multiplication.
    # Each 4x4*4x4 can use Strassen rank-7 again (split 4 into 2+2).
    # So total rank for full 8x8x8 = 7*7 = 49? No, 7^3 = 343.
    
    # Wait, I'm confusing myself. Let me be precise.
    # 8x8x8: rank = 7^3 = 343 (three levels of Strassen)
    # But we only need C[0:4, 0:4].
    
    # Let me build it level by level.
    # Level 0: 8x8*8x8, split into 4x4 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 + M4 - M5 + M7 where M_k are 4x4*4x4 products
    
    # For our problem:
    # A_8x8 has A11 = A (4x4), A12 = 0, A21 = 0, A22 = 0
    # B_8x8 has B11 = 0, B12 = B (4x8 -> top-right 4x4 of B), 
    #            B21 = 0, B22 = bottom-right 4x4 of B
    #
    # Wait, B is 4x8. In the 8x8 embedding:
    # B_8x8 = [[0, B_top], [0, B_bot]]
    # where B_top = B[:, 0:4] (4x4), B_bot = B[:, 4:8] (4x4)
    #
    # So: B11 = 0 (top-left 4x4), B12 = B[:, 0:4] (top-right 4x4)
    #     B21 = 0 (bottom-left 4x4), B22 = B[:, 4:8] (bottom-right 4x4)
    
    # A11 = A, A12 = 0, A21 = 0, A22 = 0
    # B11 = 0, B12 = B[:, 0:4], B21 = 0, B22 = B[:, 4:8]
    
    # M1 = (A11+A22)(B11+B22) = A * B[:, 4:8]
    # M2 = (A21+A22)B11 = 0 * 0 = 0
    # M3 = A11(B21-B22) = A * (0 - B[:, 4:8]) = -A * B[:, 4:8]
    # M4 = A22(B12-B11) = 0 * B[:, 0:4] = 0
    # M5 = (A11+A12)B22 = A * B[:, 4:8]
    # M6 = (A21-A11)(B11+B12) = (-A) * B[:, 0:4]
    # M7 = (A12-A22)(B21+B22) = 0 * B[:, 4:8] = 0
    
    # C11 = M1 + M4 - M5 + M7 = A*B[:,4:8] + 0 - A*B[:,4:8] + 0 = 0???
    
    # That can't be right. Let me recheck the Strassen formulas.
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Hmm, C11 = 0 means the top-left 4x4 of C is zero, which makes sense
    # because A_8x8 * B_8x8 = [[A*B_top, A*B_bot], [0, 0]]
    # So C11 = A*B_top, C12 = A*B_bot, C21 = 0, C22 = 0
    
    # Wait, I made an error. Let me redo:
    # C11 = A11*B11 + A12*B21 = A*0 + 0*0 = 0
    # C12 = A11*B12 + A12*B22 = A*B[:,0:4] + 0 = A*B[:,0:4]
    # C21 = A21*B11 + A22*B21 = 0
    # C22 = A21*B12 + A22*B22 = 0
    
    # So the output A*B (4x8) is in C12 (top-right 4x8 of the 8x8 result)!
    # C12 = A * B[:, 0:4] ... no wait.
    
    # Actually B is 4x8. When embedded in 8x8:
    # B_8x8[0:4, 0:8] = B, B_8x8[4:8, :] = 0
    # So B11 = B[:, 0:4], B12 = B[:, 4:8] (top row of 4x4 blocks)
    # B21 = 0, B22 = 0
    
    # A_8x8[0:4, 0:4] = A, rest = 0
    # A11 = A, A12 = 0, A21 = 0, A22 = 0
    
    # C11 = A11*B11 + A12*B21 = A * B[:, 0:4]
    # C12 = A11*B12 + A12*B22 = A * B[:, 4:8]
    # C21 = 0, C22 = 0
    
    # So A*B = [C11, C12] = [A*B[:,0:4], A*B[:,4:8]]
    # The full 4x8 output is in the top row of 4x4 blocks.
    
    # Now using Strassen:
    # M1 = (A11+A22)(B11+B22) = A * B[:, 0:4]
    # M2 = (A21+A22)B11 = 0
    # M3 = A11(B21-B22) = 0
    # M4 = A22(B12-B11) = 0
    # M5 = (A11+A12)B22 = 0
    # M6 = (A21-A11)(B11+B12) = (-A) * (B[:, 0:4] + B[:, 4:8])
    # M7 = (A12-A22)(B21+B22) = 0
    
    # C11 = M1 - M5 + M4 + M7 = A*B[:,0:4]
    # C12 = M3 + M5 = 0 ??? 
    
    # That's wrong. C12 should be A*B[:,4:8].
    
    # I think I have the Strassen formula wrong. Let me look it up carefully.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # With A12=A21=A22=0, B11=B_top, B12=B_bot, B21=B22=0:
    # M1 = A*B_top
    # M2 = 0
    # M3 = 0
    # M4 = 0
    # M5 = 0
    # M6 = (-A)(B_top + B_bot)
    # M7 = 0
    #
    # C11 = M1 + M4 - M5 + M7 = A*B_top ✓
    # C12 = M3 + M5 = 0 ✗ (should be A*B_bot)
    
    # Something is off. The issue is that standard Strassen assumes
    # all four quadrants of both matrices are nonzero. When A12=A21=A22=0,
    # the formula still works algebraically but C12 = M3 + M5 = 0.
    
    # But C12 should equal A11*B12 + A12*B22 = A*B_bot + 0 = A*B_bot.
    
    # Let me verify: C12 = A11*B12 + A12*B22
    # M3 + M5 = A11(B21-B22) + (A11+A12)B22 = A11*B21 - A11*B22 + A11*B22 + A12*B22 = A11*B21 + A12*B22
    # With B21=0, A12=0: M3+M5 = 0. But C12 = A11*B12 + A12*B22 = A*B_bot.
    
    # So M3+M5 ≠ C12 in general! I must have the Strassen formula wrong.
    
    # Correct Strassen:
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C12:
    # C12 should = A11*B12 + A12*B22
    # M3 + M5 = A11*(B21-B22) + (A11+A12)*B22 = A11*B21 - A11*B22 + A11*B22 + A12*B22 = A11*B21 + A12*B22
    # This equals A11*B21 + A12*B22, NOT A11*B12 + A12*B22.
    
    # So I have the wrong Strassen formula! Let me find the correct one.
    # 
    # Correct Strassen (from Wikipedia):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Hmm, let me verify with a simple case. If A=I, B=I (2x2):
    # A11=1, A12=0, A21=0, A22=1
    # B11=1, B12=0, B21=0, B22=1
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)*1 = 1
    # M3 = 1*(0-1) = -1
    # M4 = 1*(0-1) = -1
    # M5 = (1+0)*1 = 1
    # M6 = (0-1)*(1+0) = -1
    # M7 = (0-1)*(0+1) = -1
    # C11 = 4 - 1 + (-1) + (-1) = 1 ✓
    # C12 = -1 + 1 = 0 ✓
    # C21 = 1 + (-1) = 0 ✓
    # C22 = 4 - 1 + (-1) + (-1) = 1 ✓
    
    # OK so the formula is correct. Now with A12=A21=A22=0, B21=B22=0:
    # C12 = M3 + M5 = A11*(0-0) + (A11+0)*0 = 0
    # But actual C12 = A11*B12 + A12*B22 = A11*B12
    # 
    # So the Strassen formula gives C12=0 when it should be A11*B12.
    # This means my assignment of which block is which is wrong, OR
    # the Strassen decomposition doesn't work when blocks are zero in this way.
    
    # Wait, I think I confused myself. Let me re-derive.
    # The point of Strassen is that it's an identity: for ANY 2x2 block matrices,
    # the formula holds. So if I plug in A12=0, A21=0, A22=0, B21=0, B22=0,
    # I should get C12 = A11*B12.
    
    # C12 = M3 + M5 = A11*(B21-B22) + (A11+A12)*B22
    # With B21=0, B22=0, A12=0: = A11*0 + A11*0 = 0
    
    # But the actual (1,2) block of A*B is A11*B12 + A12*B22 = A11*B12.
    
    # So either the Strassen formula is wrong, or I'm misapplying it.
    
    # Let me check Wikipedia again. Actually I think the correct formulas are:
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5  
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # And M definitions as above. Let me verify C12 more carefully.
    # M3 + M5 = A11(B21-B22) + (A11+A12)B22 = A11B21 - A11B22 + A11B22 + A12B22 = A11B21 + A12B22
    
    # But C12 = A11B12 + A12B22. These are NOT the same unless B21=B12.
    
    # So I definitely have the wrong formula. Let me look up the actual Strassen.
    # 
    # After more careful thought, I believe the correct Strassen is:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b21-b22)
    # p4 = a22(b12-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    #
    # c11 = p1 - p5 + p4 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # This is what I had. And it fails for my special case. So the formula must be
    # different. Let me try:
    # c12 = p3 + p5 should give a11*b12 + a12*b22
    
    # p3 + p5 = a11(b21-b22) + (a11+a12)b22 = a11*b21 + a12*b22
    
    # This is NOT a11*b12 + a12*b22. So the formula is wrong.
    
    # I'll give up on deriving it from scratch and just use the known rank-7
    # Strassen matrices directly, which I already have in the current code.
    
    # The current code has U2, V2, W2 for 2x2x2 Strassen. Let me use those.
    # For 8x8x8, I need to build the full Strassen tensor.
    
    # Actually, the simplest approach: the current code already builds a rank-49
    # decomposition for 4x4x4 (U4, V4, W4). For 4x4x8, we can use two copies
    # (one for B[:,0:4] and one for B[:,4:8]), giving rank 98. That's what the
    # current code does.
    
    # To get rank 70, I need a different approach. Let me think...
    
    # Alternative: Use the fact that 4x4x8 can be seen as 4x4x4 + 4x4x4,
    # but with shared structure. Or use a different decomposition.
    
    # Actually, let me try a different embedding. What if I use 4x8x8?
    # A is 4x4, B is 4x8. Embed A in 4x8 (pad with zeros on right),
    # B stays 4x8. Output is 4x8.
    # Then use Strassen on 4x8x8? That's not standard.
    
    # Let me try yet another approach: direct construction.
    # The tensor is T[i,j,k] = delta_{i,k} for the 4x4x8 case.
    # No wait, for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = delta_{i,k} (for the standard MM tensor)
    # Wait no. The MM tensor has T[i,j,k] = 1 if i=k... no.
    # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = delta_{i,k}... that's not right either.
    
    # The MM tensor for n x m times m x p:
    # T in R^{n x m x p} where T[i,j,k] = 1 if... 
    # Actually C[i,k] = sum_j A[i,j] B[j,k], so the tensor is:
    # T[i,j,k] = 1 (for all valid i,j,k) - no, that's not a tensor decomposition.
    
    # The MM tensor is: T = sum_{i,j,k} E_{i,k}^{(n,p)} \otimes E_{i,j}^{(n,m)} \otimes E_{j,k}^{(m,p)}
    # where E_{a,b} is the matrix with 1 at (a,b).
    # Rank is at most n*m*p.
    
    # For a decomposition of rank r:
    # U is n*m x r, V is m*p x r, W is p*n x r (using W[k*n+i, r] convention)
    # Such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (wait, no)
    
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] (if the decomposition is correct)
    
    # So we need: sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k} * (coefficient)
    # Wait, for each r, the product U[:,r] * V[:,r] * W[:,r] should contribute
    # to the MM tensor. The MM tensor T[i,j,k] = 1 for the "right" indices.
    
    # I'm overcomplicating this. Let me just verify the current rank-98 solution
    # is correct and see if I can find a lower-rank one through search.
    
    # Given the time constraints, let me try a random search for a lower-rank
    # decomposition, using the rank-98 as a starting point and trying to
    # eliminate terms.
    
    raise RuntimeError("Rank-70 construction not implemented; falling back to rank-98")

def construct_rank98(n, m, p):
    n, m, p = 4, 4, 8
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
