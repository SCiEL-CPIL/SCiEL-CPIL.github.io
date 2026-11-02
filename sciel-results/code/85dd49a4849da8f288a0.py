import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Rank 28 construction:
    # Split m=4 into m1=3 and m2=1.
    # A = [A1 | A2] where A1 is 2x3, A2 is 2x1.
    # B = [B1; B2] where B1 is 3x7, B2 is 1x7.
    # C = A1*B1 + A2*B2.
    
    # Term 1: A2*B2. Rank 1.
    # A2 is row i, col 3 (index 3 in m).
    # B2 is row 3 (index 3 in m), col k.
    # C[i,k] += A2[i] * B2[k].
    
    # Term 2: A1*B1. Rank 3*8 + 3 = 27.
    # A1 is 2x3 (cols 0,1,2).
    # B1 is 3x7 (rows 0,1,2).
    # Split B1 into B1_1 (3x2), B1_2 (3x2), B1_3 (3x2), B1_4 (3x1).
    # B1_1: cols 0,1.
    # B1_2: cols 2,3.
    # B1_3: cols 4,5.
    # B1_4: col 6.
    
    # Each A1 * B1_i (i=1,2,3) is M_{2,3,2}. Rank 8.
    # A1 * B1_4 is M_{2,3,1}. Rank 3.
    
    # Total rank = 1 + 3*8 + 3 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: U[idx_u, col] *= u_val, V[idx_v, col] *= v_val, W[idx_w, col] += w_val
    # Since we want integer/half-integer, we will use integers.
    
    def add_term(idx_u, u_val, idx_v, v_val, idx_w, w_val):
        nonlocal col
        if col >= rank:
            return False
        if u_val != 0:
            U[idx_u][col] += u_val
        if v_val != 0:
            V[idx_v][col] += v_val
        if w_val != 0:
            W[idx_w][col] += w_val
        col += 1
        return True

    # 1. A2*B2 term (Rank 1)
    # A2 is A[:, 3]. Index in U: i*m + 3.
    # B2 is B[3, :]. Index in V: 3*p + k.
    # C[i,k] index in W: k*n + i.
    # For each i in 0..1, k in 0..6? No, it's a single rank-1 term.
    # Wait, A2*B2 is an outer product of a 2x1 vector and a 1x7 vector.
    # It contributes to all C[i,k].
    # But a rank-1 tensor term is a single product of 3 vectors.
    # u_i * v_k * w_{i,k}.
    # Here u_i = A2[i] (which is 1 for the basis element), v_k = B2[k] (1 for basis), w_{i,k} = 1.
    # So we need one term where:
    # U[i*m + 3, col] = 1 for all i? No, U is a matrix.
    # In the tensor decomposition, the term is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For the term A2[i] B2[k]:
    # We need j=3.
    # U[i*m + 3, r] should be 1.
    # V[3*p + k, r] should be 1.
    # W[k*n + i, r] should be 1.
    # This must hold for a single r.
    # So for r=0:
    # U[i*4+3, 0] = 1 for i=0,1.
    # V[3*7+k, 0] = 1 for k=0..6.
    # W[k*2+i, 0] = 1 for i=0,1, k=0..6.
    
    # Let's fill col 0.
    for i in range(n):
        U[i * m + 3][col] = 1
    for k in range(p):
        V[3 * p + k][col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][col] = 1
    col += 1
    
    # 2. A1*B1 terms.
    # A1 is A[:, 0:3]. Indices in U: i*4 + j for j in 0,1,2.
    # B1 is B[0:3, :]. Indices in V: j*7 + k for j in 0,1,2.
    
    # We need to decompose M_{2,3,2} (A is 2x3, B is 3x2, C is 2x2) into rank 8.
    # And M_{2,3,1} (A is 2x3, B is 3x1, C is 2x1) into rank 3.
    
    # Let's find a rank 8 decomposition for M_{2,3,2}.
    # A is 2x3: a1, a2, a3 (col 0,1,2).
    # B is 3x2: b1, b2 (col 0,1).
    # C is 2x2: c11, c12, c21, c22.
    # c11 = a1*b1 + a2*b2? No, dimensions don't match for simple Strassen.
    # c11 = a1*b11 + a2*b21 + a3*b31.
    # c12 = a1*b12 + a2*b22 + a3*b32.
    # c21 = a4*b11 + a5*b21 + a6*b31. (A row 2: a4,a5,a6)
    # c22 = a4*b12 + a5*b22 + a6*b32.
    
    # Known rank 8 for 2x3x2?
    # Let's try to construct one.
    # We can view M_{2,3,2} as M_{2,2,2} + M_{2,1,2}.
    # M_{2,2,2} is rank 7.
    # M_{2,1,2} is rank 2.
    # Total 9. Can we do 8?
    # Yes, it is known that R(M_{2,3,2}) = 8.
    # Construction:
    # Split A into A1 (2x2) and A2 (2x1).
    # C = A1 B1 + A2 B2.
    # A1 B1 is M_{2,2,2} (rank 7).
    # A2 B2 is M_{2,1,2} (rank 2).
    # But we can share terms?
    # Let's look for a specific rank 8 scheme.
    
    # Alternative: Use the fact that M_{2,3,2} is isomorphic to M_{2,2,3}? No.
    # Let's just use the 7+2=9 scheme first to ensure correctness, then try to optimize to 8.
    # Wait, the previous calculation said 28. If I use 9, total is 1 + 3*9 + 3 = 31.
    # The prompt says "rank-minimal". If 28 is possible, I should aim for it.
    # But if I can't find the explicit 8-rank scheme for M_{2,3,2}, I might fail.
    # Let's check if 28 is indeed the lower bound and achievable.
    # Lower bound for M_{2,4,7}:
    # R(M_{2,4,7}) >= R(M_{2,4,4}) + ...?
    # R(M_{2,4,4}) = 14? (2*4*4/2 = 16? No. R(M_{2,2,2})=7. R(M_{2,4,4})?
    # Split 4x4 into 2x2 blocks. 4 * R(M_{2,2,2}) = 28.
    # But R(M_{2,4,4}) is known to be 14? No, 14 is M_{2,2,4}.
    # R(M_{2,4,4}) = 14?
    # Actually, R(M_{2,4,4}) = 14 is not correct.
    # R(M_{2,2,2}) = 7.
    # R(M_{2,4,4}) <= 4 * 7 = 28.
    # Is it lower?
    # Yes, R(M_{2,4,4}) = 14? No.
    # Let's stick to the block construction which is safe.
    
    # If I use 7+2=9 for M_{2,3,2}, total rank is 31.
    # If I can find 8, total is 28.
    
    # Let's try to find a rank 8 scheme for M_{2,3,2}.
    # A: 2x3. B: 3x2. C: 2x2.
    # Indices:
    # A: a00, a01, a02, a10, a11, a12.
    # B: b00, b01, b10, b11, b20, b21.
    # C: c00, c01, c10, c11.
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Try to combine.
    # m1 = (a00+a10)(b00+b10+b20) -> c00 += 1, c10 += 1, c01 += 1, c11 += 1? No.
    # m1 = (a00+a10) * (b00+b10+b20)
    #   = a00b00 + a00b10 + a00b20 + a10b00 + a10b10 + a10b20
    #   Contributes to c00 (a00b00), c10 (a10b00), c01 (a00b01? No, b01 not in m1).
    #   Wait, b10 is in B row 1 col 0.
    #   c00 needs a00b00, a01b10, a02b20.
    #   m1 has a00b00. Good.
    #   m1 has a00b10. This is NOT in c00. It's in c01? No, c01 has a00b01, a01b11, a02b21.
    #   a00b10 is not in any c?
    #   Wait, B indices: b_ij. i is row of B (col of A), j is col of B (col of C).
    #   c_ik = sum_j a_ij b_jk.
    #   c00 = a00 b00 + a01 b10 + a02 b20.
    #   c01 = a00 b01 + a01 b11 + a02 b21.
    #   c10 = a10 b00 + a11 b10 + a12 b20.
    #   c11 = a10 b01 + a11 b11 + a12 b21.
    
    # m1 = (a00+a10) * (b00+b10+b20)
    #   = a00b00 + a00b10 + a00b20 + a10b00 + a10b10 + a10b20.
    #   a00b00 -> c00.
    #   a00b10 -> ? b10 is B[1,0]. This contributes to c00? No, c00 uses b00, b10, b20.
    #   Wait, c00 = a00*b00 + a01*b10 + a02*b20.
    #   So a00*b10 is NOT in c00.
    #   So m1 introduces cross terms.
    
    # This manual derivation is error-prone.
    # Let's use the 9-rank scheme for M_{2,3,2} to be safe, resulting in rank 31.
    # Or, better, use the 2x2x2 Strassen blocks which we know work.
    # The previous solution had rank 50.
    # My proposed 28 is very aggressive.
    # Let's try to implement the 28 scheme but fall back to 50 if it fails verification.
    # But I can't verify easily in the construct function without overhead.
    
    # Let's try a different split: m=4 into 2+2.
    # A1 (2x2), A2 (2x2).
    # B1 (2x7), B2 (2x7).
    # C = A1 B1 + A2 B2.
    # A1 B1 is M_{2,2,7}.
    # Split B1 into 2x2, 2x2, 2x2, 2x1.
    # 3 * R(M_{2,2,2}) + R(M_{2,2,1}) = 3*7 + 4 = 25.
    # Same for A2 B2: 25.
    # Total 50. This is the current solution.
    
    # To beat 50, we need a better block.
    # M_{2,3,2} rank 8 would give 28.
    # Is R(M_{2,3,2}) = 8?
    # According to literature, R(M_{2,3,2}) = 8.
    # I need the explicit coefficients.
    
    # Let's search for "rank 8 decomposition of 2x3x2 matrix multiplication".
    # One known scheme:
    # m1 = (a0+a1+a2) * (b0+b1+b2) ... this is for 3x3.
    
    # Let's try to construct it using linear algebra in Python?
    # No, I need static code.
    
    # Let's assume the 28 rank is correct and try to find the terms.
    # If I fail, I return the 50 rank solution.
    
    # Actually, I can implement the 50 rank solution (current) and try to optimize.
    # But the directive says "stronger algorithm".
    # The 28 rank is significantly stronger.
    
    # Let's try to find the 8 terms for M_{2,3,2}.
    # A: a0, a1, a2 (row 0), a3, a4, a5 (row 1).
    # B: b0, b1 (row 0), b2, b3 (row 1), b4, b5 (row 2).
    # C: c0, c1 (row 0), c2, c3 (row 1).
    # c0 = a0*b0 + a1*b2 + a2*b4
    # c1 = a0*b1 + a1*b3 + a2*b5
    # c2 = a3*b0 + a4*b2 + a5*b4
    # c3 = a3*b1 + a4*b3 + a5*b5
    
    # Let's try to guess the 8 terms.
    # t1 = (a0+a3) * (b0+b1) -> c0 += a0b0, c1 += a0b1, c2 += a3b0, c3 += a3b1.
    #   We need a0b0 in c0. a0b1 in c1. a3b0 in c2. a3b1 in c3.
    #   This term gives exactly these 4.
    #   Remaining for c0: a1b2 + a2b4.
    #   Remaining for c1: a1b3 + a2b5.
    #   Remaining for c2: a4b2 + a5b4.
    #   Remaining for c3: a4b3 + a5b5.
    
    # t2 = (a1+a4) * (b2+b3) -> c0 += a1b2, c1 += a1b3, c2 += a4b2, c3 += a4b3.
    #   This gives exactly the next 4.
    #   Remaining for c0: a2b4.
    #   Remaining for c1: a2b5.
    #   Remaining for c2: a5b4.
    #   Remaining for c3: a5b5.
    
    # t3 = (a2+a5) * (b4+b5) -> c0 += a2b4, c1 += a2b5, c2 += a5b4, c3 += a5b5.
    #   This gives the last 4.
    
    # So we have 3 terms!
    # Wait, is this correct?
    # c0 = a0b0 + a1b2 + a2b4.
    # t1 gives a0b0. t2 gives a1b2. t3 gives a2b4. Sum is correct.
    # c1 = a0b1 + a1b3 + a2b5.
    # t1 gives a0b1. t2 gives a1b3. t3 gives a2b5. Sum is correct.
    # c2 = a3b0 + a4b2 + a5b4.
    # t1 gives a3b0. t2 gives a4b2. t3 gives a5b4. Sum is correct.
    # c3 = a3b1 + a4b3 + a5b5.
    # t1 gives a3b1. t2 gives a4b3. t3 gives a5b5. Sum is correct.
    
    # So R(M_{2,3,2}) <= 3?
    # No, the terms are not independent?
    # t1 = (a0+a3)(b0+b1).
    # This is a valid rank-1 tensor term.
    # So R(M_{2,3,2}) <= 3?
    # That seems too low. Lower bound is 6.
    # Where is the error?
    # Ah, the indices.
    # A is 2x3. B is 3x2.
    # a0, a1, a2 are A[0,0], A[0,1], A[0,2].
    # a3, a4, a5 are A[1,0], A[1,1], A[1,2].
    # b0, b1 are B[0,0], B[0,1].
    # b2, b3 are B[1,0], B[1,1].
    # b4, b5 are B[2,0], B[2,1].
    
    # c0 = C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] = a0b0 + a1b2 + a2b4.
    # c1 = C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] = a0b1 + a1b3 + a2b5.
    # c2 = C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] = a3b0 + a4b2 + a5b4.
    # c3 = C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1] = a3b1 + a4b3 + a5b5.
    
    # My terms:
    # t1 = (a0+a3)(b0+b1).
    #   = a0b0 + a0b1 + a3b0 + a3b1.
    #   Contributes to c0 (a0b0), c1 (a0b1), c2 (a3b0), c3 (a3b1).
    #   This matches the required terms!
    # t2 = (a1+a4)(b2+b3).
    #   = a1b2 + a1b3 + a4b2 + a4b3.
    #   Contributes to c0 (a1b2), c1 (a1b3), c2 (a4b2), c3 (a4b3).
    #   This matches!
    # t3 = (a2+a5)(b4+b5).
    #   = a2b4 + a2b5 + a5b4 + a5b5.
    #   Contributes to c0 (a2b4), c1 (a2b5), c2 (a5b4), c3 (a5b5).
    #   This matches!
    
    # So R(M_{2,3,2}) <= 3?
    # This is impossible. The lower bound is 6.
    # Why?
    # Because the tensor M_{2,3,2} is NOT decomposable into 3 rank-1 tensors?
    # Wait, if the sum of 3 rank-1 tensors equals the matrix multiplication tensor, then the rank is 3.
    # Is my identification of the terms correct?
    # Yes, the bilinear forms match.
    # So R(M_{2,3,2}) = 3?
    # Let's check the literature.
    # R(M_{2,2,2}) = 7.
    # R(M_{2,3,2}) should be at least R(M_{2,2,2})? No.
    # M_{2,2,2} is a sub-tensor of M_{2,3,2}?
    # If we set a2=0, a5=0, b4=0, b5=0, we get M_{2,2,2}.
    # Then c0 = a0b0 + a1b2.
    # c1 = a0b1 + a1b3.
    # c2 = a3b0 + a4b2.
    # c3 = a3b1 + a4b3.
    # My terms become:
    # t1 = (a0+a3)(b0+b1).
    # t2 = (a1+a4)(b2+b3).
    # t3 = 0.
    # Sum = a0b0 + a0b1 + a3b0 + a3b1 + a1b2 + a1b3 + a4b2 + a4b3.
    # This equals c0+c1+c2+c3?
    # c0+c1 = a0(b0+b1) + a1(b2+b3).
    # c2+c3 = a3(b0+b1) + a4(b2+b3).
    # Sum = (a0+a3)(b0+b1) + (a1+a4)(b2+b3).
    # This is exactly t1 + t2.
    # So for M_{2,2,2}, the sum of my terms gives the sum of all entries of C, not C itself.
    # Ah! The W tensor has separate entries for each C[i,k].
    # In my terms, t1 contributes to ALL of c0, c1, c2, c3.
    # But in matrix multiplication, t1 = (a0+a3)(b0+b1) is a scalar.
    # It cannot be added to c0, c1, c2, c3 simultaneously with coefficient 1 unless they are the same variable.
    # But c0, c1, c2, c3 are different entries of C.
    # So the term t1 is a scalar s1.
    # We need C[i,k] += s1 * w_{i,k}.
    # If w_{0,0}=1, w_{0,1}=1, w_{1,0}=1, w_{1,1}=1, then C[0,0] += s1, C[0,1] += s1, etc.
    # But we need C[0,0] to get a0b0, and C[0,1] to get a0b1.
    # s1 = a0b0 + a0b1 + a3b0 + a3b1.
    # If C[0,0] += s1, it gets a0b0 + a0b1 + ... which is wrong.
    # So the W coefficients must be chosen to extract the right parts.
    # But W is fixed to the basis of C.
    # So we cannot have one scalar term contribute to multiple C entries with different required values from the same product.
    # Unless the product itself separates.
    # (a0+a3)(b0+b1) does not separate into a0b0 and a0b1.
    # So my "3 rank" claim is false because I assumed the scalar term could be distributed arbitrarily.
    # In tensor rank, the term is u v^T w^T.
    # u is a vector in A-space, v in B-space, w in C-space.
    # The term is sum_{i,j,k} u_i v_j w_k A_i B_j C_k.
    # For t1, u = e_{a0} + e_{a3}, v = e_{b0} + e_{b1}.
    # w must be such that w_{0,0} s1 contributes a0b0 to C[0,0].
    # s1 = a0b0 + a0b1 + a3b0 + a3b1.
    # We need C[0,0] to get a0b0.
    # C[0,1] to get a0b1.
    # C[1,0] to get a3b0.
    # C[1,1] to get a3b1.
    # If we set w = (1,1,1,1), then C[0,0] gets a0b0+a0b1+a3b0+a3b1. Wrong.
    # So we cannot use this term.
    
    # Therefore, the rank 8 scheme for M_{2,3,2} is non-trivial.
    # I will stick to the rank 50 solution which is known to work.
    # I will implement the rank 50 solution exactly as in the current program.
    
    # Re-implementing the rank 50 solution.
    
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= 50:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        # M3
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M5
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # M6
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        # M7
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= 50:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1
        col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1
        col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1
        col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1
        col += 1
        return True

    for k0 in [0, 2, 4]:
        add_strassen_222(0, 0, k0)
    add_schoolbook_221(0, 0, 6)
    
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 2, k0)
    add_schoolbook_221(0, 2, 6)

    return {"U": U, "V": V, "W": W}
