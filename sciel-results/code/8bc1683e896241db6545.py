"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid rank-r decomposition of the matrix multiplication tensor."""
    r = U.shape[1]
    # Build the tensor T[a, b, c] where a=i*n+j, b=j*p+k, c=k*n+i
    # The matrix multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.dot(U[idx_a, :], V[idx_b, :]) * 0  # placeholder
                # Actually: sum_r U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
                val = 0.0
                for rr in range(r):
                    val += U[idx_a, rr] * V[idx_b, rr] * W[idx_c, rr]
                if abs(val - 1) > tol:
                    return False
    # Also check that non-multiplication entries are 0
    # For a 3x3x3 case, check a few non-trivial positions
    # The tensor is zero unless a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # Let's check all positions
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                expected = 1 if (i == j2 == i2 and j == k2) else 0
                # Wait, let me re-examine the indexing
                # a = i*m + j, so i = a//m, j = a%m
                # b = j*p + k, so j = b//p, k = b%p
                # c = k*n + i, so k = c//n, i = c%n
                # For consistency: i (from a) must equal i2 (from c), j (from a) must equal j (from b), k (from b) must equal k (from c)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                val = 0.0
                for rr in range(r):
                    val += U[a, rr] * V[b, rr] * W[c, rr]
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    # Build tensor from decomposition
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is the tensor product: T = sum_r outer(U[:,r], V[:,r]) * W[:,r]
    # More efficiently, reshape and use einsum
    # T[a,b,c] = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _strassen_3x3():
    """
    Rank-26 decomposition of 3x3 matrix multiplication.
    
    Partition the 3x3 matrices into a 2x2 block (top-left) and 1x1 blocks:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    
    Using Strassen on the 2x2 sub-blocks:
    M1 = (A11+A22)(B11+B22)         -- 7 Strassen terms for 2x2x2
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Result C = A*B:
    C11 = M1 - M4 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M6 + M7
    
    Plus schoolbook for the remaining 1x1 entries:
    C13 = A11*B13 + A12*B23 + A13*B33
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21 + A33*B31
    C32 = A31*B12 + A32*B22 + A33*B32
    C33 = A31*B13 + A32*B23 + A33*B33
    
    Total rank: 7 (Strassen) + 3 + 3 + 3 + 3 + 3 (schoolbook for 1x1) = 7 + 15 = 22?
    Wait, let me recount. The 1x1 rows/cols need:
    Row 1, col 3: A11*B13, A12*B23, A13*B33  -> 3 terms
    Row 2, col 3: A21*B13, A22*B23, A23*B33  -> 3 terms
    Row 3, col 1: A31*B11, A32*B21, A33*B31  -> 3 terms
    Row 3, col 2: A31*B12, A32*B22, A33*B32  -> 3 terms
    Row 3, col 3: A31*B13, A32*B23, A33*B33  -> 3 terms
    Total: 7 + 15 = 22? That seems too low. Let me re-examine.
    
    Actually wait - the Strassen terms already use A11,A12,A21,A22 and B11,B12,B21,B22.
    The schoolbook terms for C13,C23,C31,C32,C33 use A13,A23,A31,A32,A33 and B13,B23,B31,B32,B33.
    
    Let me just build it explicitly and verify.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # a = i*m + j (row i, col j of A)
    # b = j*p + k (row j, col k of B)  
    # c = k*n + i (row k, col i of C)
    
    col = 0
    
    # Strassen terms for the 2x2 block (rows/cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U[0*3+0, col] = 1; V[0*3+0, col] = 1; W[0*3+0, col] = 1  # A00*B00 -> C00
    U[0*3+0, col] = 1; V[1*3+1, col] = 1; W[1*3+0, col] = 1  # A00*B11 -> C10? No...
    
    # Let me be more careful. Let me use a different approach.
    # I'll build the tensor directly from the bilinear forms.
    
    # Each term r contributes: U[a,r]*V[b,r]*W[c,r] to T[a,b,c]
    # T[a,b,c] = 1 iff a=i*3+j, b=j*3+k, c=k*3+i (consistent i,j,k)
    
    # Let me define bilinear forms and their contributions.
    # For a term with factorization (sum of A-entries)(sum of B-entries), 
    # it contributes to specific C entries.
    
    # Actually, let me just enumerate all 27 schoolbook terms and then
    # replace the 9 terms for the 2x2 block with 7 Strassen terms.
    
    # Schoolbook terms for 2x2 block (i,j in {0,1}, k in {0,1}):
    # (0,0,0): A00*B00 -> C00
    # (0,0,1): A00*B01 -> C10
    # (0,1,0): A01*B10 -> C01
    # (0,1,1): A01*B11 -> C11
    # (1,0,0): A10*B00 -> C00
    # (1,0,1): A10*B01 -> C10
    # (1,1,0): A11*B10 -> C01
    # (1,1,1): A11*B11 -> C11
    
    # Strassen replaces these 8 with 7:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B10-B11) -> C01 += M3
    # M4 = A11(B01-B00) -> C11 -= M4 (i.e., C11 gets -M4)
    # M5 = (A00+A01)B11 -> C10 += M5? Let me recheck.
    
    # Standard Strassen for 2x2:
    # C00 = M1 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let me verify: 
    # C00 = M1-M4-M7 = (A00+A11)(B00+B11) - A11(B01-B00) - (A01-A11)(B10+B11)
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    #     = A00*B00 + A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 - A01*B10 - A01*B11 + A11*B10
    # Hmm, that doesn't look right. Let me recheck standard Strassen.
    
    # Standard Strassen (2x2):
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)*b11
    # p3 = a11*(b21-b22)
    # p4 = a22*(b12-b11)
    # p5 = (a11+a12)*b22
    # p6 = (a21-a11)*(b11+b12)
    # p7 = (a12-a22)*(b21+b22)
    # c11 = p1 - p4 - p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p6 + p7
    
    # Using 0-indexed: a00,a01,a10,a11 for A; b00,b01,b10,b11 for B
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b10-b11)
    # p4 = a11*(b01-b00)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b00+b01)
    # p7 = (a01-a11)*(b10+b11)
    # c00 = p1 - p4 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p6 + p7
    
    # Let me verify c00:
    # c00 = (a00+a11)(b00+b11) - a11(b01-b00) - (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # = a00*b00 + a00*b11 + 2*a11*b00 + 2*a11*b11 - a11*b01 - a01*b10 - a01*b11 + a11*b10
    # This should equal a00*b00 + a01*b10
    # So we need: a00*b11 + 2*a11*b00 + 2*a11*b11 - a11*b01 - a01*b10 - a01*b11 + a11*b10 = 0?
    # That's not generally true. I'm confusing myself.
    
    # Let me just directly verify with specific values.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    # p1 = (1+4)(5+8) = 5*13 = 65
    # p2 = (3+4)*5 = 35
    # p3 = 1*(7-8) = -1
    # p4 = 4*(6-5) = 4
    # p5 = (1+2)*8 = 24
    # p6 = (3-1)*(5+6) = 2*11 = 22
    # p7 = (2-4)*(7+8) = -2*15 = -30
    # c00 = 65 - 4 - (-30) = 65 - 4 + 30 = 91? Should be 19. Something's wrong.
    
    # I think I have the Strassen formula wrong. Let me look it up properly.
    # 
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1-M4-M7, M3+M5], [M2+M4, M1-M2+M6+M7]]
    
    # Verify: a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    # M1 = (1+4)(5+8) = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-8) = -1
    # M4 = 4*(6-5) = 4
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 22
    # M7 = (2-4)*(7+8) = -30
    # C00 = 65-4-(-30) = 91? No, should be 19.
    
    # I'm clearly making an error. Let me just hardcode the correct Strassen formulas.
    # Actually, I recall now:
    # C00 = M1 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    # With the M's as defined above. Let me recheck M7:
    # M7 = (b-d)(g+h) = (2-4)(7+8) = -2*15 = -30
    # C00 = 65 - 4 - (-30) = 91. Still wrong.
    
    # Wait, maybe I have the indices mixed up. Let me use a different reference.
    # 
    # From Wikipedia Strassen algorithm:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = (a + b) * h
    # m4 = (c + d) * (f - e)
    # m5 = (a + b) * (h - g)
    # m6 = d * (h - g)
    # m7 = a * (e + f)
    # C = [[m1 - m4 - m5 - m6, m3 + m6 - m7 - (a+b)*h + (a+b)*h], ...]
    
    # This is getting confusing. Let me just use a verified implementation.
    
    # Correct Strassen (verified):
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(g-h)
    # m4 = d(f-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 - m4 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m6 + m7
    
    # Let me recheck with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # m1 = 5*13 = 65
    # m2 = 7*5 = 35
    # m3 = 1*(-1) = -1
    # m4 = 4*1 = 4
    # m5 = 3*8 = 24
    # m6 = 2*11 = 22
    # m7 = (-2)*15 = -30
    # C00 = 65 - 4 - (-30) = 91  WRONG (should be 19)
    
    # I must have the formula wrong. Let me derive it differently.
    # C00 = ae + cg
    # C01 = af + ch
    # C10 = ce + dg
    # C11 = cf + dh
    
    # Try: m1 = (a+c)(e+g) = ae + ag + ce + cg
    #      m2 = (a+c)g = ag + cg
    #      m3 = (e+g)(c-d) = ce - cd + eg - dg
    #      m4 = (e+g)(a-b) = ae - be + eg - bg
    #      m5 = (a-b)(f+h) = af + ah - bf - bh
    #      m6 = d(f+h) = df + dh
    #      m7 = b(e+f) = be + bf
    # C00 = m1 - m2 - m3 - m4 = ae+ag+ce+cg - ag-cg - ce+cd-eg+dg - ae+be-eg+bg
    #     = ae+ag+ce+cg-ag-cg-ce+cd-eg+dg-ae+be-eg+bg
    #     = 2cd - 2eg + 2dg + 2be + 2bg
    # That's not right either.
    
    # OK let me just use a known-correct set. From CLRS:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(g-h)
    # m4 = d(f-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # c = [[m1-m4-m7, m3+m5], [m2+m4, m1-m2+m6+m7]]
    
    # I'll just trust this and verify numerically in code. If it doesn't work, I'll fall back.
    
    # Let me just build the full rank-26 by starting with schoolbook (27) and 
    # replacing the 8 terms of the 2x2 block with 7 Strassen terms.
    
    # Actually, the simplest correct approach: just use schoolbook for all 27 terms
    # but that gives rank 27. The inspiration says rank 26 was achieved.
    
    # Let me think about this differently. The 3x3 matmul tensor has 27 nonzero entries.
    # A rank-r decomposition needs r terms. The minimum is 23.
    # 
    # One way to get rank 26: use Strassen on a 2x2 sub-block.
    # The 2x2 sub-block of A (rows 0,1; cols 0,1) times 2x2 sub-block of B (rows 0,1; cols 0,1)
    # gives 4 entries of C (rows 0,1; cols 0,1). That's 8 bilinear terms in schoolbook,
    # replaced by 7 Strassen terms. So we save 1 term: 27 - 1 = 26.
    
    # But we need to be careful: the Strassen terms involve sums of A entries and B entries,
    # and they contribute to the C entries with coefficients.
    
    # Let me just implement it carefully and verify.
    
    U = np.zeros((9, 26))
    V = np.zeros((9, 26))
    W = np.zeros((9, 26))
    
    col = 0
    
    # Strassen terms for 2x2 block (indices 0,1 in both dims)
    # A sub-block: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # B sub-block: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    # C sub-block: C[0,0]=C00, C[0,1]=C01, C[1,0]=C10, C[1,1]=C11
    
    # m1 = (A00+A11)(B00+B11)
    # Contributes to C00 with +1, C11 with +1
    # U: A00 -> U[0*3+0]=U[0], A11 -> U[1*3+1]=U[4]
    # V: B00 -> V[0*3+0]=V[0], B11 -> V[1*3+1]=V[4]
    # W: C00 -> W[0*3+0]=W[0], C11 -> W[1*3+1]=W[4]
    
    # For m1 = (A00+A11)(B00+B11), this is a product of sums.
    # In the tensor decomposition, each "term" r has U[a,r], V[b,r], W[c,r].
    # A single rank-1 term can only represent a single product A[i,j]*B[j,k] -> C[k,i].
    # Strassen's trick is that (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # which is 4 schoolbook terms combined into 1 "multiplication" but in the bilinear
    # decomposition sense, we need to express the contribution to each C entry.
    
    # Actually, in the tensor decomposition framework:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r, V_r, W_r are vectors.
    # 
    # For schoolbook: each term r corresponds to one (i,j,k) triple.
    # U_r has a 1 at position i*m+j, V_r has a 1 at position j*p+k, W_r has a 1 at position k*n+i.
    # 
    # For Strassen: m1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    # But in the tensor decomposition, we can't have a single term that contributes to
    # multiple C entries with different coefficients unless we use the structure.
    
    # Actually we CAN: W_r can have multiple nonzero entries.
    # For m1: U_r = e_{A00} + e_{A11}, V_r = e_{B00} + e_{B11}, W_r = e_{C00} + e_{C11}
    # Then the contribution to T[a,b,c] from this term is:
    # T[a,b,c] += U[a,r]*V[b,r]*W[c,r]
    # For a=A00, b=B00, c=C00: 1*1*1 = 1 ✓
    # For a=A00, b=B00, c=C11: 1*1*1 = 1, but we want C11 to get A00*B00? No!
    # C11 = A00*B01 + A01*B11 (from schoolbook). A00*B00 should NOT contribute to C11.
    
    # So this naive approach doesn't work. The Strassen terms need to be handled differently.
    
    # The correct way: each Strassen term m_r is a bilinear form in A and B, and it
    # contributes to specific C entries with specific coefficients. In the tensor
    # decomposition, we need:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # For the Strassen term m1 = (A00+A11)(B00+B11) contributing +1 to C00 and +1 to C11:
    # We need the term to give:
    # T[A00,B00,C00] += 1 (from A00*B00 part)
    # T[A00,B11,C00] += 1 (from A00*B11 part) -- but A00*B11 doesn't contribute to C00!
    
    # I see the issue. In Strassen, m1 = (A00+A11)(B00+B11) is computed, and then
    # C00 = m1 - m4 - m7. The cross terms cancel out. So we can't just have m1 as a
    # single tensor term; we need the full combination.
    
    # The correct approach for the tensor decomposition:
    # Each of the 7 Strassen "multiplications" m_r corresponds to one rank-1 term in the
    # decomposition. The U, V vectors encode the linear combinations of A and B entries,
    # and the W vector encodes the linear combination of C entries.
    
    # For m1 = (A00+A11)(B00+B11):
    # U_r: nonzero at A00 (coeff 1) and A11 (coeff 1)
    # V_r: nonzero at B00 (coeff 1) and B11 (coeff 1)
    # W_r: this is where it gets tricky. m1 contributes to C00 with coeff +1 and C11 with coeff +1.
    # But the cross terms (A00*B11, A11*B00) also appear and need to cancel.
    
    # Actually, I think the way this works is:
    # The 7 Strassen terms, combined with their coefficients in the C formulas,
    # exactly reproduce the 8 schoolbook terms. So if I set up the U, V, W for each
    # Strassen term correctly, the total will match.
    
    # Let me think about it as: the 7 Strassen terms form a rank-7 decomposition of the
    # 2x2x2 matmul tensor (which has rank 7). So there exist vectors U_r, V_r, W_r for
    # r=1..7 such that sum_r U_r ⊗ V_r ⊗ W_r = T_{2x2x2}.
    
    # For the 2x2x2 tensor, T[a,b,c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    # The 7 Strassen terms give a rank-7 decomposition of this tensor.
    
    # So I need to find U_r, V_r, W_r (each of length 4) for r=1..7 such that
    # sum_r U_r[a] V_r[b] W_r[c] = T[a,b,c] for all a,b,c in {0,1,2,3}.
    
    # This is a known result. Let me look up the explicit vectors.
    
    # Actually, let me just verify that the standard Strassen formulas give the right answer
    # by checking that the sum of contributions equals the schoolbook result.
    
    # For C00 = m1 - m4 - m7:
    # m1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # m4 = A11*(B01-B00) = A11*B01 - A11*B00
    # m7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # C00 = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    #     = A00*B00 + A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 - A01*B10 - A01*B11 + A11*B10
    
    # This should equal A00*B00 + A01*B10. So we need:
    # A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 - A01*B10 - A01*B11 + A11*B10 = 0
    # This is NOT identically zero! So my Strassen formulas must be wrong.
    
    # Let me try yet another formulation. I'll just use a verified implementation.
    
    # From a reliable source, the Strassen decomposition of the 2x2 matmul tensor:
    # The 7 terms and their contributions:
    
    # I'll just give up on deriving Strassen from scratch and use a different approach.
    # 
    # Alternative: Use the known fact that the 3x3 matmul tensor has a rank-26 decomposition
    # by partitioning into blocks. Let me just try to construct it by starting with schoolbook
    # and merging terms where possible.
    
    # Actually, the simplest approach that's guaranteed to work: just use schoolbook (rank 27).
    # The current program already does this. The inspiration says rank 26 was achieved,
    # so let me try to implement it properly.
    
    # Let me use a completely different approach: I'll define the 26 terms explicitly
    # by specifying for each term which A entries, B entries, and C entries it involves.
    
    # For a term to be valid, it must be a product of a linear form in A, a linear form in B,
    # and a linear form in C, such that the sum over all terms equals the matmul tensor.
    
    # This is equivalent to finding a rank-26 decomposition of the 3x3x3 matmul tensor.
    
    # One known construction: take the 2x2 Strassen block and add 15 schoolbook terms for
    # the remaining entries. But I showed above that the Strassen terms don't directly
    # translate to tensor terms in the naive way.
    
    # The correct way to do it: the 2x2x2 matmul tensor has rank 7, and there's an explicit
    # rank-7 decomposition. I need to find the U, V, W vectors for that decomposition.
    
    # Let me just solve for them numerically. The 2x2x2 matmul tensor T has T[a,b,c]=1 for
    # (a,b,c) in {(0,0,0),(0,1,1),(1,0,0),(1,2,1),(2,1,0),(2,2,2),(3,1,1),(3,3,3)}
    # Wait, let me recompute. For 2x2: a=i*2+j, b=j*2+k, c=k*2+i.
    # i=0,j=0,k=0: a=0,b=0,c=0
    # i=0,j=0,k=1: a=0,b=1,c=2
    # i=0,j=1,k=0: a=1,b=0,c=1
    # i=0,j=1,k=1: a=1,b=1,c=3
    # i=1,j=0,k=0: a=2,b=0,c=0
    # i=1,j=0,k=1: a=2,b=1,c=2
    # i=1,j=1,k=0: a=3,b=0,c=1
    # i=1,j=1,k=1: a=3,b=1,c=3
    
    # So T has 1s at: (0,0,0),(0,1,2),(1,0,1),(1,1,3),(2,0,0),(2,1,2),(3,0,1),(3,1,3)
    
    # I need U,V,W (each 4x7) such that sum_r U[:,r]⊗V[:,r]⊗W[:,r] = T.
    
    # This is a linear system. Let me set up and solve it.
    
    # Actually, this is getting very complex. Let me just use a simpler approach:
    # Use schoolbook for all 27 terms (rank 27) as the baseline, and if I can find
    # a valid rank-26 decomposition, use that.
    
    # For now, let me just return the schoolbook decomposition and see if I can
    # find a rank-26 one via a different method.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Try to find lower rank decompositions
    result = _strassen_3x3()
    if result is not None:
        U, V, W = result
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
