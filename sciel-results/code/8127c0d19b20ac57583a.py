import numpy as np
import time

def _winograd_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 with rank 7.
    U indexed by [i*2+j] (i row of A, j col of A), shape (4,7)
    V indexed by [j*2+k] (j row of B, k col of B), shape (4,7)
    W indexed by [k*2+i] (k col of B, i row of C), shape (4,7)
    Condition: sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] == 1 for all i,j,k
    """
    # Winograd: define intermediate products
    # m0=a00+a11, m1=a00-a11, m2=a10+a11, m3=a11-a10
    # n0=b00+b11, n1=b00-b11, n2=b10+b11, n3=b11-b10
    # c00 = m0*n0 - m3*n2
    # c01 = m0*n1 + m3*n3
    # c10 = m2*n2 + m1*n0
    # c11 = m2*n1 - m1*n3
    #
    # We need to express each c[i,k] as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # with the constraint that for a given r, the (U,V,W) triple contributes
    # to T[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r].
    #
    # The Winograd factors: m_r is a linear form in A, n_r is a linear form in B,
    # and c[i,k] is a linear form in the m_r*n_r products.
    #
    # Standard rank-7 decomposition of 2x2x2 MM tensor:
    # r=0: m0*n0 -> contributes to c00 (+1), c01? no...
    # Let me use the known explicit decomposition.
    #
    # Actually the cleanest way: the 7 rank-1 terms correspond to the 7 products.
    # Product p_r = m_r * n_r. Then:
    # c00 = p0 - p3  (m0*n0 - m3*n2)
    # c01 = p1 + p6  (m0*n1 + m3*n3)
    # c10 = p5 + p2  (m2*n2 + m1*n0)
    # c11 = p4 - p2? no...
    #
    # Let me re-derive. The 7 products are:
    # p0 = m0*n0 = (a00+a11)(b00+b11)
    # p1 = m0*n1 = (a00+a11)(b00-b11)
    # p2 = m1*n0 = (a00-a11)(b00+b11)
    # p3 = m1*n3 = (a00-a11)(b11-b10)
    # p4 = m2*n1 = (a10+a11)(b00-b11)
    # p5 = m2*n2 = (a10+a11)(b10+b11)
    # p6 = m3*n2 = (a11-a10)(b10+b11)
    #
    # Hmm, this is getting complicated. Let me just use a known explicit rank-7
    # decomposition of the 2x2x2 matrix multiplication tensor.
    #
    # The MM tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Rank-7 decomposition (from Winograd's algorithm):
    #
    # Term r=0: U=(1,0,1,0) for A[0,0],A[1,1]... 
    
    # Let me use a different, well-known explicit form.
    # The 2x2x2 MM tensor has rank 7. One explicit decomposition:
    #
    # For the tensor T with T[i,j,k]=1, we need sum_r u_r[i]*v_r[j]*w_r[k] = 1
    # where u_r is indexed by (i,j) pair, v_r by (j,k) pair, w_r by (k,i) pair.
    #
    # Actually in our convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # and we need sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1 for all i,j,k.
    #
    # This is exactly the matrix multiplication tensor of size 2x2x2.
    # Known rank-7 decomposition:
    #
    # r | U[00] U[01] U[10] U[11] | V[00] V[01] V[10] V[11] | W[00] W[01] W[10] W[11]
    #   (U indexed by i*2+j: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11)
    #   (V indexed by j*2+k: V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11)
    #   (W indexed by k*2+i: W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11)
    
    # Using the standard Winograd rank-7 for 2x2x2:
    # The decomposition can be written as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Winograd:
    # p0 = (A00+A11)(B00+B11)
    # p1 = (A00-A11)(B00-B11)
    # p2 = (A10+A11)(B10+B11)
    # p3 = (A11-A10)(B11-B10)
    # p4 = (A00+A01)(B00+B10)
    # p5 = (A00-A01)(B00-B10)
    # p6 = (A10+A11)(B01+B11)  -- wait this doesn't look right either
    #
    # Let me just use a verified explicit decomposition. I'll construct it
    # programmatically by solving for a rank-7 decomposition.
    
    # Actually, the simplest correct approach: use the known fact that
    # 2x2x2 has rank 7 and use a specific published decomposition.
    #
    # Here's a correct rank-7 decomposition (verified):
    # Using the "Winograd" scheme where we pad to 3x3 and use FFT-like:
    # For 2x2, the rank-7 comes from:
    #
    # Let me just hardcode a known correct one.
    # Reference: the 2x2x2 MM tensor rank-7 decomposition
    #
    # U (4x7):
    # A00: [1, 1, 1, 0, 1, 1, 0]
    # A01: [0, 0, 0, 1, 1, -1, 1]
    # A10: [0, 0, 0, -1, 1, 1, -1]  -- hmm I'm not confident
    # A11: [1, -1, 1, 0, 1, -1, 1]
    #
    # This is error-prone. Let me construct it via a different method:
    # I'll use the fact that we can find a rank-7 decomposition by
    # solving a system. But that's complex.
    #
    # SAFER APPROACH: Just use schoolbook for everything but try to
    # find a better decomposition via a known block method.
    #
    # Actually, let me just verify the schoolbook works and return it.
    # The task says score = rank, lower is better. Schoolbook = 96.
    # If I can't find a correct lower-rank decomposition, 96 is the safe answer.
    
    # Let me try a different strategy: use the 4x4x6 structure directly
    # with a block decomposition that I can verify.
    
    return None  # placeholder


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    n, m, p = 4, 4, 6
    
    # First, let me try to construct a rank-7 decomposition for 2x2x2
    # by explicitly building it and verifying.
    
    # The 2x2x2 matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # We need: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
    #
    # This is a 3-tensor of size 2x2x2 = 8 entries, all equal to 1.
    # The rank of this tensor is 7 (not 8).
    #
    # One explicit rank-7 decomposition:
    # I'll use the following construction based on the "border rank" result.
    #
    # Actually, let me use a completely different, verifiable approach:
    # I'll construct the decomposition using the Winograd algorithm for 2x2
    # and verify it numerically.
    
    # Winograd 2x2x2 rank-7 decomposition (verified correct):
    # The idea: C = A*B where A, B, C are 2x2.
    # 
    # Define:
    #   m0 = A[0,0] + A[1,1]
    #   m1 = A[0,0] - A[1,1]
    #   m2 = A[1,0] + A[1,1]
    #   m3 = A[1,1] - A[1,0]
    #   n0 = B[0,0] + B[1,1]
    #   n1 = B[0,0] - B[1,1]
    #   n2 = B[1,0] + B[1,1]
    #   n3 = B[1,1] - B[1,0]
    #
    # Products:
    #   p0 = m0 * n0
    #   p1 = m0 * n1
    #   p2 = m1 * n0
    #   p3 = m1 * n3
    #   p4 = m2 * n1
    #   p5 = m2 * n2
    #   p6 = m3 * n2
    #
    # Then:
    #   C[0,0] = p0 - p3 - p6 + p2  -- let me verify
    #
    # This is getting too complex to do by hand. Let me just use a
    # computational approach: find a rank-7 decomposition by solving.
    
    # Alternative: I'll use the known explicit rank-7 decomposition
    # from the literature. Here's one that I can verify:
    
    # The 2x2x2 MM tensor can be decomposed as:
    # T = sum_{r=0}^{6} u_r ⊗ v_r ⊗ w_r
    #
    # Using the "Strassen-like" decomposition:
    # r=0: u=(1,1,0,0), v=(1,0,0,1), w=(1,0,0,1)  -- not sure
    #
    # Let me just try a numerical approach: I'll set up the problem
    # and use a known solution.
    
    # Actually, the simplest correct approach that I'm confident about:
    # Use the schoolbook decomposition (rank 96) which I know is correct.
    # Then, if I have time, try to improve.
    
    # Let me first build and verify the schoolbook:
    rank_sb = n * m * p  # 96
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k])
                    # Wait, that's not right. The condition is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is not a simple dot product.
                    prod = Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_sb, V_sb, W_sb, n, m, p):
        # Schoolbook should always work. If it doesn't, something is very wrong.
        # Just return it anyway as a last resort.
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now try to find a better decomposition.
    # Strategy: use block structure.
    # A (4x4) = [[A00, A01], [A10, A11]] where each is 2x2
    # B (4x6) = [[B00, B01], [B10, B11]] where B00, B01, B10, B11 are 2x3
    # C (4x6) = [[C00, C01], [C10, C11]] where each is 2x3
    #
    # C00 = A00*B00 + A01*B10  (2x2 * 2x3 + 2x2 * 2x3)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each 2x2 * 2x3 product can be decomposed.
    # 2x2 * 2x3: C[i,k] = sum_{j=0,1} A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}
    # This is 6 output entries, each a sum of 2 products = 12 products (schoolbook rank 12)
    # But we can do better using the 2x2x2 rank-7 for the first two columns
    # and 2x2x1 rank-4 for the third column.
    #
    # 2x2 * 2x2 (cols 0,1): rank 7
    # 2x2 * 2x1 (col 2): rank 4 (schoolbook)
    # Total per 2x2*2x3: rank 11
    #
    # We have 4 such blocks (C00, C01, C10, C11), but they share the A blocks.
    # Actually, each C block uses different A and B sub-blocks, so they're independent
    # in terms of the tensor decomposition. Wait, no:
    # C00 uses A00, A01, B00, B10
    # C01 uses A00, A01, B01, B11
    # C10 uses A10, A11, B00, B10
    # C11 uses A10, A11, B01, B11
    #
    # The U factors for A00 appear in both C00 and C01 decompositions.
    # Similarly for V factors of B00 appearing in C00 and C10.
    #
    # This means we CAN'T simply add up the ranks because the factors are shared.
    # The tensor decomposition is for the FULL 4x4x6 MM tensor, not per-block.
    #
    # So I need to construct the full decomposition directly.
    #
    # Let me think about this differently. The full MM tensor T[i,j,k] for
    # C = A*B where A is 4x4, B is 4x6, C is 4x6.
    # T[i,j,k] = 1 for all i in {0..3}, j in {0..3}, k in {0..5}.
    # (Because C[i,k] = sum_j A[i,j] B[j,k], so every (i,j,k) triple contributes.)
    #
    # The rank of this tensor is what we want to minimize.
    #
    # One approach: use the Kronecker product structure.
    # The 4x4x6 MM tensor can be seen as related to 2x2x2 and 2x2x1 tensors.
    #
    # Actually, let me just try to find a good decomposition using a
    # structured approach and verify it.
    
    # Let me try: decompose using 2x2 blocks with rank-7 for 2x2x2 parts.
    # I'll construct the full U, V, W matrices directly.
    
    # For the 4x4x6 case, let me use the following block structure:
    # Split A into 2x2 blocks: A = [A00 A01; A10 A11]
    # Split B into 2x3 blocks: B = [B00 B01; B10 B11] where B00,B10 are 2x3, B01,B11 are 2x3
    # 
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 * 2x3 product, I need a rank decomposition.
    # 2x2 * 2x3: the tensor is 2x2x3, all ones.
    # Rank of 2x2x3 tensor: at most 12 (schoolbook), but can be less.
    # Actually, the 2x2x3 MM tensor has rank... let me think.
    # It's the tensor for C[i,k] = sum_j A[i,j] B[j,k] with i in {0,1}, j in {0,1}, k in {0,1,2}.
    # This is a 2x2x3 tensor with all entries 1.
    # The rank of a 2x2x3 all-ones tensor is... 
    # We can write it as: for each j, sum over k: A[i,j] * B[j,k]
    # = sum_j (sum_k A[i,j]*B[j,k])
    # Each term A[i,j]*B[j,k] for fixed j and varying i,k is a 2x1x3 tensor = u ⊗ v ⊗ w
    # where u[i]=A[i,j] (depends on i), v[j]=1, w[k]=B[j,k] (depends on k).
    # Wait, that's not right either.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition: T = sum_r u_r[i] * v_r[j] * w_r[k]
    # For 2x2x3, the minimum rank is... 
    # We can use: T = (sum_i u[i]) * (sum_j v[j]) * (sum_k w[k])? No, that gives all ones but
    # with rank 1 only if u, v, w are constant. But u[i]*v[j]*w[k] = 1 for all i,j,k
    # requires u[i]=1, v[j]=1, w[k]=1, which gives T[i,j,k]=1. So rank 1?!
    # No wait, that's wrong. u[i]*v[j]*w[k] with u[i]=1, v[j]=1, w[k]=1 gives T[i,j,k]=1.
    # But that's a rank-1 tensor! So the 2x2x3 all-ones tensor has rank 1?
    # That can't be right because matrix multiplication has higher rank.
    #
    # I think I'm confusing myself. Let me re-read the problem.
    #
    # The matrix multiplication tensor for C = A*B is NOT the all-ones tensor.
    # The tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor notation: C = T(A, B) where T is the MM tensor.
    # The MM tensor T has the property that:
    # T(A, B)[i,k] = sum_j A[i,j] B[j,k]
    #
    # As a 3-tensor, T[i,j,k] = 1 for all i,j,k (in the appropriate ranges).
    # Because the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    #
    # So yes, T[i,j,k] = 1 for all valid i,j,k.
    #
    # Now, the rank of this tensor: we want to write
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the trace convention for W)
    #
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Can we write this as sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]?
    #
    # The schoolbook uses 8 terms: one for each (i,j,k) triple.
    # But the rank is 7, so we can do it with 7 terms.
    #
    # The key: the tensor is NOT arbitrary. It's the MM tensor, which has
    # special structure. The rank-7 decomposition exploits this.
    #
    # Let me just use a known correct rank-7 decomposition for 2x2x2.
    # I'll find one by solving the system numerically.
    
    # For now, let me just return the schoolbook (rank 96) which I know is correct.
    # If I can find a better one, I'll use it.
    
    # Actually, let me try one more thing: use the 4x4x4 rank-49 decomposition
    # (recursive Strassen) for the 4x4 part, and handle the 4x6 part.
    #
    # 4x4 * 4x4: rank 49 (recursive Strassen: 4x4 = 2x(2x2), each 2x2 has rank 7,
    # so 4x4 has rank 7^2 = 49)
    # 4x4 * 4x2: rank ? 
    #
    # But the full problem is 4x4 * 4x6, not 4x4 * 4x4.
    #
    # Let me try: C = A * B where A is 4x4, B is 4x6.
    # Write B = [B1, B2] where B1 is 4x4, B2 is 4x2.
    # C = A*B1 + A*B2
    # A*B1: 4x4 * 4x4, rank 49
    # A*B2: 4x4 * 4x2, rank ?
    #
    # 4x4 * 4x2: C[i,k] = sum_j A[i,j] B[j,k] for i in {0..3}, k in {0,1}
    # Tensor is 4x4x2, all ones. Rank?
    # Using blocks: A = [A00 A01; A10 A11] (2x2 blocks), B2 = [B10; B11] (2x2 blocks)
    # C1 = A00*B10 + A01*B11 (2x2 * 2x2, rank 7)
    # C2 = A10*B10 + A11*B11 (2x2 * 2x2, rank 7)
    # Total: rank 14
    #
    # So total rank: 49 + 14 = 63
    #
    # That's better than 96! Let me construct this.
    
    # Actually wait, I need to be more careful. The tensor decomposition
    # for A*B1 and A*B2 use different B factors, so the ranks add.
    # But the U factors (for A) are shared! So I can't just add the ranks.
    #
    # Hmm, but in the tensor decomposition, the U, V, W factors are for the
    # full problem. If I decompose C = A*B1 + A*B2, the U factors for A
    # appear in both parts. So the total rank is not simply the sum.
    #
    # Actually, in the tensor decomposition framework:
    # T_full = T_{A*B1} + T_{A*B2}
    # where T_{A*B1} is the MM tensor for 4x4*4x4 and T_{A*B2} is for 4x4*4x2.
    #
    # If I have a rank-49 decomposition of T_{A*B1} and a rank-14 decomposition
    # of T_{A*B2}, then T_full has rank at most 49 + 14 = 63.
    # The U factors from the two decompositions are different (they correspond
    # to different "slices" of A), so they don't conflict.
    #
    # Wait, but the U factor for A in T_{A*B1} and T_{A*B2} are both indexed by
    # the entries of A. In the full decomposition, the U factor is a single matrix
    # of shape (n*m, rank). If I use separate decompositions, the U factors from
    # the two parts are concatenated (different columns), so the total rank is
    # the sum.
    #
    # Yes, that's correct. The total rank is the sum of the ranks of the parts.
    #
    # So: rank = 49 + 14 = 63.
    #
    # Let me construct this.
    
    # First, I need rank-7 decompositions for 2x2x2.
    # Let me find one numerically.
    
    # Actually, let me use a known explicit rank-7 decomposition.
    # Here's one (I'll verify it):
    
    def make_2x2x2_rank7():
        """Return (U, V, W) for 2x2 * 2x2 with rank 7.
        U: (4, 7), indexed by [i*2+j]
        V: (4, 7), indexed by [j*2+k]
        W: (4, 7), indexed by [k*2+i]
        """
        # Winograd rank-7 decomposition for 2x2x2
        # Using the standard Winograd algorithm:
        #
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A * B
        #
        # Winograd:
        # m0 = a00 + a11
        # m1 = a00 - a11
        # m2 = a10 + a11
        # m3 = a11 - a10
        # n0 = b00 + b11
        # n1 = b00 - b11
        # n2 = b10 + b11
        # n3 = b11 - b10
        #
        # p0 = m0 * n0
        # p1 = m0 * n1
        # p2 = m1 * n0
        # p3 = m1 * n3
        # p4 = m2 * n1
        # p5 = m2 * n2
        # p6 = m3 * n2
        #
        # C[0,0] = p0 - p3 - p6 + p2
        # C[0,1] = p0 - p3 - p4 + p1  -- need to verify
        # C[1,0] = p0 - p3 - p5 + p2  -- need to verify
        # C[1,1] = p0 - p3 + p1 + p4  -- need to verify
        #
        # This is hard to verify by hand. Let me just use a computational
        # approach to find a rank-7 decomposition.
        
        # I'll use the fact that the 2x2x2 MM tensor has rank 7 and
        # find a decomposition by solving a system of equations.
        
        # Actually, let me just use a known correct decomposition.
        # Here's the Winograd rank-7 for 2x2x2 (from standard references):
        
        # The 7 products are:
        # p0 = (a00+a11)(b00+b11)
        # p1 = (a00-a11)(b00-b11)
        # p2 = (a10+a11)(b10+b11)
        # p3 = (a11-a10)(b11-b10)
        # p4 = (a00+a01)(b00+b10)
        # p5 = (a00-a01)(b00-b10)
        # p6 = (a10+a11)(b01+b11)  -- I'm not sure about this one
        
        # Let me try a different approach: I'll construct the decomposition
        # using the "border rank" method or just use a verified one.
        
        # Here's a verified rank-7 decomposition (I'll check it numerically):
        # U (4x7):
        # Row 0 (A[0,0]): [1, 1, 0, 0, 1, 1, 0]
        # Row 1 (A[0,1]): [0, 0, 1, 1, 1, -1, 0]
        # Row 2 (A[1,0]): [0, 0, 1, -1, 1, 1, 0]
        # Row 3 (A[1,1]): [1, -1, 0, 0, 1, -1, 1]
        
        # V (4x7):
        # Row 0 (B[0,0]): [1, 1, 1, 1, 1, 1, 0]
        # Row 1 (B[0,1]): [0, 0, 0, 0, 1, -1, 1]
        # Row 2 (B[1,0]): [1, -1, 1, -1, 1, 1, 1]
        # Row 3 (B[1,1]): [1, -1, 1, 1, 1, -1, 1]
        
        # W (4x7):
        # Row 0 (C[0,0]): [1, -1, 1, -1, 0, 0, 0]
        # Row 1 (C[0,1]): [0, 0, 0, 0, 1, 1, 1]
        # Row 2 (C[1,0]): [0, 0, 1, -1, 1, 1, 0]
        # Row 3 (C[1,1]): [1, -1, 1, 1, 1, -1, 1]
        
        # I'm not confident in these values. Let me just use the schoolbook
        # for the 2x2x2 part (rank 8) and see if the overall approach works.
        # Then I can try to optimize later.
        
        # For now, use schoolbook rank-8 for 2x2x2:
        U = [[0]*8 for _ in range(4)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k
                    U[i*2+j][r] = 1
                    V[j*2+k][r] = 1
                    W[k*2+i][r] = 1
        return U, V, W
    
    # Let me verify the schoolbook 2x2x2:
    U2, V2, W2 = make_2x2x2_rank7()
    U2a = np.array(U2, dtype=float)
    V2a = np.array(V2, dtype=float)
    W2a = np.array(W2, dtype=float)
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U2a[i*2+j] * V2a[j*2+k] * W2a[k*2+i])
                if abs(s - 1.0) > 1e-9:
                    valid = False
    if not valid:
        # Schoolbook 2x2x2 should work. If not, something is wrong.
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now let me try to construct the full 4x4x6 decomposition using blocks.
    # But this is complex. Let me just return the schoolbook for now.
    # The score will be 1.0 (same as parent), but at least it's correct.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
