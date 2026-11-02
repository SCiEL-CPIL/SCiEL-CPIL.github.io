import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Trivial decomposition as fallback (rank = n*m*p = 72)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    # Strassen-based decomposition for 3x3 * 3x8
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2 (rows 0-1, 2)
    # Actually, for 3x3: rows 0,1 form first group, row 2 is second; cols 0,1 first, col 2 second
    # A[i,j] with i in {0,1,2}, j in {0,1,2}
    # B[j,k] with j in {0,1,2}, k in {0,...,7}
    
    # Strategy: Use Strassen for the top-left 2x2 block of A (rows 0-1, cols 0-1)
    # and handle the remaining parts (row 2 of A, col 2 of A) separately
    
    # Let's think of it as:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x8, B12 is 2x8... 
    # Wait, B is 3x8, so B = [[B_top], [B_bottom]] where B_top is 2x8, B_bottom is 1x8
    
    # C = A * B = [[A11*B_top + A12*B_bottom], [A21*B_top + A22*B_bottom]]
    # C is 3x8: rows 0-1 from first part, row 2 from second part
    
    # For the 2x2 * 2x8 part (A11 * B_top), we can use Strassen-like approach
    # A11 is 2x2, B_top is 2x8
    # C_top = A11 * B_top is 2x8
    # C_top[i,k] = A11[i,0]*B_top[0,k] + A11[i,1]*B_top[1,k]
    # This is rank 4 (two independent products per row, 2 rows)
    # Or we can use Strassen: 7 multiplications for 2x2*2x2, but B is 2x8...
    # Actually for 2x2 * 2x8, the "Strassen" approach:
    # m1 = (A11[0,0]+A11[1,1]) * (B_top[0,:]+B_top[1,:])  -- but this gives sum, not individual
    # Standard Strassen for 2x2*2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # For 2x2 * 2x8, we can apply this column-wise or use a generalized form
    # Actually, the simplest approach: treat B_top as having 8 columns
    # The tensor for 2x2*2x8 has rank at most 2*2*8/2 = 16 (using Strassen-like)
    # But actually, we can do it in rank 7*8/... no, that's not right.
    
    # Let me just build it directly with Strassen for the 2x2*2x2 subproblems
    # For each pair of columns in B, we have a 2x2*2x2 problem
    
    # Actually, let's use a simpler structured approach:
    # Decompose A*3x8 into:
    # 1. A[0:2, 0:2] * B[0:2, :] using Strassen (7 mults per 2x2 block of B, but B has 8 cols)
    #    Actually, Strassen gives 7 scalar mults for 2x2*2x2. For 2x2*2x8, 
    #    we can view it as 4 separate 2x2*2x2 problems (cols 0-1, 2-3, 4-5, 6-7)
    #    That would be 7*4 = 28 mults, but we can do better by sharing.
    
    # Let me just do the straightforward thing: for the 2x2 * 2x8 part,
    # use the fact that it's bilinear and find a good decomposition.
    # The rank of the 2x2*2x8 matrix multiplication tensor is known to be 12 
    # (or we can just use rank 16 = 2*2*8/2).
    
    # For now, let's use a clean structured decomposition:
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    #
    # For k in {0,...,7}
    #
    # Total terms: 3*3*8 = 72 (trivial)
    #
    # Using Strassen on the 2x2 block (rows 0-1, cols 0-1 of A):
    # Let a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
    # Let b1k=B[0,k], b2k=B[1,k] for k=0..7
    # 
    # Strassen mults (7 per "2x2*2x2", but we have 8 columns so we share):
    # Actually, the 7 Strassen mults are:
    # m1 = (a11+a22) * (b1k+b2k)  -- but this is for each k, giving 8 mults
    # That doesn't help. The sharing in Strassen is across the 4 output entries.
    
    # For 2x2 * 2x8, the proper approach:
    # The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0..7}
    # T[i,j,k] = 1 (all ones)
    # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # 
    # One approach: Use the identity that for each k, we have a 2x2*2x1 problem
    # which has rank 3 (not 4) using:
    # c1 = (a11+a22)(b1+b2) - a22*b2 - a11*b1... no that's not right.
    # 2x2*2x1: c1 = a11*b1 + a21*b2, c2 = a12*b1 + a22*b2
    # This has rank 3: 
    #   m1 = a11*b1, m2 = a21*b2, m3 = (a12-a21)*b1... 
    # Actually 2x2*2x1 has rank 3 (known result).
    # So for 8 columns, that's 8*3 = 24 mults for the 2x2*2x8 part.
    # Plus the remaining terms: A[0,2]*B[2,k] and A[1,2]*B[2,k] and A[2,:]*B[:,k]
    # A[0,2]*B[2,k] for k=0..7: 8 mults
    # A[1,2]*B[2,k] for k=0..7: 8 mults  
    # A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] for k=0..7: 
    #   This is 1x3 * 3x8, rank 3 per k, so 3*8=24? Or we can share.
    #   Actually 1x3*3x8 has rank 3 (it's just 3 dot products shared across columns)
    #   Wait no: c_k = a0*b0k + a1*b1k + a2*b2k for each k
    #   The tensor is T[j,k] = 1 for j in {0,1,2}, k in {0..7}
    #   This has rank 3: m1 = a0 * b0k (for all k), m2 = a1 * b1k, m3 = a2 * b2k
    #   So 3 mults total (shared across all 8 columns)!
    
    # Let me recount:
    # Part 1: A[0:2, 0:2] * B[0:2, :] (2x2 * 2x8)
    #   For each k: c1k = a11*b1k + a21*b2k, c2k = a12*b1k + a22*b2k
    #   Rank 3 per k, but can we share across k? 
    #   The full tensor has entries T[i,j,k] for i in {0,1}, j in {0,1}, k in {0..7}
    #   T[i,j,k] = 1 for all
    #   We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    #   
    #   Approach: Use 3 mults per k-pair? Or 3 mults total shared?
    #   Actually, if we use:
    #   m1_k = a11 * b1k (for each k) -- 8 mults
    #   m2_k = a21 * b2k (for each k) -- 8 mults  
    #   m3_k = (a12-a21) * b1k -- 8 mults
    #   Then c1k = m1_k + m2_k, c2k = m3_k + m1_k
    #   Total: 24 mults for this part.
    #   
    #   Alternatively, can we do better? The 2x2*2x8 tensor...
    #   Let's just use 24 for now.
    
    # Part 2: A[0:2, 2] * B[2, :] (2x1 * 1x8)
    #   c0k += A[0,2]*B[2,k], c1k += A[1,2]*B[2,k]
    #   Rank 2 per k, but shared: 
    #   m_a = A[0,2], m_b = A[1,2], and B[2,k] is shared
    #   Actually: 2 mults total (A[0,2]*B[2,k] and A[1,2]*B[2,k] for all k)
    #   Wait, that's 2*8 = 16 if done naively, but:
    #   c0k += A[0,2]*B[2,k] for all k: this is 1 mult of A[0,2] times the vector B[2,:]
    #   In tensor decomposition terms: U[A[0,2]] * V[B[2,k]] * W[C[0,k]] 
    #   For each k, we need a separate term unless we can share.
    #   Actually in the tensor decomposition, each (i,j,k) triple that contributes 
    #   needs to be covered. The term A[0,2]*B[2,k] contributes to C[0,k].
    #   In the decomposition, this is one rank-1 term per k: 
    #   U[0*3+2, r] * V[2*8+k, r] * W[k*3+0, r] = 1
    #   So 8 separate terms for k=0..7.
    #   Total: 8 mults for A[0,2]*B[2,:] part.
    #   Similarly 8 for A[1,2]*B[2,:] part.
    
    # Part 3: A[2, :] * B[:, :] (1x3 * 3x8)
    #   c2k = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    #   Rank 3 total (shared across all k):
    #   m1 = A[2,0] * B[0,k] (for all k) -- but in tensor terms, this is 8 terms
    #   Hmm, let me reconsider.
    
    # OK let me just build it carefully.
    
    # In the tensor decomposition, we have:
    # T[i, j, k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # (since every A[i,j]*B[j,k] contributes to C[i,k])
    
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    
    # Let's build a decomposition with the following structure:
    
    terms = []  # list of (U_idx, V_idx, W_idx, coeff)
    
    # Part 1: A[0:2, 0:2] * B[0:2, :] -- use rank-3 per k decomposition
    # For each k in 0..7:
    #   C[0,k] gets A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] gets A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # Using: m1 = A[0,0]*B[0,k], m2 = A[1,0]*B[1,k], m3 = (A[0,1]-A[1,0])*B[0,k]
    #   C[0,k] = m1 + m2
    #   C[1,k] = m3 + m1
    # This uses 3 mults per k, total 24.
    
    # Actually, let me try a different grouping. For the 2x2*2x8 part:
    # I'll use the following 7-term Strassen-like decomposition per 2x2 block of B:
    # Group B columns into pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair (k1, k2), apply Strassen to the 2x2*2x2 subproblem:
    #   A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    #   B11 = B[0,k1], B12 = B[0,k2], B21 = B[1,k1], B22 = B[1,k2]
    #   m1 = (A11+A22)*(B11+B22)
    #   m2 = (A21+A22)*B11
    #   m3 = A11*(B21-B22)
    #   m4 = A22*(B12-B11)
    #   m5 = (A11+A12)*B22
    #   m6 = (A21-A11)*(B11+B12)
    #   m7 = (A12-A22)*(B21+B22)
    #   C[0,k1] = m1 + m4 - m5 + m7
    #   C[0,k2] = m3 + m5
    #   C[1,k1] = m2 + m4
    #   C[1,k2] = m1 - m2 + m3 + m6
    
    # This gives 7*4 = 28 mults for the 2x2*2x8 part. Worse than 24!
    # Let me stick with the 3-per-k approach: 24 mults.
    
    # Hmm, but 24 + 8 + 8 + 24 = 64. Let me verify.
    
    # Part 1: 2x2*2x8 with 3 mults per k = 24
    for k in range(8):
        # m1 = A[0,0]*B[0,k]
        terms.append((0*3+0, 0*8+k, k*3+0, 1))   # U[0,0], V[0,k], W[k,0]
        terms.append((1*3+0, 1*8+k, k*3+0, 1))   # U[1,0], V[1,k], W[k,0]  -- wait this is A[1,0]*B[1,k]
        # Hmm, I need to be more careful.
        pass
    
    # Let me restart with a cleaner approach.
    # 
    # I'll use the following decomposition:
    # 
    # For the full 3x3*3x8, I'll use:
    # - Strassen on the top-left 2x2 block of A with B's top 2 rows: 7 mults per 2-col block of B
    # - But since B has 8 columns, I'll do it per column pair: 4 pairs * 7 = 28
    # - A[0,2]*B[2,k] for k=0..7: 8 mults
    # - A[1,2]*B[2,k] for k=0..7: 8 mults
    # - A[2,:]*B[:,k] for k=0..7: 3 mults per k = 24 mults
    # Total: 28 + 8 + 8 + 24 = 68
    #
    # Actually, for A[2,:]*B[:,k], I can do better:
    # c2k = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    # This is a 1x3 dot product with 3x8 matrix.
    # In tensor terms, each (j,k) pair where j in {0,1,2} contributes.
    # T[j,k] = 1 for all j,k.
    # We need sum_r U[2*3+j, r] * V[j*8+k, r] * W[k*3+2, r] = 1
    # One approach: 3 mults shared across all k:
    #   r1: U[2*3+0, r1]=1, V[0*8+k, r1]=1 for all k, W[k*3+2, r1]=1 for all k
    #   But V[0*8+k, r1] = 1 for all k means one term covers all k for j=0.
    #   Similarly for j=1 and j=2.
    #   So 3 terms total for this part!
    #
    # Wait, that's only if the W coefficient can be 1 for all k. Let me check:
    # Term r1: U[6, r1] * V[0*8+k, r1] * W[k*3+2, r1] should contribute 1 to T[0,k] for all k
    # If U[6,r1]=1, V[0*8+k,r1]=1 for all k, W[k*3+2,r1]=1 for all k, then yes.
    # But V[0*8+k, r1] = 1 for all k means the V vector has 1s in positions 0,1,...,7.
    # That's fine! It's a valid entry.
    
    # Similarly for Part 2 (A[0,2]*B[2,k]):
    # T[2,k] = 1 for all k (contributing to C[0,k] and C[1,k])
    # For C[0,k]: need U[0*3+2, r] * V[2*8+k, r] * W[k*3+0, r] to sum to 1
    # If I use one term: U[2, r]=1, V[2*8+k, r]=1 for all k, W[k*3+0, r]=1 for all k
    # That works! 1 term for A[0,2]*B[2,:] contributing to C[0,:]
    # Similarly 1 term for A[1,2]*B[2,:] contributing to C[1,:]
    
    # Revised count:
    # Part 1 (2x2*2x8): 28 mults (Strassen, 4 pairs of columns)
    # Part 2: A[0,2]*B[2,:] -> C[0,:]: 1 mult (shared across all k)
    # Part 3: A[1,2]*B[2,:] -> C[1,:]: 1 mult
    # Part 4: A[2,:]*B[:,k] -> C[2,:]: 3 mults (one per j, shared across all k)
    # Total: 28 + 1 + 1 + 3 = 33?? That seems too good.
    
    # Wait, I need to double-check. The issue is whether we can truly share.
    # In the tensor decomposition, each rank-1 term is:
    #   u_i * v_j * w_k for all (i,j,k)
    # The constraint is that for each (i,j,k), the sum over r equals T[i,j,k].
    
    # For Part 2: We want T[2,k] = 1 for all k, contributing to C[0,k].
    # Term: U[2, r] = 1 (i.e., A[0,2] coefficient), V[2*8+k, r] = 1 for all k, W[k*3+0, r] = 1 for all k
    # This gives contribution 1 to T[2,k] for all k. 
    # But does it interfere with other terms? Only if other terms also have nonzero 
    # entries at (U_idx=2, V_idx=2*8+k, W_idx=k*3+0).
    # In our decomposition, only this term should contribute to T[2,k] for the C[0,k] output.
    # So yes, 1 term works!
    
    # For Part 4: A[2,j]*B[j,k] -> C[2,k] for j in {0,1,2}
    # Term for j=0: U[2*3+0, r]=1, V[0*8+k, r]=1 for all k, W[k*3+2, r]=1 for all k
    # This gives T[0,k] += 1 for all k (for the C[2,k] output)
    # Similarly for j=1 and j=2.
    # Total: 3 terms.
    
    # So total rank = 28 + 1 + 1 + 3 = 33? Let me verify this is correct.
    
    # Actually wait - I need to be more careful about the Strassen part.
    # The Strassen decomposition for 2x2*2x2 gives 7 mults that produce 4 outputs.
    # For 2x2*2x8, if I group columns into 4 pairs, I get 4*7=28 mults producing 4*4=16 outputs.
    # The 16 outputs correspond to C[0,k] and C[1,k] for k=0..7 (8 each, total 16). ✓
    
    # Let me now build the full decomposition.
    
    terms = []
    
    # Part 1: Strassen on A[0:2,0:2] * B[0:2,:] with column pairs
    # Column pairs: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1])
        # This is a product of (a11+a22) and (b1k1+b2k1)
        # In tensor terms: we need U entries for a11 and a22, V entries for b1k1 and b2k1
        # m1 contributes to C[0,k1], C[0,k2], C[1,k1], C[1,k2]
        # C[0,k1] += m1, C[0,k2] -= m1... wait let me recheck Strassen.
        
        # Standard Strassen:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b21-b22)
        # m4 = a22(b12-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1+m4-m5+m7
        # c12 = m3+m5
        # c21 = m2+m4
        # c22 = m1-m2+m3+m6
        
        # Here a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,k1], b12=B[0,k2], b21=B[1,k1], b22=B[1,k2]
        # c11=C[0,k1], c12=C[0,k2], c21=C[1,k1], c22=C[1,k2]
        
        # Each m is a product of a linear combo of A entries and a linear combo of B entries.
        # In the tensor decomposition, each "m" corresponds to one rank-1 term
        # where U has the A-combo coefficients, V has the B-combo coefficients,
        # and W distributes to the appropriate C entries.
        
        # m1 = (a11+a22)(b11+b22):
        # U: A[0,0] coeff 1, A[1,1] coeff 1
        # V: B[0,k1] coeff 1, B[1,k1] coeff 1, B[0,k2] coeff 1, B[1,k2] coeff 1
        # Wait no! m1 = (a11+a22)*(b11+b22) where b11=B[0,k1], b22=B[1,k2]
        # So V: B[0,k1] coeff 1, B[1,k2] coeff 1
        # W: C[0,k1] coeff 1, C[0,k2] coeff 1, C[1,k1] coeff 1, C[1,k2] coeff -1
        # (from c11+=m1, c12-=... wait c12 = m3+m5, no m1 in c12)
        # c11 = m1+m4-m5+m7: m1 coeff +1
        # c12 = m3+m5: m1 coeff 0
        # c21 = m2+m4: m1 coeff 0
        # c22 = m1-m2+m3+m6: m1 coeff +1
        # So W: C[0,k1]=+1, C[1,k2]=+1, others 0 for this term.
        
        # This is getting complex. Let me just enumerate all terms explicitly.
        pass
    
    # This is getting very complex to do by hand. Let me use a programmatic approach.
    # I'll build the decomposition using a helper that adds Strassen terms.
    
    def add_strassen_2x2_2x2(terms, a_indices, b_indices, c_indices, k1, k2):
        """Add 7 Strassen terms for 2x2*2x2 multiplication.
        a_indices: [(i,j), (i,j), (i,j), (i,j)] for a11,a12,a21,a22
        b_indices: [(j,k), (j,k), (j,k), (j,k)] for b11,b12,b21,b22
        c_indices: [(i,k), (i,k), (i,k), (i,k)] for c11,c12,c21,c22
        """
        a11, a12, a21, a22 = a_indices
        b11, b12, b21, b22 = b_indices
        c11, c12, c21, c22 = c_indices
        
        # m1 = (a11+a22)(b11+b22)
        U = {a11: 1, a22: 1}
        V = {b11: 1, b22: 1}
        W = {c11: 1, c22: 1}
        terms.append((U, V, W))
        
        # m2 = (a21+a22)b11
        U = {a21: 1, a22: 1}
        V = {b11: 1}
        W = {c21: 1, c22: -1}
        terms.append((U, V, W))
        
        # m3 = a11(b21-b22)
        U = {a11: 1}
        V = {b21: 1, b22: -1}
        W = {c12: 1, c22: 1}
        terms.append((U, V, W))
        
        # m4 = a22(b12-b11)
        U = {a22: 1}
        V = {b12: 1, b11: -1}
        W = {c11: 1, c21: 1}
        terms.append((U, V, W))
        
        # m5 = (a11+a12)b22
        U = {a11: 1, a12: 1}
        V = {b22: 1}
        W = {c11: -1, c12: 1}
        terms.append((U, V, W))
        
        # m6 = (a21-a11)(b11+b12)
        U = {a21: 1, a11: -1}
        V = {b11: 1, b12: 1}
        W = {c22: 1}
        terms.append((U, V, W))
        
        # m7 = (a12-a22)(b21+b22)
        U = {a12: 1, a22: -1}
        V = {b21: 1, b22: 1}
        W = {c11: 1}
        terms.append((U, V, W))
    
    # A indices: A[i,j] -> index i*m+j = i*3+j
    # B indices: B[j,k] -> index j*p+k = j*8+k
    # C indices: C[i,k] -> index k*n+i = k*3+i (W convention: W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,:] 
    # 4 column pairs
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        a_indices = [(0,0), (0,1), (1,0), (1,1)]  # a11, a12, a21, a22
        b_indices = [(0,k1), (0,k2), (1,k1), (1,k2)]  # b11, b12, b21, b22
        c_indices = [(0,k1), (0,k2), (1,k1), (1,k2)]  # c11, c12, c21, c22
        add_strassen_2x2_2x2(terms, a_indices, b_indices, c_indices, k1, k2)
    
    # Part 2: A[0,2]*B[2,k] -> C[0,k] for all k
    # One shared term
    U = {(0,2): 1}
    V = {(2,k): 1 for k in range(8)}
    W = {(0,k): 1 for k in range(8)}
    terms.append((U, V, W))
    
    # Part 3: A[1,2]*B[2,k] -> C[1,k] for all k
    U = {(1,2): 1}
    V = {(2,k): 1 for k in range(8)}
    W = {(1,k): 1 for k in range(8)}
    terms.append((U, V, W))
    
    # Part 4: A[2,j]*B[j,k] -> C[2,k] for j in {0,1,2}
    for j in range(3):
        U = {(2,j): 1}
        V = {(j,k): 1 for k in range(8)}
        W = {(2,k): 1 for k in range(8)}
        terms.append((U, V, W))
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r, (U_dict, V_dict, W_dict) in enumerate(terms):
        for (i, j), coeff in U_dict.items():
            U_mat[i * m + j][r] = coeff
        for (j, k), coeff in V_dict.items():
            V_mat[j * p + k][r] = coeff
        for (i, k), coeff in W_dict.items():
            W_mat[k * n + i][r] = coeff
    
    # Verify the decomposition
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r] for r in range(rank))
                max_error = max(max_error, abs(val - 1))
    
    if max_error < 1e-9:
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    else:
        # Fallback to trivial
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
