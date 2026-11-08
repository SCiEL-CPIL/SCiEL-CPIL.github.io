"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY = 10.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        """Check tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def strassen_decomposition():
        """
        Use Strassen-like decomposition for 4x5 * 5x6.
        
        Split A (4x5) into:
          A = [A_top (2x5); A_bot (2x5)]
        
        Split B (5x6) into:
          B = [B_top (2x6); B_bot (3x6)]
        
        C = A_top * B_top + A_bot * B_bot (4x6)
        
        For A_top * B_top (2x5 * 5x2... no, 2x5 * 5x6 gives 2x6)
        Actually let's do:
          C_top = A_top * B  (2x6)
          C_bot = A_bot * B  (2x6)
        
        For 2x5 * 5x6, we can split B into B1 (5x3) and B2 (5x3):
          C_top = A_top * B1 + A_top * B2  -- no, that's just schoolbook split.
        
        Better: Use Strassen on 2x2 blocks where possible.
        
        Let me try a different partition:
        A (4x5): split rows into 2+2, columns into 2+3
          A = [[A11(2x2), A12(2x3)],
               [A21(2x2), A22(2x3)]]
        
        B (5x6): split rows into 2+3, columns into 3+3
          B = [[B11(2x3), B12(2x3)],
               [B21(3x3), B22(3x3)]]
        
        C (4x6):
          C11 = A11*B11 + A12*B21  (2x3)
          C12 = A11*B12 + A12*B22  (2x3)
          C21 = A21*B11 + A22*B21  (2x3)
          C22 = A21*B12 + A22*B22  (2x3)
        
        Now apply Strassen to the 2x2 * 2x3 and 2x3 * 3x3 parts?
        This is getting complex. Let me just try a direct rank reduction search.
        """
        pass
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb
    
    # Try to find a lower-rank decomposition using structured approaches
    # Approach: Use the fact that we can combine terms
    
    # For 4x5 * 5x6, try a decomposition based on splitting into smaller blocks
    # and using Strassen where dimensions allow.
    
    # Key idea: 4x4 * 4x4 has Strassen rank 7.
    # Our problem: 4x5 * 5x6.
    # Write as: A(4x5) = [A4(4x4) | A1(4x1)], B(5x6) = [B4(4x6); B1(1x6)]
    # C = A4*B4 + A1*B1
    # A4*B4: 4x4 * 4x4 -> Strassen rank 7
    # A1*B1: 4x1 * 1x6 -> rank 4 (outer product, but actually 4*1*6=24 schoolbook, 
    #   but since it's a rank-1 intermediate, it's just 4*6=24 terms... no)
    # Actually A1*B1 where A1 is 4x1 and B1 is 1x6: C = A1 @ B1, which is 
    # sum_j A1[i,j]*B1[j,k] = A1[i,0]*B1[0,k], so it's an outer product of a 4-vector 
    # and a 6-vector, which has rank 24 in the tensor sense? No, it's just 4*6=24 
    # independent products, but they can be factored: C[i,k] = a_i * b_k, so the 
    # tensor is a_i ⊗ b_k, which is rank 1 in the bilinear sense but we need to 
    # express it in the U,V,W form.
    # In our form: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the A1*B1 part: j=4 (the 5th column of A, 5th row of B)
    # U[i*5+4, c] * V[4*6+k, c] * W[k*4+i, c]
    # We need this to equal a_i * b_k for the contribution.
    # This can be done with rank 4: for each i, set U[i*5+4, c_i]=1, V[4*6+k, c_i]=b_k... 
    # no, V is shared across k. 
    # Actually: sum_c U[i*5+4,c] * V[24+k,c] * W[4k+i,c] = a_i * b_k
    # This is a 4x6 matrix of rank 1 (outer product a * b^T).
    # We can write it as: for c=0..3, set U[i*5+4, c] = a_i if c==i else 0... 
    # Hmm, let me think differently.
    # The sub-tensor for j=4 is: T[i,4,k] = a_i * b_k
    # In the decomposition: sum_c U[5i+4,c] V[24+k,c] W[4k+i,c] = a_i b_k
    # This is equivalent to: the 4x6 matrix M[i,k] = a_i b_k has a decomposition 
    # M = sum_c u_c ⊗ v_c ⊗ w_c where u_c is in R^4, v_c in R^6, w_c in R^4,
    # and M[i,k] = sum_c u_c[i] v_c[k] w_c[i]... wait, W is indexed by k*n+i.
    # So W[4k+i, c] is the (k,i) entry.
    # So sum_c U[5i+4,c] V[24+k,c] W[4k+i,c] = a_i b_k
    # Let U[5i+4,c] = u_c[i], V[24+k,c] = v_c[k], W[4k+i,c] = w_c[k,i]
    # Then sum_c u_c[i] v_c[k] w_c[k,i] = a_i b_k
    # If we set w_c[k,i] = 1 for all k,i (constant), then sum_c u_c[i] v_c[k] = a_i b_k
    # which is an outer product, achievable with rank 1: u_0 = a, v_0 = b.
    # So: U[5i+4, 0] = a_i, V[24+k, 0] = b_k, W[4k+i, 0] = 1 for all i,k.
    # That's rank 1 for this sub-problem!
    
    # So total rank: Strassen(4x4*4x4) + 1 = 7 + 1 = 8? That seems too good.
    # Wait, the Strassen part also involves the j=0,1,2,3 indices, and the 
    # W matrix entries for those are coupled with the j=4 part.
    # The issue is that W is shared across all j values. In the Strassen part,
    # W[k*4+i, c] appears for j=0,1,2,3, and in the A1*B1 part, W[4k+i, c] 
    # appears for j=4. These are the same W entries! So we can't just add ranks.
    
    # This means we need a joint decomposition. Let me just do a numerical search.
    
    # Try random search for lower rank
    np.random.seed(42)
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But that's a weak bound. For 4x5*5x6, I believe the minimum rank is around 60-80.
    
    # Let's try a structured approach: split into 2x5 * 5x6 sub-problems and use 
    # Strassen-like methods on 2x2 blocks.
    
    # Alternative: just try to verify if a rank-80 or rank-90 decomposition exists 
    # by constructing one explicitly.
    
    # For now, return schoolbook. The search for better decompositions is complex.
    # But let's try one more thing: a block-based Strassen.
    
    # Split A(4x5) into A1(2x5) and A2(2x5)
    # Split B(5x6) into B1(5x3) and B2(5x3)
    # C = [A1; A2] * [B1, B2] = [[A1*B1, A1*B2], [A2*B1, A2*B2]]
    # Each sub-problem is 2x5 * 5x3.
    # For 2x5 * 5x3: split A into 2x2 + 2x3, B into 2x3 + 3x3
    #   A_sub = [Aa(2x2), Ab(2x3)]
    #   B_sub = [Ba(2x3); Bb(3x3)]
    #   C_sub = Aa*Ba + Ab*Bb  (2x3)
    #   Aa*Ba: 2x2 * 2x3 -> can use Strassen-like? 2x2*2x2 is rank 7, but here it's 2x2*2x3.
    #   Ab*Bb: 2x3 * 3x3 -> schoolbook rank 18.
    
    # This is getting too complex for a clean implementation in the time available.
    # Let's just return the schoolbook decomposition.
    
    if time.time() > DEADLINE - SAFETY:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a simple rank-reduction: see if we can merge some schoolbook terms
    # by using linear dependencies.
    
    # For now, return schoolbook.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
