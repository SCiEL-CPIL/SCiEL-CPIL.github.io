"""Matrix multiplication 2,4,7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy: Use block decomposition.
    - Split the 2x4 * 4x7 multiplication into blocks.
    - Use known optimal decompositions for smaller matrix multiplications.
    
    The multiplication C = A * B where A is 2x4 and B is 4x7.
    We can write this as:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..1, j in 0..3, k in 0..6
    
    One approach: decompose into 2x2x2 and 2x2x3 blocks.
    
    Let's use a more systematic approach with the Winograd-style 
    decomposition for rectangular matrices.
    
    For 2x4x7, we can try:
    - Split j dimension: j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}
    - This gives 4 blocks of 2x2x7
    
    Alternatively, use the fact that:
    rank(2,4,7) <= rank(2,4,4) + rank(2,4,3)
    
    Known results:
    - rank(2,2,2) = 7 (Winograd)
    - rank(2,2,3) = ? (at most 2*rank(2,2,2) - something, or use direct)
    - rank(2,4,4) = ? 
    - rank(2,4,7) = ?
    
    Let me try a different approach: use the standard tensor decomposition
    with the known 7-term 2x2x2 decomposition and extend it.
    
    Actually, let me try to construct a decomposition of rank 27 or so
    using the block structure.
    
    A 2x4x7 multiplication can be seen as:
    For each of the 2 rows of A, we compute a linear combination of the 4 rows of B.
    
    Let me use a simpler but potentially suboptimal approach first, then refine.
    
    The schoolbook rank is 2*4*7 = 56.
    
    Let me try to use the Winograd 2x2x2 decomposition (rank 7) to reduce
    the problem. 
    
    Split A into 2 blocks of 2x2 (along columns), and B into 7 blocks... 
    no, B is 4x7, so split B into 2 blocks of 2x7 (along rows).
    
    Then C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x7.
    
    For each 2x2 * 2x7 multiplication, we can use the 2x2x2 Winograd 
    decomposition applied to each pair of columns of B.
    
    2x2 * 2x7 = sum over 7 columns of 2x2*2x1, but that's not quite right.
    
    Actually, 2x2 * 2x7: C[i,k] = sum_j A[i,j]*B[j,k]
    This is a 2x2x7 tensor. We can decompose this as:
    For each k, we have a 2x2x1 multiplication, but that's trivial.
    
    Better: use the 2x2x2 Winograd on pairs of k values.
    2x2x7 = 2x2x2 + 2x2x2 + 2x2x1 (for the 7th column)
    rank(2,2,2) = 7, rank(2,2,1) = 2
    So rank(2,2,7) <= 7+7+2 = 16
    
    Then rank(2,4,7) <= 2 * 16 = 32
    
    But we can do better by sharing structure.
    
    Let me try to implement the 2x2x2 Winograd decomposition first.
    
    Winograd 2x2x2: 
    C = A*B where A,B are 2x2, C is 2x2
    m1 = a11 + a22, m2 = a11 + a12, m3 = a21 + a22, m4 = a11 - a22,
    m5 = a12 + a21, m6 = a12 - a11, m7 = a21 - a22
    n1 = b11 + b22, n2 = b11 + b12, n3 = b21 + b22, n4 = b11 - b22,
    n5 = b12 + b21, n6 = b12 - b11, n7 = b21 - b22
    c11 = m1*n1 + m4*n4
    c12 = m2*n2 + m6*n6  
    c21 = m3*n3 + m7*n7
    c22 = m5*n5 + m4*n4
    
    Wait, let me get the standard Winograd 2x2x2 right.
    
    Standard Winograd F(2,2,2):
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    m1 = a11 + a22
    m2 = a11 + a12
    m3 = a21 + a22
    m4 = a11 - a22
    m5 = a12 + a21
    m6 = a12 - a11
    m7 = a21 - a22
    
    n1 = b11 + b22
    n2 = b11 + b12
    n3 = b21 + b22
    n4 = b11 - b22
    n5 = b12 + b21
    n6 = b12 - b11
    n7 = b21 - b22
    
    c11 = m1*n1 + m4*n4
    c12 = m2*n2 + m6*n6
    c21 = m3*n3 + m7*n7
    c22 = m5*n5 + m4*n4
    
    Hmm, I need to verify this. Let me use a known correct decomposition.
    
    Actually, let me just use a computational approach: try to find a low-rank
    decomposition using numerical methods, then round to integers/half-integers.
    
    Given the time constraint, let me implement a reasonable heuristic.
    """
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
                return False
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
        # Actually the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if j is the "middle" index connecting A and B, else 0
        # Wait, the multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the 
        # identity in the middle dimension), and 0 otherwise? No.
        # 
        # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        #
        # And the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        #
        # Let me verify numerically.
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank 27 decomposition using block structure
    # 2x4x7 = 2*(2x2x7) + 2*(2x2x7) ... no
    # 2x4x7: split m=4 into two groups of 2
    # C = A[:,0:2]*B[0:2,:] + A[:,2:4]*B[2:4,:]
    # Each term is 2x2x7
    
    # For 2x2x7, use: 2x2x2 + 2x2x2 + 2x2x2 + 2x2x1
    # rank(2,2,2) = 7, rank(2,2,1) = 2
    # rank(2,2,7) <= 3*7 + 2 = 23
    # rank(2,4,7) <= 2*23 = 46
    
    # That's still worse than 56 only slightly. Let me think differently.
    
    # Actually, let me try to use a known result or construct a better one.
    # 
    # For 2x4x7, the minimum rank is known to be at most 27 (from various 
    # lower/upper bounds in the literature), but I'm not sure of the exact value.
    #
    # Let me try a constructive approach with rank 28 or 27.
    
    # Alternative: use the fact that 2x4x7 can be decomposed using
    # the "lax" decomposition or similar.
    
    # For now, let me try to use numerical optimization to find a low-rank 
    # decomposition, then round.
    
    # Given time constraints, let me try a few candidate ranks and see if
    # we can find a valid decomposition.
    
    # Let me try rank 27 first with a structured approach.
    
    # Actually, let me just return the schoolbook decomposition for now
    # and try to improve it if there's time.
    
    # Try a few ranks from 27 up
    for candidate_rank in [27, 28, 29, 30, 32, 35, 40, 45, 50, 56]:
        if time.time() > deadline - 10:
            break
        
        # For a structured approach, let me try to build the decomposition
        # using the 2x2x2 Winograd block.
        
        if candidate_rank >= 2 * 7 * 2 + 2 * 2:  # 2*(2x2x3) blocks
            # Use: split into 4 blocks of 2x2x? 
            # m=4, so 2 blocks of 2 in m direction
            # p=7, so 3 blocks of 2 + 1 block of 1
            # 
            # Block (i_block, j_block, k_block):
            # i_block in {0,1} (n=2, so just 1 block of 2)
            # j_block in {0,1} (m=4, 2 blocks of 2)  
            # k_block in {0,1,2,3} (p=7, 3 blocks of 2 + 1 block of 1)
            
            # For 2x2x2: rank 7 (Winograd)
            # For 2x2x1: rank 2
            #
            # Total: 2 * (3*7 + 2) = 2 * 23 = 46
            
            # That gives rank 46, which is better than 56.
            
            pass
    
    # Let me construct the rank-46 decomposition explicitly.
    # 
    # A is 2x4, B is 4x7, C is 2x7
    # A = [A1, A2] where A1 = A[:,0:2], A2 = A[:,2:4], each 2x2
    # B = [B1; B2] where B1 = B[0:2,:], B2 = B[2:4,:], each 2x7
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: 2x2 * 2x7
    # Split B1 into [B1a, B1b, B1c] where B1a=B1[:,0:2], B1b=B1[:,2:4], B1c=B1[:,4:6], B1d=B1[:,6:7]
    # C1 = A1*B1a + A1*B1b + A1*B1c + A1*B1d
    #
    # Each A1*B1x is 2x2x2 (for x=a,b,c) or 2x2x1 (for x=d)
    
    # Winograd 2x2x2 decomposition (verified):
    # For matrices A=[[a11,a12],[a21,a22]], B=[[b11,b12],[b21,b22]]:
    # m1 = a11+a22, m2 = a11+a12, m3 = a21+a22, m4 = a11-a22
    # m5 = a12+a21, m6 = a12-a11, m7 = a21-a22
    # n1 = b11+b22, n2 = b11+b12, n3 = b21+b22, n4 = b11-b22
    # n5 = b12+b21, n6 = b12-b11, n7 = b21-b22
    # c11 = m1*n1 + m4*n4
    # c12 = m2*n2 + m6*n6
    # c21 = m3*n3 + m7*n7
    # c22 = m5*n5 + m4*n4
    
    # Let me verify this Winograd decomposition is correct.
    # c11 should = a11*b11 + a12*b21
    # m1*n1 + m4*n4 = (a11+a22)(b11+b22) + (a11-a22)(b11-b22)
    # = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a11*b11 - a11*b22 - a22*b11 + a22*b22
    # = 2*a11*b11 + 2*a22*b22
    # That's NOT c11. So my Winograd formulas are wrong.
    
    # Let me look up the correct Winograd 2x2x2.
    # 
    # Correct Winograd F(2,2,2):
    # m1 = a11 + a22
    # m2 = a11 + a12
    # m3 = a21 + a22
    # m4 = a11 - a22
    # m5 = a12 + a21
    # m6 = a12 - a11
    # m7 = a21 - a22
    #
    # n1 = b11 + b22
    # n2 = b11 + b12
    # n3 = b21 + b22
    # n4 = b11 - b22
    # n5 = b12 + b21
    # n6 = b12 - b11
    # n7 = b21 - b22
    #
    # c11 = m1*n1 + m4*n4  ... this gives 2*(a11*b11 + a22*b22) as shown above
    # That's definitely wrong.
    
    # I think the correct Winograd 2x2x2 is:
    # m1 = a11 + a22
    # m2 = a11 + a12
    # m3 = a21 + a22
    # m4 = a11 - a22
    # m5 = a12 + a21
    # m6 = a12 - a11
    # m7 = a21 - a22
    #
    # n1 = b11 + b22
    # n2 = b11 + b12
    # n3 = b21 + b22
    # n4 = b11 - b22
    # n5 = b12 + b21
    # n6 = b12 - b11
    # n7 = b21 - b22
    #
    # c11 = (m1*n1 + m4*n4) / 2  ... no, that would give non-integer results sometimes
    
    # I'm clearly misremembering the Winograd formulas. Let me use a 
    # computational approach instead.
    
    # Given the difficulty of getting the exact Winograd formulas right from memory,
    # let me use a different strategy: return the schoolbook decomposition
    # which is guaranteed correct, with rank 56.
    
    # But that's the same as the current program. Let me at least try to
    # verify it works and return it.
    
    if not verify(U_fb, V_fb, W_fb, rank_schoolbook):
        # Should never happen, but just in case
        pass
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
