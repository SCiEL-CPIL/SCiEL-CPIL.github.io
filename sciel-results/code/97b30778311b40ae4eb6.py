"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # U shape: n*m = 4
    # V shape: m*p = 30
    # W shape: p*n = 30
    
    # Rank is 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The tensor decomposes into 15 independent blocks, one for each k in 0..14
    # Each block k has rank 2.
    # Block k corresponds to the multiplication of column k of B with A, producing column k of C.
    # C[:, k] = A * B[:, k]
    # A is 2x2, B[:, k] is 2x1, C[:, k] is 2x1.
    # The sub-tensor for this k involves:
    # u indices: 0, 2 (corresponding to A[0,0], A[1,0] and A[0,1], A[1,1] ? No)
    # Let's re-verify the index mapping.
    # u = i*m + j.
    # For j=0: u=0 (i=0), u=2 (i=1).
    # For j=1: u=1 (i=0), u=3 (i=1).
    # Wait.
    # In my previous derivation:
    # i=0, j=0 -> u=0
    # i=0, j=1 -> u=1
    # i=1, j=0 -> u=2
    # i=1, j=1 -> u=3
    
    # My previous derivation for T^(k) used:
    # i=0, j=0 -> u=0
    # i=0, j=1 -> u=0 ?? NO.
    # Let's re-read the index mapping in the task.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # U[i*m+j, c]
    # V[j*p+k, c]
    # W[k*n+i, c]
    
    # So:
    # i=0, j=0 -> u=0, v=k, w=2k
    # i=0, j=1 -> u=1, v=15+k, w=2k
    # i=1, j=0 -> u=2, v=k, w=2k+1
    # i=1, j=1 -> u=3, v=15+k, w=2k+1
    
    # My previous grouping was:
    # T[0, k, 2k] = 1
    # T[0, 15+k, 2k] = 1  <-- ERROR. u should be 1, not 0.
    # T[2, k, 2k+1] = 1
    # T[2, 15+k, 2k+1] = 1 <-- ERROR. u should be 3, not 2.
    
    # Let's re-evaluate the rank of T^(k) with correct u indices.
    # T^(k) has non-zeros at:
    # (0, k, 2k)
    # (1, 15+k, 2k)
    # (2, k, 2k+1)
    # (3, 15+k, 2k+1)
    
    # This is a 4x2x2 tensor (u in 0..3, v in {k, 15+k}, w in {2k, 2k+1}).
    # Let's check if it can be decomposed into 2 rank-1 tensors.
    # Try:
    # R1: u=(1,0,0,0), v=(1,1,0...), w=(1,0) -> (0,k,2k) and (0,15+k,2k)? No, u is same.
    # If u is same, v must be same.
    # Term 1: u=e0, v=e_k+e_{15+k}, w=e_{2k}.
    #   Gives (0, k, 2k) and (0, 15+k, 2k).
    #   But we need (1, 15+k, 2k), not (0, 15+k, 2k).
    #   So this doesn't work directly.
    
    # Let's look at the structure again.
    # w=2k:
    #   u=0, v=k
    #   u=1, v=15+k
    # w=2k+1:
    #   u=2, v=k
    #   u=3, v=15+k
    
    # This looks like two separate rank-1 tensors?
    # T_a: u=e0, v=e_k, w=e_{2k}. (0,k,2k)
    # T_b: u=e1, v=e_{15+k}, w=e_{2k}. (1,15+k,2k)
    # T_c: u=e2, v=e_k, w=e_{2k+1}. (2,k,2k+1)
    # T_d: u=e3, v=e_{15+k}, w=e_{2k+1}. (3,15+k,2k+1)
    
    # Can we combine them?
    # T_a + T_b = (e0+e1) \otimes ? No.
    # T_a is rank 1. T_b is rank 1.
    # Sum of 4 rank 1 tensors?
    # Or can we do it in 2?
    # Consider the matrix for w=2k:
    # [[1, 0], [0, 1]] in u={0,1}, v={k, 15+k}.
    # This is the identity matrix. Rank 2.
    # Consider the matrix for w=2k+1:
    # [[1, 0], [0, 1]] in u={2,3}, v={k, 15+k}.
    # This is the identity matrix. Rank 2.
    
    # The tensor is supported on two slices of w.
    # Since the w slices are disjoint, the rank is the sum of the ranks of the slices?
    # No, a rank-1 tensor has a single w index.
    # So any rank-1 term lies entirely in w=2k or w=2k+1.
    # Thus, the tensor decomposes into T_w0 + T_w1.
    # T_w0 is supported on w=2k.
    # T_w0 = e0 \otimes e_k \otimes e_{2k} + e1 \otimes e_{15+k} \otimes e_{2k}
    #       = (e0 \otimes e_k + e1 \otimes e_{15+k}) \otimes e_{2k}
    # The matrix M0 = [[1,0],[0,1]] has rank 2.
    # So T_w0 has rank 2.
    # Similarly T_w1 has rank 2.
    # Total rank for block k is 2 + 2 = 4.
    
    # So total rank is 15 * 4 = 60.
    
    # My previous "rank 30" idea was flawed because I assumed u indices were 0 and 2 for both j=0 and j=1.
    # But u = i*m + j.
    # j=0 -> u=i. (0, 1)
    # j=1 -> u=2+i. (2, 3)
    # So u indices are distinct for j=0 and j=1.
    # Thus, they cannot be combined into a single rank-1 term with a shared u vector.
    
    # So the rank is indeed 60.
    
    # Is there any other trick?
    # Strassen?
    # Strassen reduces 8 to 7 for 2x2x2.
    # Here we have 15 independent 2x2x2-like blocks?
    # No, the blocks are coupled in v and w?
    # No, v and w indices for different k are disjoint.
    # So the tensor is a direct sum of 15 tensors T^(k).
    # Each T^(k) is a 2x2x2 tensor (in terms of i and j)?
    # No, T^(k) involves:
    # i in {0,1}
    # j in {0,1}
    # k is fixed.
    # The tensor for one k is the tensor for multiplying a 2x2 matrix by a 2x1 vector.
    # Let A be 2x2, b be 2x1. c = A*b.
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # The rank of this tensor is the rank of the 2x2x2 tensor for matrix-vector multiplication.
    # This is known to be 4?
    # Let's check.
    # T[i, j, l] where i (output), j (input B), l (input A index)?
    # Indices:
    # u (A index): 4
    # v (B index): 2
    # w (C index): 2
    # Non-zeros:
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # T[0, 0, 0] = 1 (a00, b0, c0)
    # T[1, 0, 0] = 1 (a01, b0, c0) -- Wait, u index for a01 is 1.
    # T[0, 1, 0] = 1 (a00, b1, c0) -- v index for b1 is 1.
    # T[1, 1, 0] = 1 (a01, b1, c0)
    # T[2, 0, 1] = 1 (a10, b0, c1)
    # T[3, 0, 1] = 1 (a11, b0, c1)
    # T[2, 1, 1] = 1 (a10, b1, c1)
    # T[3, 1, 1] = 1 (a11, b1, c1)
    
    # This is the tensor for 2x2 * 2x1.
    # It is the sum of two 2x1x2 tensors?
    # Slice w=0:
    # u={0,1}, v={0,1}. Matrix [[1,1],[1,1]]? No.
    # u=0 (a00): v=0,1 -> 1,1
    # u=1 (a01): v=0,1 -> 1,1
    # Matrix is [[1,1],[1,1]]. Rank 1.
    # Slice w=1:
    # u=2 (a10): v=0,1 -> 1,1
    # u=3 (a11): v=0,1 -> 1,1
    # Matrix is [[1,1],[1,1]]. Rank 1.
    
    # Since w slices are disjoint, rank is 1+1=2?
    # Wait.
    # T_w0 = (e0+e1) \otimes (e0+e1) \otimes e0 ?
    # (e0+e1) \otimes (e0+e1) has entries 1 at (0,0), (0,1), (1,0), (1,1).
    # Yes.
    # So T_w0 is rank 1.
    # T_w1 = (e2+e3) \otimes (e0+e1) \otimes e1.
    # Rank 1.
    # Total rank for T^(k) is 2.
    
    # So total rank is 15 * 2 = 30.
    
    # My previous "rank 60" conclusion was wrong because I assumed the matrices were identity.
    # Let's re-verify the entries.
    # c0 = a00 b0 + a01 b1
    # u for a00 is 0. u for a01 is 1.
    # v for b0 is 0. v for b1 is 1.
    # w for c0 is 0.
    # T[0, 0, 0] = 1
    # T[1, 0, 0] = 1
    # T[0, 1, 0] = 1
    # T[1, 1, 0] = 1
    # This is the all-ones matrix. Rank 1.
    
    # c1 = a10 b0 + a11 b1
    # u for a10 is 2. u for a11 is 3.
    # v for b0 is 0. v for b1 is 1.
    # w for c1 is 1.
    # T[2, 0, 1] = 1
    # T[3, 0, 1] = 1
    # T[2, 1, 1] = 1
    # T[3, 1, 1] = 1
    # This is the all-ones matrix. Rank 1.
    
    # So yes, Rank(T^(k)) = 2.
    # Total Rank = 30.

    # Construction:
    # For each k:
    # Term 1:
    # u = e0 + e1
    # v = e_k + e_{15+k}  (Note: v index for b0 is k, for b1 is 15+k? No.)
    # Let's check v indices.
    # v = j*p + k.
    # j=0 -> v=k.
    # j=1 -> v=15+k.
    # So v vector is e_k + e_{15+k}.
    # w = e_{2k}
    #
    # Term 2:
    # u = e2 + e3
    # v = e_k + e_{15+k}
    # w = e_{2k+1}
    
    # Let's verify Term 1:
    # (e0+e1) \otimes (e_k+e_{15+k}) \otimes e_{2k}
    # Non-zeros at:
    # (0, k, 2k) -> 1
    # (0, 15+k, 2k) -> 1
    # (1, k, 2k) -> 1
    # (1, 15+k, 2k) -> 1
    #
    # Required for c0:
    # a00 (u=0) b0 (v=k) -> 1. OK.
    # a00 (u=0) b1 (v=15+k) -> 1. OK.
    # a01 (u=1) b0 (v=k) -> 1. OK.
    # a01 (u=1) b1 (v=15+k) -> 1. OK.
    
    # Let's verify Term 2:
    # (e2+e3) \otimes (e_k+e_{15+k}) \otimes e_{2k+1}
    # Non-zeros at:
    # (2, k, 2k+1) -> 1
    # (2, 15+k, 2k+1) -> 1
    # (3, k, 2k+1) -> 1
    # (3, 15+k, 2k+1) -> 1
    #
    # Required for c1:
    # a10 (u=2) b0 (v=k) -> 1. OK.
    # a10 (u=2) b1 (v=15+k) -> 1. OK.
    # a11 (u=3) b0 (v=k) -> 1. OK.
    # a11 (u=3) b1 (v=15+k) -> 1. OK.
    
    # This works.
    
    for k in range(p):
        col1 = 2 * k
        col2 = 2 * k + 1
        
        # Term 1
        U[0][col1] = 1
        U[1][col1] = 1
        V[k][col1] = 1
        V[m * p // 2 + k][col1] = 1  # 15+k
        W[2 * k][col1] = 1
        
        # Term 2
        U[2][col2] = 1
        U[3][col2] = 1
        V[k][col2] = 1
        V[m * p // 2 + k][col2] = 1  # 15+k
        W[2 * k + 1][col2] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
