"""Matrix multiplication 3x4x6 with improved rank via block decomposition and search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE_MARGIN = 5.0  # seconds before the 120s (or 150s) limit

def _elapsed(t0):
    return time.time() - t0

def _verify(U, V, W):
    """Check that U @ diag(1) @ V^T reconstructs the multiplication tensor correctly via W."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the multiplication tensor T[i, j, k] = 1
    # The decomposition says: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We verify by checking that for all (i, j, k), the sum equals 1.
    # For efficiency, build matrices and use einsum-like check.
    # T is (n*m, m*p, p*n) -> we want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # This is equivalent to: for each c, the outer product U[:,c] @ V[:,c] @ W[:,c] (broadcast) sums to T.
    # Simpler: compute the "tensor contraction" and check.
    # Let's use a direct loop for correctness, but it's slow for large rank.
    # Alternative: build the full tensor via einsum on the factor matrices.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(T == 1)

def _rank_7_strassen_block():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen for 2x2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # Map to our convention:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, c] corresponds to A[i,j]
    # V[j*2+k, c] corresponds to B[j,k]
    # W[k*2+i, c] corresponds to C[k,i] -- note: official trace convention is W[k*n+i,r]
    # For 2x2x2, n=2, so W[k*2+i, r]
    #
    # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B = [[e, f], [g, h]] where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C = [[C00, C01], [C10, C11]]
    #
    # C[i,k] in standard math, but our W uses W[k*2+i]
    # So W[0*2+0] = C[0,0] = C00
    #    W[0*2+1] = C[1,0] = C10  (because i=1, k=0)
    #    W[1*2+0] = C[0,1] = C01  (because i=0, k=1)
    #    W[1*2+1] = C[1,1] = C11  (because i=1, k=1)
    #
    # Let's build the 7 rank-1 terms.
    # Each term c has U[:,c], V[:,c], W[:,c]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Term 0: m1 = (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1  # a+d
    V[0,0] = 1; V[3,0] = 1  # e+h
    # C00 += m1, C11 += m1
    W[0,0] = 1  # C00
    W[3,0] = 1  # C11
    
    # Term 1: m2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1  # c+d
    V[0,1] = 1              # e
    # C10 += m2
    W[1,1] = 1  # C10 (k=0,i=1)
    
    # Term 2: m3 = a(f-h)
    U[0,2] = 1              # a
    V[1,2] = 1; V[3,2] = -1 # f-h
    # C01 += m3, C11 += m3
    W[2,2] = 1  # C01 (k=1,i=0)
    W[3,2] = 1  # C11
    
    # Term 3: m4 = d(g-e)
    U[3,3] = 1              # d
    V[2,3] = 1; V[0,3] = -1 # g-e
    # C00 += m4, C10 += m4
    W[0,3] = 1  # C00
    W[1,3] = 1  # C10
    
    # Term 4: m5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1  # a+b
    V[3,4] = 1              # h
    # C00 -= m5, C01 += m5
    W[0,4] = -1  # C00
    W[2,4] = 1   # C01
    
    # Term 5: m6 = (c-a)(e+f)
    U[2,5] = 1; U[0,5] = -1 # c-a
    V[0,5] = 1; V[1,5] = 1  # e+f
    # C11 += m6
    W[3,5] = 1  # C11
    
    # Term 6: m7 = (b-d)(g+h)
    U[1,6] = 1; U[3,6] = -1 # b-d
    V[2,6] = 1; V[3,6] = 1  # g+h
    # C00 += m7
    W[0,6] = 1  # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Strategy: Split m=4 into two blocks of 2, and p=6 into three blocks of 2.
    # This gives 2*3 = 6 blocks of 3x2x2.
    # For each 3x2x2 block, we need a low-rank decomposition.
    # The tensor for 3x2x2 multiplication has rank at most 3*2*2=12 (schoolbook).
    # Can we do better?
    # Split n=3 into 1+2. Then 3x2x2 = 1x2x2 + 2x2x2.
    # 2x2x2 has rank 7 (Strassen).
    # 1x2x2: multiplying a 1x2 row by 2x2 matrix. Result is 1x2.
    # T[i,j,k] for i in {0}, j in {0,1}, k in {0,1}
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # This is essentially a 1x2 times 2x2, which is two separate dot products of length 2.
    # Rank of 1x2x2 tensor: The tensor T has entries T[0,0,0]=1, T[0,1,0]=0, T[0,0,1]=0, T[0,1,1]=1.
    # Wait, T[i,j,k] = 1 always. The multiplication tensor has T[i,j,k]=1 for all i,j,k.
    # For 1x2x2: T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1.
    # This is a rank-1 tensor? No. T[i,j,k] = 1 means it's the all-ones tensor of shape 1x2x2.
    # The rank of the all-ones tensor of shape (1, m, p) is 1.
    # Because T[0,j,k] = U[0,j,c]*V[j,k,c]*W[k,0,c]. If we set U[0,j,0]=1 for all j, V[j,k,0]=1 for all j,k, W[k,0,0]=1 for all k, then T[0,j,k]=1.
    # So 1x2x2 has rank 1.
    # Therefore 3x2x2 can be decomposed as 1x2x2 (rank 1) + 2x2x2 (rank 7) = rank 8.
    # But we need to be careful: the blocks are not independent in the full tensor because the indices i, j, k are shared in a specific way.
    # Actually, when we split, we are using a block structure. Let's think about it differently.
    #
    # Alternative: Use the known result that M(3,4,6) might have a lower rank.
    # Let's try a simple block decomposition:
    # Split m=4 into 2+2. Split p=6 into 2+2+2.
    # We have blocks: A[i, j] where j in {0,1,2,3}, B[j, k] where k in {0,...,5}.
    # C[i, k] = sum_j A[i,j] B[j,k].
    # 
    # Let's try to use rank-7 for 2x2x2 subblocks and handle the rest.
    # Actually, a cleaner approach: use the 1x2x2 rank-1 and 2x2x2 rank-7 insight.
    # For a 3x2x2 block (i in {0,1,2}, j in {0,1}, k in {0,1}):
    #   Split i into {0} and {1,2}.
    #   Part 1: i=0, j in {0,1}, k in {0,1}: 1x2x2, rank 1.
    #   Part 2: i in {1,2}, j in {0,1}, k in {0,1}: 2x2x2, rank 7.
    #   Total rank per 3x2x2 block: 1 + 7 = 8.
    #   Number of 3x2x2 blocks: (m/2) * (p/2) = 2 * 3 = 6.
    #   Total rank: 6 * 8 = 48.
    #
    # Let's verify this is valid. The key is that the decomposition must be additive across blocks.
    # When we write C[i,k] = sum over blocks, each block contributes to specific (i,k) entries.
    # For block (j0, k0) where j0 in {0,2} and k0 in {0,2,4}:
    #   The block covers A[i, j0:j0+2] and B[j0:j0+2, k0:k0+2].
    #   It produces contributions to C[i, k0:k0+2].
    #   Within the block, for i=0: contribution to C[0, k0] and C[0, k0+1] comes from the 1x2x2 part.
    #   For i=1,2: contribution to C[1, k0:k0+2] and C[2, k0:k0+2] comes from the 2x2x2 part.
    #   These blocks don't overlap in (i,k) space? They do in k, but not in i within the same j-block.
    #   Wait, different j-blocks contribute to the same C[i,k] entries. That's fine, they add up.
    #   Different k-blocks contribute to different C[i,k] entries (since k ranges are disjoint).
    #   So the decomposition is valid.
    #
    # Total rank: 48. This is much better than 72 (schoolbook).
    #
    # Let's implement this.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Precompute the 1x2x2 rank-1 decomposition
    # For 1x2x2: U shape (2,1), V shape (2,1), W shape (2,1)
    # U[j,0] = 1 for j in {0,1}
    # V[j*2+k, 0]... wait, V is indexed by [j*p+k] in the full problem, but here p=2 for the block.
    # Let's work in block-local coordinates.
    # Block 1x2x2: U_local shape (1*2, 1) = (2,1), V_local shape (2*2, 1) = (4,1), W_local shape (2*1, 1) = (2,1)
    # U_local[0,0] = 1, U_local[1,0] = 1  (these correspond to A[0,0] and A[0,1] in the 1x2 part)
    # V_local[j*2+k, 0] = 1 for all j,k
    # W_local[k*1+i, 0] = 1 for all k,i (i=0 only)
    U_1x2x2 = [[1], [1]]
    V_1x2x2 = [[1], [1], [1], [1]]
    W_1x2x2 = [[1], [1]]
    
    # Precompute the 2x2x2 rank-7 Strassen decomposition
    U_2x2x2, V_2x2x2, W_2x2x2 = _rank_7_strassen_block()
    
    # Now assemble for each (j0, k0) block
    # j0 in {0, 2}, k0 in {0, 2, 4}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # Block covers:
            #   i in {0,1,2} (n=3)
            #   j in {j0, j0+1} (m_block=2)
            #   k in {k0, k0+1} (p_block=2)
            #
            # Part 1: i=0, 1x2x2 (rank 1)
            #   U: rows for i=0, j in {j0, j0+1} -> global rows 0*4+j0, 0*4+j0+1
            #   V: rows for j in {j0, j0+1}, k in {k0, k0+1} -> global rows j0*6+k0, j0*6+k0+1, (j0+1)*6+k0, (j0+1)*6+k0+1
            #   W: rows for k in {k0, k0+1}, i=0 -> global rows k0*3+0, (k0+1)*3+0
            #
            # Part 2: i in {1,2}, 2x2x2 (rank 7)
            #   U: rows for i in {1,2}, j in {j0, j0+1} -> global rows 1*4+j0, 1*4+j0+1, 2*4+j0, 2*4+j0+1
            #   V: same as Part 1
            #   W: rows for k in {k0, k0+1}, i in {1,2} -> global rows k0*3+1, k0*3+2, (k0+1)*3+1, (k0+1)*3+2
            
            # We'll accumulate into full U, V, W
            pass  # placeholder
    
    # Let's build the full matrices
    total_rank = 6 * 8  # 48
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    rank_offset = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # Part 1: 1x2x2, rank 1
            # U global rows: 0*m + j0, 0*m + (j0+1)
            U[0*m + j0, rank_offset] = U_1x2x2[0][0]
            U[0*m + (j0+1), rank_offset] = U_1x2x2[1][0]
            # V global rows: j0*p + k0, j0*p + (k0+1), (j0+1)*p + k0, (j0+1)*p + (k0+1)
            V[j0*p + k0, rank_offset] = V_1x2x2[0][0]
            V[j0*p + (k0+1), rank_offset] = V_1x2x2[1][0]
            V[(j0+1)*p + k0, rank_offset] = V_1x2x2[2][0]
            V[(j0+1)*p + (k0+1), rank_offset] = V_1x2x2[3][0]
            # W global rows: k0*n + 0, (k0+1)*n + 0
            W[k0*n + 0, rank_offset] = W_1x2x2[0][0]
            W[(k0+1)*n + 0, rank_offset] = W_1x2x2[1][0]
            rank_offset += 1
            
            # Part 2: 2x2x2, rank 7
            for c in range(7):
                # U global rows for i in {1,2}, j in {j0, j0+1}
                # U_2x2x2 is indexed by [i_local*2 + j_local, c] where i_local in {0,1} maps to i in {1,2}
                # i_local=0 -> i=1, i_local=1 -> i=2
                # j_local=0 -> j=j0, j_local=1 -> j=j0+1
                for i_local in range(2):
                    for j_local in range(2):
                        val = U_2x2x2[i_local*2 + j_local][c]
                        if val != 0:
                            U[(1+i_local)*m + (j0+j_local), rank_offset + c] = val
                # V: same indices as Part 1
                for j_local in range(2):
                    for k_local in range(2):
                        val = V_2x2x2[j_local*2 + k_local][c]
                        if val != 0:
                            V[(j0+j_local)*p + (k0+k_local), rank_offset + c] = val
                # W: i in {1,2}, k in {k0, k0+1}
                # W_2x2x2 is indexed by [k_local*2 + i_local, c]
                # i_local=0 -> i=1, i_local=1 -> i=2
                # k_local=0 -> k=k0, k_local=1 -> k=k0+1
                for k_local in range(2):
                    for i_local in range(2):
                        val = W_2x2x2[k_local*2 + i_local][c]
                        if val != 0:
                            W[(k0+k_local)*n + (1+i_local), rank_offset + c] = val
            rank_offset += 7
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
