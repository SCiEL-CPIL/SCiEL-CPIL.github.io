"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x2 matrix by a 2x16 matrix.
    
    The task is to decompose the matrix multiplication tensor T_{i,j,k} for:
    - A is 2x2 (indices i in {0,1}, j in {0,1})
    - B is 2x16 (indices j in {0,1}, k in {0,...,15})
    - C is 2x16 (indices i in {0,1}, k in {0,...,15})
    
    The output factor W follows the trace convention W[k*n+i, r] where n=2.
    
    We use a block-circulant approach: treat the 16-column dimension as 8 independent
    2x2 matrix multiplications (since 16 = 8*2). Each 2x2x2 block can be decomposed
    with rank 7 (Strassen's algorithm). Since the A matrix (2x2) is shared across all
    8 blocks, we can share the U factors, giving total rank = 8 * 7 = 56.
    
    However, we can do better. Let's use the fact that the 2x2x2 tensor has rank 7,
    and we have 8 such blocks. If we can share U vectors across blocks, we get rank 56.
    
    Actually, let me reconsider. The tensor is:
    T_{(i,j), (j,k)} = C_{(i,k)}
    
    For a fixed k pair (2g, 2g+1), we have a 2x2x2 sub-problem where:
    - A is 2x2 (i, j)
    - B is 2x2 (j, k') where k' in {0,1} corresponds to the two columns of B
    - C is 2x2 (i, k')
    
    So we have 8 independent 2x2x2 problems, all sharing the same A matrix structure.
    
    Using Strassen's algorithm for each 2x2x2 block, each block needs rank 7.
    Since the A factors (U vectors) can be shared across all 8 blocks (they depend only
    on i,j which are the same for all blocks), the total rank is 8 * 7 = 56.
    
    Wait, that's not quite right. Let me think more carefully.
    
    In the tensor decomposition, we have:
    T_{(i,j,k)} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    
    For each of the 8 blocks (g = 0,...,7), the k indices are (2g, 2g+1).
    The U vectors depend only on (i,j), so they can be the same for all blocks.
    The V vectors depend on (j,k), so they differ per block.
    The W vectors depend on (k,i), so they differ per block.
    
    So for each block g, we need 7 (U, V_g, W_g) triples. Since U is shared,
    the total number of rank-1 terms is 8 * 7 = 56.
    
    Let me implement Strassen's algorithm for 2x2x2 and use it 8 times.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen's algorithm for 2x2x2 matrix multiplication
    # The 7 products and their combinations:
    #
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    #
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22) b11
    # M3 = a11 (b12 - b22)
    # M4 = a22 (b21 - b11)
    # M5 = (a11 + a12) b22
    # M6 = (a21 - a11) (b11 + b12)
    # M7 = (a12 - a22) (b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U vectors (for A):
    # U1: a11 + a22 -> [1, 0, 0, 1] in (i,j) order: (0,0)=1, (1,1)=1
    # U2: a21 + a22 -> [0, 0, 1, 1]
    # U3: a11 -> [1, 0, 0, 0]
    # U4: a22 -> [0, 0, 0, 1]
    # U5: a11 + a12 -> [1, 1, 0, 0]
    # U6: a21 - a11 -> [-1, 0, 1, 0]
    # U7: a12 - a22 -> [0, 1, 0, -1]
    
    U_strassen = [
        [1, 0, 0, 1],   # U1: (0,0)=1, (1,1)=1
        [0, 0, 1, 1],   # U2: (1,0)=1, (1,1)=1
        [1, 0, 0, 0],   # U3: (0,0)=1
        [0, 0, 0, 1],   # U4: (1,1)=1
        [1, 1, 0, 0],   # U5: (0,0)=1, (0,1)=1
        [-1, 0, 1, 0],  # U6: (0,0)=-1, (1,0)=1
        [0, 1, 0, -1],  # U7: (0,1)=1, (1,1)=-1
    ]
    
    # V vectors (for B):
    # V1: b11 + b22 -> [1, 0, 0, 1] in (j,k) order: (0,0)=1, (1,1)=1
    # V2: b11 -> [1, 0, 0, 0]
    # V3: b12 - b22 -> [0, 1, 0, -1]
    # V4: b21 - b11 -> [-1, 0, 1, 0]
    # V5: b22 -> [0, 0, 0, 1]
    # V6: b11 + b12 -> [1, 1, 0, 0]
    # V7: b21 + b22 -> [0, 0, 1, 1]
    
    V_strassen = [
        [1, 0, 0, 1],   # V1: (0,0)=1, (1,1)=1
        [1, 0, 0, 0],   # V2: (0,0)=1
        [0, 1, 0, -1],  # V3: (0,1)=1, (1,1)=-1
        [-1, 0, 1, 0],  # V4: (0,0)=-1, (1,0)=1
        [0, 0, 0, 1],   # V5: (1,1)=1
        [1, 1, 0, 0],   # V6: (0,0)=1, (0,1)=1
        [0, 0, 1, 1],   # V7: (1,0)=1, (1,1)=1
    ]
    
    # W vectors (for C):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # W1: C11 += 1, C22 += 1 -> (0,0)=1, (1,1)=1
    # W2: C21 += 1, C22 -= 1 -> (1,0)=1, (1,1)=-1
    # W3: C12 += 1, C22 += 1 -> (0,1)=1, (1,1)=1
    # W4: C11 += 1, C21 += 1 -> (0,0)=1, (1,0)=1
    # W5: C11 -= 1, C12 += 1 -> (0,0)=-1, (0,1)=1
    # W6: C22 += 1 -> (1,1)=1
    # W7: C11 += 1 -> (0,0)=1
    
    W_strassen = [
        [1, 0, 0, 1],   # W1: (0,0)=1, (1,1)=1
        [0, 0, 1, -1],  # W2: (1,0)=1, (1,1)=-1
        [0, 1, 0, 1],   # W3: (0,1)=1, (1,1)=1
        [1, 0, 1, 0],   # W4: (0,0)=1, (1,0)=1
        [-1, 1, 0, 0],  # W5: (0,0)=-1, (0,1)=1
        [0, 0, 0, 1],   # W6: (1,1)=1
        [1, 0, 0, 0],   # W7: (0,0)=1
    ]
    
    # Verify Strassen decomposition for 2x2x2
    # T_{(i,j),(j,k)} = sum_s U_s[i*m+j] * V_s[j*p2+k] * W_s[k*n+i]
    # where n=2, m=2, p2=2
    
    def verify_2x2x2(U_s, V_s, W_s, n=2, m=2, p2=2):
        """Verify that the Strassen decomposition correctly computes 2x2x2 matmul."""
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    # Expected: T_{(i,j),(j,k)} = 1 (since C_{ik} = sum_j A_{ij} B_{jk},
                    # the tensor entries are 1 when the indices match the matmul pattern)
                    # Actually, the tensor T_{(i,j,k)} = 1 for all valid (i,j,k) in matmul
                    # The decomposition should satisfy:
                    # sum_s U_s[i*m+j] * V_s[j*p2+k] * W_s[k*n+i] = 1 for all i,j,k
                    # Wait, that's not right. Let me reconsider.
                    #
                    # The matrix multiplication tensor is:
                    # T_{(i,j),(j,k)} = delta_{proper} -- no, it's just the identity in the
                    # sense that C_{ik} = sum_j A_{ij} B_{jk}
                    #
                    # In tensor form, the multiplication tensor has entries:
                    # T_{(i,j,k)} = 1 for all i, j, k (it's the "multiplication" tensor)
                    # and the decomposition is:
                    # T_{(i,j,k)} = sum_s U_s[(i,j)] * V_s[(j,k)] * W_s[(k,i)]
                    #
                    # So for each (i,j,k), we need:
                    # sum_s U_s[i*m+j] * V_s[j*p2+k] * W_s[k*n+i] = 1
                    
                    total = 0
                    for s in range(len(U_s)):
                        total += U_s[s][i * m + j] * V_s[s][j * p2 + k] * W_s[s][k * n + i]
                    if total != 1:
                        return False
        return True
    
    # Let's verify
    # is_valid = verify_2x2x2(U_strassen, V_strassen, W_strassen)
    # if not is_valid:
    #     # Fallback to schoolbook
    #     pass
    
    # Now construct the full 2x2x16 decomposition
    # We have 8 blocks (g = 0, ..., 7), each handling columns 2g and 2g+1 of B and C
    # For each block, we use the same U_strassen vectors but scaled V and W vectors
    
    # U matrix: shape (n*m, rank) = (4, 56)
    # Each U vector is reused for all 8 blocks
    rank_per_block = 7
    num_blocks = p // 2  # 8 blocks
    total_rank = rank_per_block * num_blocks  # 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for g in range(num_blocks):
        for s in range(rank_per_block):
            col = g * rank_per_block + s
            
            # U is shared across all blocks
            for idx in range(n * m):
                U[idx][col] = U_strassen[s][idx]
            
            # V: for block g, k' in {0,1} maps to k = 2g + k'
            for j in range(m):
                for k_prime in range(2):
                    k = 2 * g + k_prime
                    V[j * p + k][col] = V_strassen[s][j * 2 + k_prime]
            
            # W: for block g, k' in {0,1} maps to k = 2g + k'
            # W index: k * n + i
            for i in range(n):
                for k_prime in range(2):
                    k = 2 * g + k_prime
                    W[k * n + i][col] = W_strassen[s][k_prime * 2 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
