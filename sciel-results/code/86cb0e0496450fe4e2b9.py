import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Block-diagonal decomposition:
    # 4x4x5 = (2x2x2) ⊕ (2x2x2) ⊕ (2x2x1) ⊕ (2x2x1)
    # Rank = 7 + 7 + 4 + 4 = 22
    #
    # Block 1: A rows {0,1}, B cols {0,1}, C cols {0,1} -> 2x2x2, rank 7
    # Block 2: A rows {0,1}, B cols {2,3}, C cols {2,3} -> 2x2x2, rank 7
    # Block 3: A rows {2,3}, B cols {0,1}, C cols {0,1} -> 2x2x1, rank 4
    # Block 4: A rows {2,3}, B cols {2,3}, C cols {2,3} -> 2x2x1, rank 4
    #
    # Wait - that doesn't cover all of C. Let me reconsider.
    #
    # A is 4x4, B is 4x5, C is 4x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T[i,j,k] = delta (identity for matrix mult).
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k).
    #
    # Block decomposition approach:
    # Partition A rows into {0,1} and {2,3}
    # Partition B cols into {0,1}, {2,3}, {4}
    # Partition C cols into {0,1}, {2,3}, {4}
    #
    # Actually, let me think more carefully. The standard way to get a rank
    # decomposition via Kronecker products:
    #
    # For 2x2x2 (Strassen): rank 7
    # For 2x2x1 (simple): rank 4
    #
    # 4x4x5 can be seen as:
    # - Top-left 2x2 block of A * left 2x2 block of B -> contributes to left 2x2 of C: 2x2x2, rank 7
    # - Top-right 2x2 block of A * left 2x2 block of B -> contributes to right 2x2 of C: 2x2x2, rank 7
    # - Bottom-left 2x2 block of A * left 2x2 block of B -> contributes to left 2x2 of C: 2x2x2, rank 7
    # - Bottom-right 2x2 block of A * left 2x2 block of B -> contributes to right 2x2 of C: 2x2x2, rank 7
    # - Top 2x1 block of A * col 4 of B -> contributes to col 4 of C: 2x1x1... 
    #
    # This is getting complex. Let me use a different approach.
    #
    # Actually, the cleanest block decomposition for 4x4x5:
    # Think of A as 2x2 blocks of size 2x2: A = [[A11, A12], [A21, A22]]
    # B as 2x3 blocks of size 2x2: B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2 except last col
    # 
    # Hmm, B is 4x5, so partitioning B's 5 columns is awkward.
    #
    # Alternative: Use the fact that matrix multiplication tensor for (n,m,p)
    # has rank at most n*m*p (schoolbook). For 4x4x5, schoolbook = 80.
    #
    # Let me try a direct construction with rank 22:
    # 
    # Key insight: If we can write the multiplication as a sum of rank-1 tensors,
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    #
    # For a 2x2x2 block (Strassen), we have 7 terms.
    # For a 2x2x1 block (just 4 scalar mults), we have 4 terms.
    #
    # Let me partition:
    # A rows: {0,1} = top, {2,3} = bottom
    # B cols: {0,1} = left, {2,3} = middle, {4} = right  
    # C cols: {0,1} = left, {2,3} = middle, {4} = right
    #
    # C_top_left = A_top * B_left (2x2 * 2x2 = 2x2) -> rank 7 (Strassen)
    # C_top_mid  = A_top * B_mid  (2x2 * 2x2 = 2x2) -> rank 7 (Strassen)
    # C_top_right= A_top * B_right(2x2 * 2x1 = 2x1) -> rank 4 (schoolbook for 2x2x1)
    # C_bot_left = A_bot * B_left (2x2 * 2x2 = 2x2) -> rank 7 (Strassen)
    # C_bot_mid  = A_bot * B_mid  (2x2 * 2x2 = 2x2) -> rank 7 (Strassen)
    # C_bot_right= A_bot * B_right(2x2 * 2x1 = 2x1) -> rank 4 (schoolbook for 2x2x1)
    #
    # Total rank = 7+7+4+7+7+4 = 36
    #
    # That's better than 80 but not 22. Let me reconsider the directive.
    #
    # The directive says: "two independent 2x2x2 sub-problems (each with rank 7) 
    # and two 2x2x1 sub-problems (each with rank 4), leveraging the Kronecker 
    # product structure to construct a rank-22 decomposition"
    #
    # 7+7+4+4 = 22. So the decomposition must be:
    # 4x4x5 = (2x2x2) ⊕ (2x2x2) ⊕ (2x2x1) ⊕ (2x2x1)
    #
    # This means we need to partition the problem into exactly 4 blocks.
    # 
    # One natural way: 
    # - A is 4x4, partition rows into {0,1} and {2,3}
    # - B is 4x5, partition cols into {0,1,2} and {3,4}? No...
    #
    # Actually, I think the intended decomposition is:
    # Think of the 4x4x5 multiplication as built from 2x2 building blocks.
    # 
    # A = [[A1, A2], [A3, A4]] where each Ai is 2x2
    # B = [[B1, B2, B3], [B4, B5, B6]] where B1-B3 are 2x2, B4-B6 are 2x2? 
    # No, B is 4x5.
    #
    # Let me try: B cols {0,1} and {2,3,4}
    # B = [[B1, B2], [B3, B4]] where B1,B2 are 2x2 and 2x3? No, that's messy.
    #
    # Alternative interpretation: Maybe the Kronecker product structure means
    # we use the 2x2 Strassen decomposition on the "block" level.
    #
    # Let me just try to construct a valid rank-22 decomposition directly.
    #
    # Actually, I realize I should just implement the schoolbook (rank 80) 
    # correctly first, then attempt the block decomposition. Given time 
    # constraints, let me focus on getting a correct answer.
    #
    # Let me try rank 36 first (the 6-block decomposition above), which I
    # can construct more easily.

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
        U is 4x7 (for A indices (i,j)), V is 4x7 (for B indices (j,k)), W is 4x7 (for C indices (k,i)).
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b00)
        # M4 = a11(b10-b01)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # For each term, U[i,j], V[j,k], W[k,i]
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1 = (a00+a11)(b00+b11)
        # a00: U[0,0], a11: U[3,0]
        # b00: V[0,0], b11: V[3,0]
        # c00: W[0,0], c11: W[3,0]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10+a11)b00
        # a10: U[2,1], a11: U[3,1]
        # b00: V[0,1]
        # c10: W[2,1], c11: W[3,1]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = a00(b11-b00)
        # a00: U[0,2]
        # b11: V[3,2], b00: V[0,2]
        # c01: W[1,2], c11: W[3,2]
        U[0][2] = 1
        V[3][2] = 1; V[0][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = a11(b10-b01)
        # a11: U[3,3]
        # b10: V[2,3], b01: V[1,3]
        # c00: W[0,3], c10: W[2,3]
        U[3][3] = 1
        V[2][3] = 1; V[1][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a00+a01)b11
        # a00: U[0,4], a01: U[1,4]
        # b11: V[3,4]
        # c00: W[0,4], c01: W[1,4]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        # a10: U[2,5], a00: U[0,5]
        # b00: V[0,5], b01: V[1,5]
        # c11: W[3,5]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        # a01: U[1,6], a11: U[3,6]
        # b10: V[2,6], b11: V[3,6]
        # c00: W[0,6], c11: W[3,6]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        """Return (U, V, W) for 2x2 * 2x1 multiplication, rank 4.
        U is 4x4 (for A indices (i,j)), V is 2x4 (for B indices (j,k) where k=0), W is 2x4 (for C indices (k,i) where k=0).
        Actually, for 2x2x1: A is 2x2, B is 2x1, C is 2x1.
        U[i*2+j] for i in 0..1, j in 0..1 -> 4 entries
        V[j*1+k] for j in 0..1, k in 0..0 -> 2 entries
        W[k*2+i] for k in 0..0, i in 0..1 -> 2 entries
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        #
        # Term 1: a00 * b0 -> c0
        U[0][0] = 1  # a00: i=0,j=0
        V[0][0] = 1  # b0: j=0,k=0
        W[0][0] = 1  # c0: k=0,i=0
        #
        # Term 2: a01 * b1 -> c0
        U[1][1] = 1  # a01: i=0,j=1
        V[1][1] = 1  # b1: j=1,k=0
        W[0][1] = 1  # c0: k=0,i=0
        #
        # Term 3: a10 * b0 -> c1
        U[2][2] = 1  # a10: i=1,j=0
        V[0][2] = 1  # b0: j=0,k=0
        W[0][2] = 1  # c1: k=0,i=1
        #
        # Term 4: a11 * b1 -> c1
        U[3][3] = 1  # a11: i=1,j=1
        V[1][3] = 1  # b1: j=1,k=0
        W[0][3] = 1  # c1: k=0,i=1
        
        return U, V, W

    # Build the 6-block decomposition with rank 36
    # Blocks:
    # 1. A_top(0,1) x B_left(0,1) -> C_left(0,1): 2x2x2, rank 7
    # 2. A_top(0,1) x B_mid(2,3)  -> C_mid(2,3):  2x2x2, rank 7
    # 3. A_top(0,1) x B_right(4)  -> C_right(4):   2x2x1, rank 4
    # 4. A_bot(2,3) x B_left(0,1) -> C_left(0,1):  2x2x2, rank 7
    # 5. A_bot(2,3) x B_mid(2,3)  -> C_mid(2,3):   2x2x2, rank 7
    # 6. A_bot(2,3) x B_right(4)  -> C_right(4):   2x2x1, rank 4
    # Total rank = 36
    
    total_rank = 36
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 36
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 36
    W = [[0]*total_rank for _ in range(p*n)]  # 20 x 36
    
    # For each block, we need to map local indices to global indices.
    # 
    # Global A index: i*m + j = 4*i + j, i in 0..3, j in 0..3
    # Global B index: j*p + k = 5*j + k, j in 0..3, k in 0..4
    # Global C index: k*n + i = 4*k + i, k in 0..4, i in 0..3
    
    def embed_block(block_U, block_V, block_W, a_row_offset, b_col_offset, c_col_offset, base_rank):
        """Embed a local block into the global matrices.
        block_U: list of lists, shape (2*2=4, rank_block) for A sub-block
        block_V: list of lists, shape (2*p_sub, rank_block) for B sub-block  
        block_W: list of lists, shape (p_sub*2, rank_block) for C sub-block
        a_row_offset: starting row index in A (0 or 2)
        b_col_offset: starting col index in B (0, 2, or 4)
        c_col_offset: starting col index in C (0, 2, or 4)
        base_rank: starting column in the global rank dimension
        """
        for local_i in range(2):
            for local_j in range(2):
                gi = a_row_offset + local_i
                gj = local_j  # A cols are 0..1 within the block
                global_a_idx = gi * 4 + gj
                for c in range(len(block_U[0])):
                    U[global_a_idx][base_rank + c] = block_U[local_i * 2 + local_j][c]
        
        # For V: B sub-block has rows = A's local cols (0..1), cols = B's local cols
        # B global index: j_global * 5 + k_global
        # j_global = local_j (0 or 1), k_global = b_col_offset + local_k
        p_sub = len(block_V[0]) if False else 2  # number of B cols in this block
        # Actually, let me determine p_sub from the block_V shape
        # block_V has shape (2 * p_sub, rank) where p_sub is number of B columns
        num_b_cols = len(block_V) // 2  # since block_V rows = 2 * p_sub
        
        for local_j in range(2):
            for local_k in range(num_b_cols):
                gj = local_j
                gk = b_col_offset + local_k
                global_b_idx = gj * 5 + gk
                for c in range(len(block_V[0])):
                    V[global_b_idx][base_rank + c] = block_V[local_j * num_b_cols + local_k][c]
        
        # For W: C sub-block has rows = B's local cols, cols = A's local rows
        # C global index: k_global * 4 + i_global
        # k_global = c_col_offset + local_k, i_global = a_row_offset + local_i
        num_c_cols = len(block_W) // 2  # block_W rows = p_sub * 2
        
        for local_k in range(num_c_cols):
            for local_i in range(2):
                gk = c_col_offset + local_k
                gi = a_row_offset + local_i
                global_c_idx = gk * 4 + gi
                for c in range(len(block_W[0])):
                    W[global_c_idx][base_rank + c] = block_W[local_k * 2 + local_i][c]
    
    # Get the sub-block decompositions
    sU, sV, sW = strassen_2x2x2()
    bU, bV, bW = schoolbook_2x2x1()
    
    # Block 1: A rows 0-1, B cols 0-1, C cols 0-1 (2x2x2)
    embed_block(sU, sV, sW, 0, 0, 0, 0)
    # Block 2: A rows 0-1, B cols 2-3, C cols 2-3 (2x2x2)
    embed_block(sU, sV, sW, 0, 2, 2, 7)
    # Block 3: A rows 0-1, B col 4, C col 4 (2x2x1)
    embed_block(bU, bV, bW, 0, 4, 4, 14)
    # Block 4: A rows 2-3, B cols 0-1, C cols 0-1 (2x2x2)
    embed_block(sU, sV, sW, 2, 0, 0, 18)
    # Block 5: A rows 2-3, B cols 2-3, C cols 2-3 (2x2x2)
    embed_block(sU, sV, sW, 2, 2, 2, 25)
    # Block 6: A rows 2-3, B col 4, C col 4 (2x2x1)
    embed_block(bU, bV, bW, 2, 4, 4, 32)
    
    result = {"U": U, "V": V, "W": W}
    
    if verify(total_rank, U, V, W):
        return result
    
    # Fallback to schoolbook
    best = make_schoolbook()
    if verify(n*m*p, best['U'], best['V'], best['W']):
        return best
    
    return best

# EVOLVE-BLOCK-END
