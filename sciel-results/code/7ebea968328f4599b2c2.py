import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    DEADLINE = 120

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def schoolbook_block(n_b, m_b, p_b):
        r = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, r))
        V = np.zeros((m_b * p_b, r))
        W = np.zeros((p_b * n_b, r))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, col] = 1
                    V[j * p_b + k, col] = 1
                    W[k * n_b + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A10+A11)(B00+B01) -> C[0,1]
        U[2, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[1, 0] += 1
        W[2, 0] += 1
        # M2 = (A10+A11)B10 -> C[1,0]-C[0,0]
        U[2, 1] += 1; U[3, 1] += 1
        V[2, 1] += 1
        W[1, 1] += 1; W[0, 1] -= 1
        # M3 = A00(B01-B11) -> C[1,0]+C[1,1]
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1; W[3, 2] += 1
        # M4 = A11*B10 -> C[0,0]+C[1,0]
        U[3, 3] += 1
        V[2, 3] += 1
        W[0, 3] += 1; W[1, 3] += 1
        # M5 = (A00+A11)(B00+B11) -> C[0,0]+C[1,1]
        U[0, 4] += 1; U[3, 4] += 1
        V[0, 4] += 1; V[3, 4] += 1
        W[0, 4] += 1; W[3, 4] += 1
        # M6 = (A01-A00)(B00+B10) -> C[1,1]
        U[1, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[2, 5] += 1
        W[3, 5] += 1
        # M7 = (A10-A01)(B11+B01) -> C[0,1]-C[0,0]
        U[2, 6] += 1; U[1, 6] -= 1
        V[3, 6] += 1; V[1, 6] += 1
        W[2, 6] += 1; W[0, 6] -= 1
        return U, V, W

    def place_block(global_U, global_V, global_W, sub_U, sub_V, sub_W,
                    n_sub, m_sub, p_sub, i_off, j_off, k_off, col_off):
        rank_sub = sub_U.shape[1]
        for r in range(rank_sub):
            for i in range(n_sub):
                for j in range(m_sub):
                    gi = i_off + i
                    gj = j_off + j
                    global_U[gi * m + gj, col_off + r] = sub_U[i * m_sub + j, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    gj = j_off + j
                    gk = k_off + k
                    global_V[gj * p + gk, col_off + r] = sub_V[j * p_sub + k, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    gk = k_off + k
                    gi = i_off + i
                    global_W[gk * n + gi, col_off + r] = sub_W[k * n_sub + i, r]

    # Strategy: Split m=5 into m1=2, m2=3
    # Block A: n=3, m=2, p=7 -> split p into 2+2+3
    #   A1: 3x2x2 (Strassen, rank 7)
    #   A2: 3x2x2 (Strassen, rank 7)
    #   A3: 3x2x3 (schoolbook, rank 18)
    # Block B: n=3, m=3, p=7 (schoolbook, rank 63)
    # Total: 7+7+18+63 = 95

    total_rank = 7 + 7 + 18 + 63
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # Block A1: 3x2x2, i_off=0, j_off=0, k_off=0
    sU, sV, sW = strassen_222()
    # For 3x2x2: we need to extend the 2x2 Strassen to 3 rows
    # The 3x2x2 tensor: C[i,k] = sum_j A[i,j] B[j,k] for i=0,1,2
    # This is NOT the same as 2x2x2! We need a 3x2x2 decomposition.
    # Actually, 3x2x2 is three independent 1x2x2 multiplications sharing B.
    # But in tensor form, we can't easily share. Let me use schoolbook for 3x2x2.
    # Schoolbook 3x2x2 has rank 12.
    
    # Let me reconsider. For 3x2x2:
    # U has 3*2=6 rows, V has 2*2=4 rows, W has 2*3=6 rows
    # Schoolbook: rank 12
    # Can we do better? The tensor is the "3x2x2 matrix multiplication tensor"
    # Its rank is actually 12 (no known improvement for such small dimensions)
    
    # So let me just use schoolbook for all 3x2xp blocks and focus on getting it right.
    # Actually wait - let me think about this differently.
    
    # For the 3x2x2 block, I can view it as: for each of the 3 rows of A,
    # compute a 1x2 * 2x2 multiplication. Each has rank 4. Total 12.
    # There's no sharing possible in the tensor decomposition framework.
    
    # So my best bet is:
    # Block A: 3x2x7 schoolbook = rank 42
    # Block B: 3x3x7 schoolbook = rank 63
    # Total: 105 (same as full schoolbook)
    
    # I need to find actual improvements. Let me think about what blocks can use Strassen.
    # Strassen works for 2x2x2. I need a 2x2x2 sub-block.
    # That requires n_sub=2, m_sub=2, p_sub=2.
    # 
    # Can I split n=3 into 1+2? Then:
    # Block A: n=1, m=5, p=7 -> schoolbook 35
    # Block B: n=2, m=5, p=7 -> split m=5 into 2+3, p=7 into 2+5
    #   B1: 2x2x2 (Strassen, rank 7)
    #   B2: 2x2x5 (schoolbook, rank 20)
    #   B3: 2x3x7 (schoolbook, rank 42)
    #   Total B: 69
    # Grand total: 35 + 69 = 104
    # 
    # Or split p=7 into 2+2+3:
    #   B1: 2x2x2 (Strassen, rank 7)
    #   B2: 2x2x2 (Strassen, rank 7)
    #   B3: 2x2x3 (schoolbook, rank 12)
    #   B4: 2x3x2 (schoolbook, rank 12)
    #   B5: 2x3x2 (schoolbook, rank 12)
    #   B6: 2x3x3 (schoolbook, rank 18)
    #   Total B: 68
    # Grand total: 35 + 68 = 103
    #
    # Better! Let me try more splits.
    #
    # Split n=3 into 1+2, m=5 into 2+3, p=7 into 2+2+3:
    # Block (1,2,2): 1x2x2 schoolbook = 4
    # Block (1,2,2): 1x2x2 schoolbook = 4
    # Block (1,2,3): 1x2x3 schoolbook = 6
    # Block (1,3,2): 1x3x2 schoolbook = 6
    # Block (1,3,2): 1x3x2 schoolbook = 6
    # Block (1,3,3): 1x3x3 schoolbook = 9
    # Block (2,2,2): 2x2x2 Strassen = 7
    # Block (2,2,2): 2x2x2 Strassen = 7
    # Block (2,2,3): 2x2x3 schoolbook = 12
    # Block (2,3,2): 2x3x2 schoolbook = 12
    # Block (2,3,2): 2x3x2 schoolbook = 12
    # Block (2,3,3): 2x3x3 schoolbook = 18
    # Total: 4+4+6+6+6+9+7+7+12+12+12+18 = 103
    
    # Let me try n=3 into 1+2, m=5 into 2+3, p=7 into 3+4:
    # (1,2,3): 6, (1,2,4): 8, (1,3,3): 9, (1,3,4): 12
    # (2,2,3): 12, (2,2,4): 16, (2,3,3): 18, (2,3,4): 24
    # Total: 6+8+9+12+12+16+18+24 = 105 (worse)
    
    # The best so far is 103. Let me verify this is correct by implementing it.
    
    # Actually, let me try yet another split: n=3 into 2+1, m=5 into 2+3, p=7 into 2+2+3
    # Same as above by symmetry: 103
    
    # Let me try to be smarter. What about using Strassen more aggressively?
    # If I split n=3 into 1+2, and for the 2-row block, split m=5 into 2+2+1:
    # 2x2x2: Strassen 7
    # 2x2x2: Strassen 7  
    # 2x2x3: schoolbook 12
    # 2x1x2: schoolbook 4
    # 2x1x2: schoolbook 4
    # 2x1x3: schoolbook 6
    # For the 1-row block:
    # 1x2x2: 4, 1x2x2: 4, 1x2x3: 6
    # 1x1x2: 2, 1x1x2: 2, 1x1x3: 3
    # Total: 7+7+12+4+4+6+4+4+6+2+2+3 = 61... wait that's for p=7 split as 2+2+3
    # Let me recount:
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=3: 12
    # n=2, m=1, p=2: 4
    # n=2, m=1, p=2: 4
    # n=2, m=1, p=3: 6
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=3: 6
    # n=1, m=1, p=2: 2
    # n=1, m=1, p=2: 2
    # n=1, m=1, p=3: 3
    # Sum: 7+7+12+4+4+6+4+4+6+2+2+3 = 61
    
    # Wait, that can't be right. The schoolbook for 3x5x7 is 105.
    # Let me recheck: for n=2,m=1,p=2: rank = 2*1*2 = 4. Yes.
    # For n=1,m=1,p=2: rank = 1*1*2 = 2. Yes.
    # Total should be sum of all block ranks.
    # 7+7+12+4+4+6+4+4+6+2+2+3 = 61? That seems too good.
    
    # Oh wait, I'm double counting or making an error. Let me be more careful.
    # n splits: [2, 1]
    # m splits: [2, 2, 1]
    # p splits: [2, 2, 3]
    # 
    # For each combination (n_i, m_j, p_k):
    # (2,2,2): 7 (Strassen)
    # (2,2,2): 7 (Strassen)
    # (2,2,3): 12 (schoolbook)
    # (2,1,2): 4 (schoolbook)
    # (2,1,2): 4 (schoolbook)
    # (2,1,3): 6 (schoolbook)
    # (1,2,2): 4 (schoolbook)
    # (1,2,2): 4 (schoolbook)
    # (1,2,3): 6 (schoolbook)
    # (1,1,2): 2 (schoolbook)
    # (1,1,2): 2 (schoolbook)
    # (1,1,3): 3 (schoolbook)
    # Total: 7+7+12+4+4+6+4+4+6+2+2+3 = 61
    
    # Hmm, 61 seems suspiciously low. Let me verify with a simpler case.
    # If all blocks use schoolbook: sum(n_i * m_j * p_k) = (sum n_i)(sum m_j)(sum p_k) = 3*5*7 = 105.
    # With Strassen: we save 1 per 2x2x2 block (8->7). We have 4 such blocks.
    # So 105 - 4 = 101? But I got 61. Something is wrong.
    
    # OH WAIT. I see my error. When I split m into [2,2,1], I have 3 m-blocks.
    # When I split p into [2,2,3], I have 3 p-blocks.
    # So for n=2: 3*3 = 9 blocks, not 6.
    # For n=1: 3*3 = 9 blocks, not 6.
    # Total: 18 blocks.
    
    # Let me redo:
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=3: 12
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=2: 7
    # n=2, m=2, p=3: 12
    # n=2, m=1, p=2: 4
    # n=2, m=1, p=2: 4
    # n=2, m=1, p=3: 6
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=3: 6
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=2: 4
    # n=1, m=2, p=3: 6
    # n=1, m=1, p=2: 2
    # n=1, m=1, p=2: 2
    # n=1, m=1, p=3: 3
    # Total: 7+7+12+7+7+12+4+4+6+4+4+6+4+4+6+2+2+3 = 101
    
    # OK so 101 with 4 Strassen blocks. That's an improvement from 105!
    
    # Let me implement this properly.
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 3]
    
    total_rank = 0
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    print(f"Total rank: {total_rank}")
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    i_off = 0
    for ni in n_splits:
        j_off = 0
        for mj in m_splits:
            k_off = 0
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    sU, sV, sW = strassen_222()
                else:
                    sU, sV, sW = schoolbook_block(ni, mj, pk)
                place_block(U, V, W, sU, sV, sW, ni, mj, pk, i_off, j_off, k_off, col)
                col += sU.shape[1]
                k_off += pk
            j_off += mj
        i_off += ni
    
    # Verify
    def verify(U_arr, V_arr, W_arr, n, m, p):
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
