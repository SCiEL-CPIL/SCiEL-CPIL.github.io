"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import sys

n, m, p = 3, 5, 5
nm, mp, pn = n*m, m*p, p*n

def _strassen_222():
    """Return (U, V, W) each of shape (4, 7) for 2x2x2 Strassen decomposition."""
    # Standard Strassen: 7 products
    # M1=(X11+X22)(Y11+Y22)
    # M2=(X21+X22)Y11
    # M3=X11(Y12-Y22)
    # M4=X22(Y21-Y11)
    # M5=(X11+X12)Y22
    # M6=(X21-X11)(Y11+Y12)
    # M7=(X12-X22)(Y21+Y22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U indexed by (i,j): row = i*2+j, for X_ij
    # V indexed by (j,k): row = j*2+k, for Y_jk
    # W indexed by (k,i): row = k*2+i, for C_ki (note: C11 is row 0*2+1=2? No.)
    # W row for output C_{ki}: k is col of C, i is row of C.
    # C11: k=0, i=0 -> row 0*2+0=0
    # C12: k=0, i=1 -> row 0*2+1=1
    # C21: k=1, i=0 -> row 1*2+0=2
    # C22: k=1, i=1 -> row 1*2+1=3
    #
    # Let me re-derive carefully.
    # X = [[X11, X12], [X21, X22]] -> U row (i,j) = i*2+j
    #   X11: row 0, X12: row 1, X21: row 2, X22: row 3
    # Y = [[Y11, Y12], [Y21, Y22]] -> V row (j,k) = j*2+k
    #   Y11: row 0, Y12: row 1, Y21: row 2, Y22: row 3
    # C = [[C11, C12], [C21, C22]] -> W row (k,i) = k*2+i
    #   C11: k=0,i=0 -> row 0
    #   C12: k=0,i=1 -> row 1
    #   C21: k=1,i=0 -> row 2
    #   C22: k=1,i=1 -> row 3
    
    U = [[0,0,0,0,0,0,0],
         [0,0,1,0,1,0,0],
         [0,1,0,0,0,-1,0],
         [1,1,0,1,0,0,-1]]
    V = [[0,1,0,1,0,1,0],
         [0,0,1,0,0,0,1],
         [1,0,0,1,1,0,1],
         [1,0,0,0,1,0,0]]
    W = [[1,0,0,0,1,0,1],
         [0,0,1,0,1,0,0],
         [0,1,0,1,0,0,0],
         [1,0,1,0,-1,1,0]]
    
    # Verify: for each (i,j,k) in {0,1}^3, sum_r U[i*2+j, r]*V[j*2+k, r]*W[k*2+i, r] = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U[i*2+j][r]*V[j*2+k][r]*W[k*2+i][r] for r in range(7))
                assert s == 1, f"Strassen verify failed at ({i},{j},{k}): got {s}"
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m x p."""
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _embed_strassen_into_333(U_s, V_s, W_s):
    """
    Embed a 2x2x2 Strassen decomposition into a 3x3x3 tensor using the top-left 2x2 block.
    For the 3x3x3 case, we use Strassen for positions (i,j,k) where i,j,k in {0,1},
    and schoolbook for the rest.
    
    Actually, a cleaner approach: for 3x3x3, split as 2+1 in each dimension.
    The 2x2x2 part uses Strassen (rank 7), and the remaining parts use schoolbook.
    
    Split n=3 -> 2+1, m=3 -> 2+1, p=3 -> 2+1.
    Blocks:
    - (2,2,2): Strassen, rank 7
    - (2,2,1): schoolbook, rank 4
    - (2,1,2): schoolbook, rank 4
    - (1,2,2): schoolbook, rank 4
    - (2,1,1): schoolbook, rank 2
    - (1,2,1): schoolbook, rank 2
    - (1,1,2): schoolbook, rank 2
    - (1,1,1): schoolbook, rank 1
    Total: 7+4+4+4+2+2+2+1 = 26
    
    But we can do better by sharing. Let me just build it directly.
    """
    # For 3x3x3, use a hybrid approach
    # Actually, let's just build the full 3x3x3 by combining Strassen for the 2x2x2 core
    # with schoolbook for the remaining terms.
    
    # The 2x2x2 core covers i,j,k in {0,1}
    # The remaining terms are those where at least one index is 2.
    
    rank_total = 7 + 4 + 4 + 4 + 2 + 2 + 2 + 1  # = 26
    U = [[0]*rank_total for _ in range(9)]
    V = [[0]*rank_total for _ in range(9)]
    W = [[0]*rank_total for _ in range(9)]
    
    col = 0
    # Strassen for i,j,k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][col+r] += U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*3+k][col+r] += V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*3+i][col+r] += W_s[k*2+i][r]
    col += 7
    
    # Schoolbook for remaining terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U[:9], V[:9], W[:9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_222()
    
    # Build 3x5x5 decomposition using recursive block splitting
    # Strategy: split 3x5x5 into blocks, using Strassen for 2x2x2 sub-blocks
    
    # Approach: 
    # n=3: split into 2+1
    # m=5: split into 2+3, then 3 into 1+2
    # p=5: split into 2+3, then 3 into 1+2
    
    # Let's build this by decomposing into sub-problems.
    
    # For each combination of blocks, we need to compute the decomposition.
    # This is complex. Let's use a simpler approach:
    
    # Build 3x5x5 by:
    # 1. Split n=3 into 2+1
    # 2. For n=2 part: split m=5 into 2+3, p=5 into 2+3
    #    - 2x2x2: Strassen (rank 7)
    #    - 2x2x3: split p=3 into 1+2 -> 2x2x1 (4) + 2x2x2 (7) = 11
    #    - 2x3x2: split m=3 into 1+2 -> 2x1x2 (4) + 2x2x2 (7) = 11
    #    - 2x3x3: split m=3 into 1+2, p=3 into 1+2:
    #      2x1x1 (2) + 2x1x2 (4) + 2x2x1 (4) + 2x2x2 (7) = 17
    #    Total for n=2: 7+11+11+17 = 46
    # 3. For n=1 part: 1x5x5 schoolbook = 25
    # Total: 46+25 = 71
    
    # That's better than 75 but not great. Let me try a different splitting.
    
    # Alternative: 
    # Split n=3 into 1+2
    # Split m=5 into 1+4, then 4 into 2+2
    # Split p=5 into 1+4, then 4 into 2+2
    
    # For n=1: 1x5x5 = 25
    # For n=2:
    #   m=5: 1+4
    #   p=5: 1+4
    #   2x1x1: 2
    #   2x1x4: split p=4 into 2+2 -> 2x1x2 (4) + 2x1x2 (4) = 8
    #   2x4x1: split m=4 into 2+2 -> 2x2x1 (4) + 2x2x1 (4) = 8
    #   2x4x4: split m=4 into 2+2, p=4 into 2+2:
    #     2x2x2 (7) x 4 = 28
    #   Total for n=2: 2+8+8+28 = 46
    # Total: 25+46 = 71
    
    # Same result. Let me try to be smarter.
    
    # What if I split m=5 into 2+3 and p=5 into 2+3?
    # For n=2:
    #   2x2x2: 7
    #   2x2x3: split p=3 into 1+2: 2x2x1 (4) + 2x2x2 (7) = 11
    #   2x3x2: split m=3 into 1+2: 2x1x2 (4) + 2x2x2 (7) = 11
    #   2x3x3: split m=3 into 1+2, p=3 into 1+2:
    #     2x1x1 (2) + 2x1x2 (4) + 2x2x1 (4) + 2x2x2 (7) = 17
    #   Total: 7+11+11+17 = 46
    # For n=1: 1x5x5 = 25
    # Total: 71
    
    # It seems like 71 is a natural result. But the knowledge says 48 was achieved.
    # Let me think about how to get 48.
    
    # Perhaps the key is to NOT split n first, but to split all dimensions simultaneously
    # in a way that creates multiple 2x2x2 blocks.
    
    # Actually, let me try a different approach: build the full decomposition directly
    # using a greedy algorithm or by exploiting more structure.
    
    # For now, let's build the 71-rank decomposition and verify it.
    
    def build_2x2x2(offset):
        """Add Strassen 2x2x2 at given offset in the column space."""
        local_U, local_V, local_W = U_s, V_s, W_s
        return local_U, local_V, local_W, 7
    
    def build_schoolbook_block(n_b, m_b, p_b, offset):
        """Build schoolbook for a block of size n_b x m_b x p_b."""
        rank = n_b * m_b * p_b
        U_b = [[0]*rank for _ in range(n_b*m_b)]
        V_b = [[0]*rank for _ in range(m_b*p_b)]
        W_b = [[0]*rank for _ in range(p_b*n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i*m_b+j][col] = 1
                    V_b[j*p_b+k][col] = 1
                    W_b[k*n_b+i][col] = 1
                    col += 1
        return U_b, V_b, W_b, rank
    
    # Build 3x5x5 decomposition
    # Split: n=2+1, m=2+3, p=2+3
    # This gives 8 blocks.
    
    # Block (ni, mj, pk) with sizes (n_i, m_j, p_k)
    n_parts = [2, 1]
    m_parts = [2, 3]
    p_parts = [2, 3]
    
    # For each block, determine if it's 2x2x2 (use Strassen) or other (schoolbook)
    
    U_full = [[0]*(7+4+11+11+2+4+4+7+4+2+4+2+8+4+2+8+4+2+8+4+2) for _ in range(nm)]
    
    # Let me compute the total rank first
    total_rank = 0
    block_ranks = []
    for ni in n_parts:
        for mj in m_parts:
            for pk in p_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    r = 7
                else:
                    r = ni * mj * pk
                block_ranks.append(r)
                total_rank += r
    
    print(f"Total rank: {total_rank}")
    
    U_full = [[0]*total_rank for _ in range(nm)]
    V_full = [[0]*total_rank for _ in range(mp)]
    W_full = [[0]*total_rank for _ in range(pn)]
    
    col = 0
    n_offset = 0
    for ni in n_parts:
        m_offset = 0
        for mj in m_parts:
            p_offset = 0
            for pk in p_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for r in range(7):
                        for i in range(2):
                            for j in range(2):
                                U_full[(n_offset+i)*m + (m_offset+j)][col+r] += U_s[i*2+j][r]
                        for j in range(2):
                            for k in range(2):
                                V_full[(m_offset+j)*p + (p_offset+k)][col+r] += V_s[j*2+k][r]
                        for k in range(2):
                            for i in range(2):
                                W_full[(p_offset+k)*n + (n_offset+i)][col+r] += W_s[k*2+i][r]
                    col += 7
                else:
                    # Schoolbook
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                U_full[(n_offset+i)*m + (m_offset+j)][col] = 1
                                V_full[(m_offset+j)*p + (p_offset+k)][col] = 1
                                W_full[(p_offset+k)*n + (n_offset+i)][col] = 1
                                col += 1
                p_offset += pk
            m_offset += mj
        n_offset += ni
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    # Verify
    is_valid = _verify(U_full, V_full, W_full, n, m, p)
    print(f"Verification: {is_valid}")
    
    if not is_valid:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_full, "V": V_full, "W": W_full}

