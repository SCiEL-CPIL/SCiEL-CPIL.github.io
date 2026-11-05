"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W as lists of lists (2x2 entries per factor, 7 rank factors).
    Convention: C = A @ B where A is 2x2, B is 2x2.
    U[row, c] where row = i*2+j (i=0,1 row of A, j=0,1 col of A)
    V[row, c] where row = j*2+k (j=0,1 row of B, k=0,1 col of B)
    W[col, c] where col = k*2+i (k=0,1 col of B, i=0,1 row of A)
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    # U: 4 rows (i*2+j), 7 cols
    # V: 4 rows (j*2+k), 7 cols
    # W: 4 rows (k*2+i), 7 cols
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Factor 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00 (i=0,j=0)
    U[3][0] = 1  # A11 (i=1,j=1)
    V[0][0] = 1  # B00 (j=0,k=0)
    V[2][0] = 1  # B11 (j=1,k=1)
    W[0][0] = 1  # C00 (k=0,i=0)
    W[2][0] = 1  # C11 (k=1,i=1)
    
    # Factor 1: M2 = (A10+A11)B00
    U[2][1] = 1  # A10 (i=1,j=0)
    U[3][1] = 1  # A11 (i=1,j=1)
    V[0][1] = 1  # B00 (j=0,k=0)
    W[1][1] = 1  # C10 (k=0,i=1)
    
    # Factor 2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00 (i=0,j=0)
    V[1][2] = 1  # B01 (j=0,k=1)
    V[3][2] = -1 # B11 (j=1,k=1)
    W[1][2] = 1  # C01 (k=1,i=0)
    W[3][2] = 1  # C11 (k=1,i=1)
    
    # Factor 3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11 (i=1,j=1)
    V[2][3] = 1  # B10 (j=1,k=0)
    V[0][3] = -1 # B00 (j=0,k=0)
    W[0][3] = 1  # C00 (k=0,i=0)
    W[1][3] = 1  # C10 (k=0,i=1)
    
    # Factor 4: M5 = (A00+A10)B11
    U[0][4] = 1  # A00 (i=0,j=0)
    U[2][4] = 1  # A10 (i=1,j=0)
    V[3][4] = 1  # B11 (j=1,k=1)
    W[1][4] = 1  # C01 (k=1,i=0)
    W[0][4] = -1 # C00 (k=0,i=0)
    
    # Factor 5: M6 = (A11-A00)(B00+B10)
    U[3][5] = 1  # A11 (i=1,j=1)
    U[0][5] = -1 # A00 (i=0,j=0)
    V[0][5] = 1  # B00 (j=0,k=0)
    V[2][5] = 1  # B10 (j=1,k=0)
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # Factor 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01 (i=0,j=1)
    U[3][6] = -1 # A11 (i=1,j=1)
    V[1][6] = 1  # B01 (j=0,k=1)
    V[3][6] = 1  # B11 (j=1,k=1)
    W[0][6] = 1  # C00 (k=0,i=0)
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(U[0])
    # Check: for each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook baseline
    sb_rank = n * m * p
    U_sb = [[0] * sb_rank for _ in range(n * m)]
    V_sb = [[0] * sb_rank for _ in range(m * p)]
    W_sb = [[0] * sb_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = sb_rank
    
    # Try Strassen-based block decomposition
    # Split n=4 into 2+2, m=5 into 2+3 or 3+2, p=6 into 2+2+2 or 2+4 or 4+2 or 3+3
    # Key insight: if we can partition into blocks where some are 2x2x2, we save 2 per such block
    # vs schoolbook (7 vs 8 for 2x2x2)
    
    # Strategy: Try all partitions of n, m, p into sums of 2s and other sizes
    # For simplicity, try splitting n into [2,2], p into [2,2,2], and m into various
    # But m=5 is odd, so it can't be all 2s. Try m = [2,3] or [3,2]
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Try: n=[2,2], m=[2,3], p=[2,2,2]
    # Block sizes: (2,2,2), (2,2,2), (2,2,2) -> Strassen
    #              (2,3,2), (2,3,2), (2,3,2) -> schoolbook (2*3*2=12 each, 3 blocks = 36)
    #              Wait, m=2+3=5, so blocks are:
    # Row n-blocks: 2, 2
    # Col m-blocks: 2, 3
    # Col p-blocks: 2, 2, 2
    # Total blocks: 2*2*3 = 12 blocks
    # Blocks with all dims = 2: (2,2,2) -> 7 rank each. How many such blocks?
    # n=2, m=2, p=2: 1 block of each (n-block, m-block, p-block) combo
    # (n_blk=0, m_blk=0, p_blk=0): (2,2,2) -> Strassen, rank 7
    # (n_blk=0, m_blk=0, p_blk=1): (2,2,2) -> Strassen, rank 7
    # (n_blk=0, m_blk=0, p_blk=2): (2,2,2) -> Strassen, rank 7
    # (n_blk=1, m_blk=0, p_blk=0): (2,2,2) -> Strassen, rank 7
    # (n_blk=1, m_blk=0, p_blk=1): (2,2,2) -> Strassen, rank 7
    # (n_blk=1, m_blk=0, p_blk=2): (2,2,2) -> Strassen, rank 7
    # (n_blk=0, m_blk=1, p_blk=0): (2,3,2) -> schoolbook, rank 12
    # (n_blk=0, m_blk=1, p_blk=1): (2,3,2) -> schoolbook, rank 12
    # (n_blk=0, m_blk=1, p_blk=2): (2,3,2) -> schoolbook, rank 12
    # (n_blk=1, m_blk=1, p_blk=0): (2,3,2) -> schoolbook, rank 12
    # (n_blk=1, m_blk=1, p_blk=1): (2,3,2) -> schoolbook, rank 12
    # (n_blk=1, m_blk=1, p_blk=2): (2,3,2) -> schoolbook, rank 12
    # Total: 6*7 + 6*12 = 42 + 72 = 114
    # That's better than 120!
    
    def build_block_decomp(n_parts, m_parts, p_parts):
        """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
        if time.time() > deadline:
            return None
        
        total_rows_U = sum(n_parts) * sum(m_parts)  # = n * m
        total_rows_V = sum(m_parts) * sum(p_parts)  # = m * p
        total_rows_W = sum(p_parts) * sum(n_parts)  # = p * n
        
        # Build block-level matrices
        # For each triple (a, b, c) where a indexes n-blocks, b indexes m-blocks, c indexes p-blocks:
        # We have a sub-decomposition of size (n_parts[a], m_parts[b], p_parts[c])
        
        # First compute total rank
        total_rank = 0
        block_ranks = {}
        for a in range(len(n_parts)):
            for b in range(len(m_parts)):
                for c in range(len(p_parts)):
                    na, mb, pc = n_parts[a], m_parts[b], p_parts[c]
                    if na == 2 and mb == 2 and pc == 2:
                        block_ranks[(a,b,c)] = 7
                    else:
                        block_ranks[(a,b,c)] = na * mb * pc
                    total_rank += block_ranks[(a,b,c)]
        
        U = [[0] * total_rank for _ in range(total_rows_U)]
        V = [[0] * total_rank for _ in range(total_rows_V)]
        W = [[0] * total_rank for _ in range(total_rows_W)]
        
        # Compute offset for each block
        offset = {}
        cur = 0
        for a in range(len(n_parts)):
            for b in range(len(m_parts)):
                for c in range(len(p_parts)):
                    offset[(a,b,c)] = cur
                    cur += block_ranks[(a,b,c)]
        
        # Compute row/col offsets
        n_off = [0]
        for na in n_parts[:-1]:
            n_off.append(n_off[-1] + na)
        m_off = [0]
        for mb in m_parts[:-1]:
            m_off.append(m_off[-1] + mb)
        p_off = [0]
        for pc in p_parts[:-1]:
            p_off.append(p_off[-1] + pc)
        
        for a in range(len(n_parts)):
            for b in range(len(m_parts)):
                for c in range(len(p_parts)):
                    na, mb, pc = n_parts[a], m_parts[b], p_parts[c]
                    off = offset[(a,b,c)]
                    if na == 2 and mb == 2 and pc == 2:
                        # Use Strassen
                        for i in range(2):
                            for j in range(2):
                                u_row = (n_off[a] + i) * m + (m_off[b] + j)
                                for t in range(7):
                                    U[u_row][off + t] = sU[i*2+j][t]
                        for j in range(2):
                            for k in range(2):
                                v_row = (m_off[b] + j) * p + (p_off[c] + k)
                                for t in range(7):
                                    V[v_row][off + t] = sV[j*2+k][t]
                        for k in range(2):
                            for i in range(2):
                                w_row = (p_off[c] + k) * n + (n_off[a] + i)
                                for t in range(7):
                                    W[w_row][off + t] = sW[k*2+i][t]
                    else:
                        # Schoolbook for this block
                        idx = 0
                        for i in range(na):
                            for j in range(mb):
                                for k in range(pc):
                                    u_row = (n_off[a] + i) * m + (m_off[b] + j)
                                    v_row = (m_off[b] + j) * p + (p_off[c] + k)
                                    w_row = (p_off[c] + k) * n + (n_off[a] + i)
                                    U[u_row][off + idx] = 1
                                    V[v_row][off + idx] = 1
                                    W[w_row][off + idx] = 1
                                    idx += 1
        
        return U, V, W, total_rank
    
    # Try various partitions
    # n=4: [2,2]
    # m=5: [2,3], [3,2], [5], [1,4], [4,1], [1,1,3], [1,3,1], [3,1,1], [1,1,1,2], [2,1,1,1], [1,2,1,1], [1,1,2,1], [1,1,1,1,1]
    # p=6: [2,2,2], [2,4], [4,2], [3,3], [6], [2,1,3], [3,1,2], [1,2,3], [3,2,1], [1,3,2], [2,3,1], [1,1,4], [4,1,1], [1,4,1], [2,2,1,1], [1,1,2,2], [2,1,1,2], [1,2,2,1], [1,2,1,2], [2,1,2,1], [1,1,1,3], [3,1,1,1], [1,3,1,1], [1,1,3,1], [1,1,1,1,2], etc.
    
    candidate_partitions = [
        ([2,2], [2,3], [2,2,2]),
        ([2,2], [3,2], [2,2,2]),
        ([2,2], [2,3], [2,4]),
        ([2,2], [2,3], [4,2]),
        ([2,2], [2,3], [3,3]),
        ([2,2], [3,2], [2,4]),
        ([2,2], [3,2], [4,2]),
        ([2,2], [3,2], [3,3]),
        ([2,2], [1,4], [2,2,2]),
        ([2,2], [4,1], [2,2,2]),
        ([2,2], [2,1,2], [2,2,2]),
        ([2,2], [1,2,2], [2,2,2]),
        ([2,2], [2,2,1], [2,2,2]),
        ([2,2], [1,1,3], [2,2,2]),
        ([2,2], [3,1,1], [2,2,2]),
        ([2,2], [1,3,1], [2,2,2]),
        ([2,2], [2,2,2,2,2,2,2], [2,2,2]),  # m split into many 2s + 1
        ([4], [2,3], [2,2,2]),
        ([4], [3,2], [2,2,2]),
        ([2,2], [5], [2,2,2]),
    ]
    
    for n_parts, m_parts, p_parts in candidate_partitions:
        if time.time() > deadline:
            break
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        result = build_block_decomp(n_parts, m_parts, p_parts)
        if result is None:
            continue
        U_c, V_c, W_c, rank_c = result
        if rank_c < best_rank:
            # Verify
            if _verify_decomposition(n, m, p, U_c, V_c, W_c):
                best_rank = rank_c
                best_result = {"U": U_c, "V": V_c, "W": W_c}
    
    return best_result
# EVOLVE-BLOCK-END
