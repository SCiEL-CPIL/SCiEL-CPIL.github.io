"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_tensors():
    """
    Returns U, V, W tensors for 2x2x2 Strassen decomposition.
    U[i, j, r], V[j, k, r], W[k, i, r]
    """
    U = [[0]*7 for _ in range(4)] # 2x2
    V = [[0]*7 for _ in range(4)] # 2x2
    W = [[0]*7 for _ in range(4)] # 2x2
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # W[k, i]: k=0,i=0 (C00); k=1,i=1 (C11) -> W[0,0]=1, W[1,1]=1
    
    # M2: (A10+A11)B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1 # C10: k=0, i=1 -> W[0,1]=1

    # M3: A00(B01-B11) -> C01
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01: k=1, i=0 -> W[1,0]=1

    # M4: A11(B10-B00) -> C10 - C00
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1 # C00: k=0, i=0 -> W[0,0]=-1
    W[0][1] = 1  # C10: k=0, i=1 -> W[0,1]=1

    # M5: (A00+A01)B11 -> C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1 # C01: k=1, i=0 -> W[1,0]=1

    # M6: (A10-A00)(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11: k=1, i=1 -> W[1,1]=1

    # M7: (A01-A11)(B10+B11) -> C00
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00: k=0, i=0 -> W[0,0]=1

    return U, V, W

def _standard_2x2x1_tensors():
    """
    Returns U, V, W tensors for 2x2x1 standard decomposition.
    U[i, j, r], V[j, k, r], W[k, i, r]
    k is fixed to 0 in local coords.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    
    # C00 = A00 B00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # C00 += A01 B10
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # C10 = A10 B00
    U[2][2] = 1
    V[0][2] = 1
    W[0][2] = 1 # Wait, C10 is i=1, k=0. W[k, i] = W[0, 1].
    # My W indexing in seed was W[k*n+i]. Here n=2.
    # C00: i=0, k=0 -> W[0,0]
    # C01: i=0, k=1 -> W[1,0]
    # C10: i=1, k=0 -> W[0,1]
    # C11: i=1, k=1 -> W[1,1]
    
    # Let's re-verify W indices for standard 2x2x1.
    # Term 1: A00 B00 -> C00. W[0,0] = 1.
    # Term 2: A01 B10 -> C00. W[0,0] += 1.
    # Term 3: A10 B00 -> C10. W[0,1] = 1.
    # Term 4: A11 B10 -> C10. W[0,1] += 1.
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][0] = 1 # W[0,0] is now 2? No, separate terms.
    # The decomposition is a sum. So we just add to the matrices.
    # But here we are building the matrices for the block.
    # We need to append these to the global U, V, W.
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(j_offset, k_offsets, block_U, block_V, block_W, rank_offset):
        # block_U is 4xR, block_V is 4xR, block_W is 4xR (local 2x2 indices)
        # Map local (i, j) to global (i, j+j_offset)
        # Map local (j, k) to global (j+j_offset, k) -- wait, k is global in this block?
        # The block is for specific j range and k range.
        # Local j is 0,1. Global j is j_offset, j_offset+1.
        # Local k is 0,1 (for 2x2x2) or 0 (for 2x2x1).
        # Global k is k_offsets[local_k].
        
        R = len(block_U[0])
        for r in range(R):
            # U
            for li in range(2):
                for lj in range(2):
                    val = block_U[li*2 + lj][r]
                    if val != 0:
                        gi = li
                        gj = lj + j_offset
                        U[gi*m + gj][rank_offset + r] = val
            
            # V
            for lj in range(2):
                for lk in range(2): # Loop up to 2, but 2x2x1 only uses 0
                    val = block_V[lj*2 + lk][r]
                    if val != 0:
                        gj = lj + j_offset
                        # Determine global k
                        # For 2x2x2, k_offsets has 2 entries.
                        # For 2x2x1, k_offsets has 1 entry.
                        if lk < len(k_offsets):
                            gk = k_offsets[lk]
                        else:
                            continue
                        V[gj*p + gk][rank_offset + r] = val
            
            # W
            for lk in range(2):
                for li in range(2):
                    val = block_W[lk*2 + li][r]
                    if val != 0:
                        if lk < len(k_offsets):
                            gk = k_offsets[lk]
                        else:
                            continue
                        gi = li
                        W[gk*n + gi][rank_offset + r] = val
        return rank_offset + R

    # Get Strassen tensors
    sU, sV, sW = _strassen_tensors()
    # Get Standard 2x2x1 tensors
    # We need to construct them properly
    std_U = [[0]*4 for _ in range(4)]
    std_V = [[0]*4 for _ in range(4)]
    std_W = [[0]*4 for _ in range(4)]
    # Term 1: A00 B0 -> C0
    std_U[0][0] = 1; std_V[0][0] = 1; std_W[0][0] = 1
    # Term 2: A01 B1 -> C0
    std_U[1][1] = 1; std_V[1][1] = 1; std_W[0][1] = 1
    # Term 3: A10 B0 -> C1
    std_U[2][2] = 1; std_V[0][2] = 1; std_W[0][2] = 1 # W[0,1] is C10
    # Term 4: A11 B1 -> C1
    std_U[3][3] = 1; std_V[1][3] = 1; std_W[0][3] = 1 # W[0,1] is C10
    
    # Note: std_W[0][1] accumulates to 2?
    # In the decomposition, we sum terms.
    # C10 = A10 B0 + A11 B1.
    # So W[0,1] should be 1 for term 3 and 1 for term 4.
    # My loop above adds them.
    
    rank_offset = 0
    
    # Block 1: j in {0, 1}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    
    # k=0,1 (Strassen)
    rank_offset = add_block(0, [0, 1], sU, sV, sW, rank_offset)
    # k=2,3 (Strassen)
    rank_offset = add_block(0, [2, 3], sU, sV, sW, rank_offset)
    # k=4,5 (Strassen)
    rank_offset = add_block(0, [4, 5], sU, sV, sW, rank_offset)
    # k=6 (Standard)
    rank_offset = add_block(0, [6], std_U, std_V, std_W, rank_offset)
    
    # Block 2: j in {2, 3}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    
    # k=0,1 (Strassen)
    rank_offset = add_block(2, [0, 1], sU, sV, sW, rank_offset)
    # k=2,3 (Strassen)
    rank_offset = add_block(2, [2, 3], sU, sV, sW, rank_offset)
    # k=4,5 (Strassen)
    rank_offset = add_block(2, [4, 5], sU, sV, sW, rank_offset)
    # k=6 (Standard)
    rank_offset = add_block(2, [6], std_U, std_V, std_W, rank_offset)
    
    # rank_offset should be 50
    if rank_offset != rank:
        # Fallback to schoolbook if something went wrong (should not happen)
        return {
            "U": [[0] * (n*m*p) for _ in range(n*m)],
            "V": [[0] * (n*m*p) for _ in range(m*p)],
            "W": [[0] * (n*m*p) for _ in range(p*n)]
        }
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
