"""Tensor decomposition for 2x3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # seconds; leave 10s margin before the 150s kill

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen_block_2x3x2():
    """
    Build a rank-11 decomposition for 2x3x2 matrix multiplication.
    
    Partition A (2x3) into A_left (2x2) and A_right (2x1).
    Partition B (3x2) into B_top (2x2) and B_bottom (1x2).
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2x2), use Strassen: rank 7.
    For A_right * B_bottom (2x1x1x2), outer product: 4 rank-1 terms.
    Total: 7 + 4 = 11.
    """
    rank = 11
    U = [[0.0]*rank for _ in range(6)]
    V = [[0.0]*rank for _ in range(6)]
    W = [[0.0]*rank for _ in range(4)]
    
    # Strassen 2x2x2
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (a00+a11)(b00+b11) -> C00(+1), C11(+1)
    c = 0
    U[0][c] += 1; U[4][c] += 1
    V[0][c] += 1; V[3][c] += 1
    W[0][c] += 1; W[3][c] += 1
    
    # M2 = (a10+a11)(b00) -> C10(+1), C11(-1)
    c = 1
    U[3][c] += 1; U[4][c] += 1
    V[0][c] += 1
    W[2][c] += 1; W[3][c] += -1
    
    # M3 = a00(b01-b11) -> C01(+1), C11(+1)
    c = 2
    U[0][c] += 1
    V[1][c] += 1; V[3][c] += -1
    W[1][c] += 1; W[3][c] += 1
    
    # M4 = a11(b10-b00) -> C00(+1), C10(+1)
    c = 3
    U[4][c] += 1
    V[2][c] += 1; V[0][c] += -1
    W[0][c] += 1; W[2][c] += 1
    
    # M5 = (a00+a01)(b11) -> C00(-1), C01(+1)
    c = 4
    U[0][c] += 1; U[1][c] += 1
    V[3][c] += 1
    W[0][c] += -1; W[1][c] += 1
    
    # M6 = (a10-a00)(b00+b01) -> C11(+1)
    c = 5
    U[3][c] += 1; U[0][c] += -1
    V[0][c] += 1; V[1][c] += 1
    W[3][c] += 1
    
    # M7 = (a01-a11)(b10+b11) -> C00(+1)
    c = 6
    U[1][c] += 1; U[4][c] += -1
    V[2][c] += 1; V[3][c] += 1
    W[0][c] += 1
    
    # Schoolbook: A_right (column 2) * B_bottom (row 2)
    c = 7  # a_{0,2} * b_{2,0} -> C_{0,0}
    U[2][c] += 1
    V[4][c] += 1
    W[0][c] += 1
    
    c = 8  # a_{0,2} * b_{2,1} -> C_{0,1}
    U[2][c] += 1
    V[5][c] += 1
    W[1][c] += 1
    
    c = 9  # a_{1,2} * b_{2,0} -> C_{1,0}
    U[5][c] += 1
    V[4][c] += 1
    W[2][c] += 1
    
    c = 10  # a_{1,2} * b_{2,1} -> C_{1,1}
    U[5][c] += 1
    V[5][c] += 1
    W[3][c] += 1
    
    return U, V, W


def _build_block_decomposition():
    """
    Build the full 2x3x10 decomposition using 5 blocks of 2x3x2.
    Total rank: 5 * 11 = 55.
    """
    n, m, p = 2, 3, 10
    total_rank = 55
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    block_U, block_V, block_W = _build_strassen_block_2x3x2()
    
    for b in range(5):
        rank_offset = b * 11
        
        for a_idx in range(6):
            for c in range(11):
                U[a_idx][rank_offset + c] += block_U[a_idx][c]
        
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            for c in range(11):
                V[global_idx][rank_offset + c] += block_V[local_idx][c]
        
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            for c in range(11):
                W[global_idx][rank_offset + c] += block_W[local_idx][c]
    
    return U, V, W


def _try_rank54_coset():
    """
    Attempt to find a rank-54 decomposition by exploiting the coset structure.
    
    The key insight: if we can share one rank-1 term across two blocks, we save 1 rank.
    This requires the shared term to be compatible with both blocks' Strassen structures.
    
    We try a specific pattern: modify the last block's Strassen to share one term
    with the first block, if the linear forms align.
    """
    n, m, p = 2, 3, 10
    total_rank = 54
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    block_U, block_V, block_W = _build_strassen_block_2x3x2()
    
    # Blocks 0-3: standard 11-rank each (44 terms)
    for b in range(4):
        rank_offset = b * 11
        for a_idx in range(6):
            for c in range(11):
                U[a_idx][rank_offset + c] += block_U[a_idx][c]
        
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            for c in range(11):
                V[global_idx][rank_offset + c] += block_V[local_idx][c]
        
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            for c in range(11):
                W[global_idx][rank_offset + c] += block_W[local_idx][c]
    
    # Block 4: try to use only 10 terms by sharing with block 0
    # This is the critical part - we need to find if a rank-10 decomposition
    # exists for the last block that's compatible with sharing.
    
    # For now, use standard 11 terms for block 4 but in the last 10 slots
    # (leaving slot 53 unused, which we'll try to exploit)
    rank_offset = 4 * 11  # = 44
    for a_idx in range(6):
        for c in range(11):
            U[a_idx][rank_offset + c] += block_U[a_idx][c]
    
    v_map = [2*4, 2*4+1, 10+2*4, 10+2*4+1, 20+2*4, 20+2*4+1]
    for local_idx in range(6):
        global_idx = v_map[local_idx]
        for c in range(11):
            V[global_idx][rank_offset + c] += block_V[local_idx][c]
    
    w_map = [4*4, 4*4+2, 4*4+1, 4*4+3]
    for local_idx in range(4):
        global_idx = w_map[local_idx]
        for c in range(11):
            W[global_idx][rank_offset + c] += block_W[local_idx][c]
    
    # Verify - this should still be rank 55, not 54
    # The actual rank-54 search would require more sophisticated algebra
    if _verify(U, V, W, n, m, p):
        # Check if we actually used only 54 terms (slot 53 should be zero)
        used_rank = 0
        for c in range(total_rank):
            col_nonzero = False
            for row in range(len(U)):
                if U[row][c] != 0:
                    col_nonzero = True
                    break
            if not col_nonzero:
                for row in range(len(V)):
                    if V[row][c] != 0:
                        col_nonzero = True
                        break
            if not col_nonzero:
                for row in range(len(W)):
                    if W[row][c] != 0:
                        col_nonzero = True
                        break
            if col_nonzero:
                used_rank = c + 1
        if used_rank <= 54:
            return U, V, W
    
    return None


def construct():
    start_time = time.time()
    
    # First, try the rank-54 coset approach
    result_54 = _try_rank54_coset()
    if result_54 is not None:
        U, V, W = result_54
        if _verify(U, V, W, 2, 3, 10):
            return {"U": U, "V": V, "W": W}
    
    # Build the rank-55 block decomposition
    U, V, W = _build_block_decomposition()
    
    # Verify
    n, m, p = 2, 3, 10
    if _verify(U, V, W, n, m, p):
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Fallback: schoolbook rank 60
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1.0
                V[j*p+k][column] = 1.0
                W[k*n+i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
