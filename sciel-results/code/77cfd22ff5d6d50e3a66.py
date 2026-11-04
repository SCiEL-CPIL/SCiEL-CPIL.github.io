import numpy as np

DIMENSIONS = [2, 3, 7]

def _strassen_2x2x2(u_offsets, v_offsets, w_offsets, rank_start, U, V, W):
    """
    Adds 7 rank terms for 2x2x2 multiplication using Strassen.
    u_offsets: list of 4 global indices for A (i*m + j)
    v_offsets: list of 4 global indices for B (j*p + k)
    w_offsets: list of 4 global indices for C (k*n + i)
    rank_start: current rank index
    U, V, W: factor matrices
    """
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00, A11. V: B00, B11. W: C00, C11
    U[u_offsets[0], rank_start] = 1
    U[u_offsets[3], rank_start] = 1
    V[v_offsets[0], rank_start] = 1
    V[v_offsets[3], rank_start] = 1
    W[w_offsets[0], rank_start] = 1
    W[w_offsets[3], rank_start] = 1
    rank_start += 1
    
    # M2 = (A10 + A11) B00
    # U: A10, A11. V: B00. W: C10
    U[u_offsets[2], rank_start] = 1
    U[u_offsets[3], rank_start] = 1
    V[v_offsets[0], rank_start] = 1
    W[w_offsets[2], rank_start] = 1
    rank_start += 1
    
    # M3 = A00 (B01 - B11)
    # U: A00. V: B01, -B11. W: C01
    U[u_offsets[0], rank_start] = 1
    V[v_offsets[1], rank_start] = 1
    V[v_offsets[3], rank_start] = -1
    W[w_offsets[1], rank_start] = 1
    rank_start += 1
    
    # M4 = A11 (B10 - B00)
    # U: A11. V: B10, -B00. W: C00, C10
    U[u_offsets[3], rank_start] = 1
    V[v_offsets[2], rank_start] = 1
    V[v_offsets[0], rank_start] = -1
    W[w_offsets[0], rank_start] = 1
    W[w_offsets[2], rank_start] = 1
    rank_start += 1
    
    # M5 = (A00 + A10) B11
    # U: A00, A10. V: B11. W: C00, C01
    U[u_offsets[0], rank_start] = 1
    U[u_offsets[2], rank_start] = 1
    V[v_offsets[3], rank_start] = 1
    W[w_offsets[0], rank_start] = 1
    W[w_offsets[1], rank_start] = 1
    rank_start += 1
    
    # M6 = (A01 - A00)(B00 + B10)
    # U: A01, -A00. V: B00, B10. W: C11
    U[u_offsets[1], rank_start] = 1
    U[u_offsets[0], rank_start] = -1
    V[v_offsets[0], rank_start] = 1
    V[v_offsets[2], rank_start] = 1
    W[w_offsets[3], rank_start] = 1
    rank_start += 1
    
    # M7 = (A10 - A11)(B01 + B11)
    # U: A10, -A11. V: B01, B11. W: C00, C11
    U[u_offsets[2], rank_start] = 1
    U[u_offsets[3], rank_start] = -1
    V[v_offsets[1], rank_start] = 1
    V[v_offsets[3], rank_start] = 1
    W[w_offsets[0], rank_start] = 1
    W[w_offsets[3], rank_start] = 1
    rank_start += 1
    
    return rank_start

def _schoolbook_block(u_offsets, v_offsets, w_offsets, rank_start, U, V, W):
    """
    Adds n*m*p rank terms for a block multiplication.
    u_offsets: list of n*m global indices
    v_offsets: list of m*p global indices
    w_offsets: list of p*n global indices
    """
    n = int(np.sqrt(len(w_offsets))) # This is fragile, better to pass n,m,p
    # We will pass n, m, p explicitly in the main logic instead of this generic one
    # For now, assume we call this for small blocks where we know n,m,p
    pass

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Helper to get global indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Block 1: A (2x3) * B1 (3x2) -> C1 (2x2)
    # B1 is cols 0,1 of B
    # Split A into A1a (2x2, cols 0,1) and A1b (2x1, col 2)
    # Split B1 into B1a (2x2, rows 0,1) and B1b (1x2, row 2)
    
    # A1a * B1a (2x2 * 2x2)
    u_off = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_off = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_off = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    r = _strassen_2x2x2(u_off, v_off, w_off, r, U, V, W)
    
    # A1b * B1b (2x1 * 1x2)
    # Rank 4
    # i in 0..1, j=2, k in 0..1
    for i in range(2):
        for k in range(2):
            U[u_idx(i, 2), r] = 1
            V[v_idx(2, k), r] = 1
            W[w_idx(k, i), r] = 1
            r += 1
            
    # Block 2: A (2x3) * B2 (3x2) -> C2 (2x2)
    # B2 is cols 2,3 of B
    # Same structure
    
    # A2a * B2a (2x2 * 2x2)
    u_off = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_off = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    w_off = [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    r = _strassen_2x2x2(u_off, v_off, w_off, r, U, V, W)
    
    # A2b * B2b (2x1 * 1x2)
    for i in range(2):
        for k in range(2):
            U[u_idx(i, 2), r] = 1
            V[v_idx(2, 2+k), r] = 1
            W[w_idx(2+k, i), r] = 1
            r += 1
            
    # Block 3: A (2x3) * B3 (3x3) -> C3 (2x3)
    # B3 is cols 4,5,6 of B
    # Split A into A3a (2x2, cols 0,1) and A3b (2x1, col 2)
    # Split B3 into B3a (2x3, rows 0,1) and B3b (1x3, row 2)
    
    # A3b * B3b (2x1 * 1x3)
    # Rank 6
    for i in range(2):
        for k in range(3):
            U[u_idx(i, 2), r] = 1
            V[v_idx(2, 4+k), r] = 1
            W[w_idx(4+k, i), r] = 1
            r += 1
            
    # A3a * B3a (2x2 * 2x3)
    # Split B3a into B3a1 (2x2, cols 0,1) and B3a2 (2x1, col 2)
    
    # A3a * B3a1 (2x2 * 2x2)
    u_off = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_off = [v_idx(0,4), v_idx(0,5), v_idx(1,4), v_idx(1,5)]
    w_off = [w_idx(4,0), w_idx(5,0), w_idx(4,1), w_idx(5,1)]
    r = _strassen_2x2x2(u_off, v_off, w_off, r, U, V, W)
    
    # A3a * B3a2 (2x2 * 2x1)
    # Rank 4
    for i in range(2):
        for j in range(2):
            U[u_idx(i, j), r] = 1
            V[v_idx(j, 6), r] = 1
            W[w_idx(6, i), r] = 1
            r += 1
            
    if r != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {r}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
