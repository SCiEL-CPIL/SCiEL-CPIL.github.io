"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 22.
    # Split p=4 into two parts of size 2.
    # Each part is M_{2,3,2}.
    # M_{2,3,2} is decomposed into M_{2,2,2} (rank 7) and M_{2,1,2} (rank 4).
    # Total rank per part: 11.
    # Two parts: 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Helper to add Strassen terms for a 2x2x2 block
    # i in 0..1, j in 0..1, k in k_offset..k_offset+1
    def add_strassen_block(k_offset, start_col):
        # M1
        add_term(0, 0 + k_offset*2, 0 + k_offset*n, 1, start_col) # U[0,0]=1 (idx 0), V[0,0]=1 (idx 0+k*2), W[0,0]=1 (idx 0+k*2)
        add_term(3, 3 + k_offset*2, 3 + k_offset*n, 1, start_col) # U[1,1]=1 (idx 3), V[1,1]=1 (idx 3+k*2), W[1,1]=1 (idx 3+k*2)
        
        # M2
        add_term(2, 0 + k_offset*2, 1 + k_offset*n, 1, start_col+1) # U[1,0]=1 (idx 2), V[0,0]=1, W[0,1]=1 (idx 1+k*2)
        add_term(3, 0 + k_offset*2, 1 + k_offset*n, 1, start_col+1) # U[1,1]=1 (idx 3), V[0,0]=1, W[0,1]=1
        
        # M3
        add_term(0, 2 + k_offset*2, 2 + k_offset*n, 1, start_col+2) # U[0,0]=1 (idx 0), V[1,0]=-1 (idx 2+k*2), W[1,0]=1 (idx 2+k*2)
        add_term(0, 3 + k_offset*2, 2 + k_offset*n, -1, start_col+2) # U[0,0]=1, V[1,1]=1, W[1,0]=1
        # Correction: M3 = a00(b10 - b11). 
        # U: a00 -> idx 0.
        # V: b10 -> idx 2+k*2, b11 -> idx 3+k*2.
        # W: c10 -> idx 2+k*2.
        # So:
        # add_term(0, 2+k*2, 2+k*2, 1, ...)
        # add_term(0, 3+k*2, 2+k*2, -1, ...)
        
        # Let's re-verify M3 indices.
        # U[0,0]=1.
        # V[1,0]=-1, V[1,1]=1.
        # W[1,0]=1.
        # My previous add_term calls were:
        # add_term(0, 2 + k_offset*2, 2 + k_offset*n, 1, start_col+2) -> U[0], V[1,0], W[1,0]. Correct.
        # add_term(0, 3 + k_offset*2, 2 + k_offset*n, -1, start_col+2) -> U[0], V[1,1], W[1,0]. Correct.

        # M4
        # M4 = a11(b01 - b00)
        # U[1,1]=1 (idx 3).
        # V[0,1]=1 (idx 1+k*2), V[0,0]=-1 (idx 0+k*2).
        # W[0,1]=1 (idx 1+k*2).
        add_term(3, 1 + k_offset*2, 1 + k_offset*n, 1, start_col+3)
        add_term(3, 0 + k_offset*2, 1 + k_offset*n, -1, start_col+3)

        # M5
        # M5 = (a00 + a01)b11
        # U[0,0]=1 (idx 0), U[0,1]=1 (idx 1).
        # V[1,1]=1 (idx 3+k*2).
        # W[1,0]=1 (idx 2+k*2).
        add_term(0, 3 + k_offset*2, 2 + k_offset*n, 1, start_col+4)
        add_term(1, 3 + k_offset*2, 2 + k_offset*n, 1, start_col+4)

        # M6
        # M6 = (a10 - a00)(b00 + b01)
        # U[1,0]=1 (idx 2), U[0,0]=-1 (idx 0).
        # V[0,0]=1 (idx 0+k*2), V[0,1]=1 (idx 1+k*2).
        # W[0,0]=1 (idx 0+k*2).
        add_term(2, 0 + k_offset*2, 0 + k_offset*n, 1, start_col+5)
        add_term(2, 1 + k_offset*2, 0 + k_offset*n, 1, start_col+5)
        add_term(0, 0 + k_offset*2, 0 + k_offset*n, -1, start_col+5)
        add_term(0, 1 + k_offset*2, 0 + k_offset*n, -1, start_col+5)

        # M7
        # M7 = (a01 - a11)(b10 + b11)
        # U[0,1]=1 (idx 1), U[1,1]=-1 (idx 3).
        # V[1,0]=1 (idx 2+k*2), V[1,1]=1 (idx 3+k*2).
        # W[1,1]=1 (idx 3+k*2).
        add_term(1, 2 + k_offset*2, 3 + k_offset*n, 1, start_col+6)
        add_term(1, 3 + k_offset*2, 3 + k_offset*n, 1, start_col+6)
        add_term(3, 2 + k_offset*2, 3 + k_offset*n, -1, start_col+6)
        add_term(3, 3 + k_offset*2, 3 + k_offset*n, -1, start_col+6)

    # Helper to add trivial terms for a 2x1x2 block (j=2)
    # i in 0..1, j=2, k in k_offset..k_offset+1
    def add_trivial_block(k_offset, start_col):
        # C[i,k] = A[i,2] B[2,k]
        # 4 terms
        # i=0, k=0
        add_term(2, 4 + k_offset*2, 0 + k_offset*n, 1, start_col) # U[0,2]=1 (idx 2), V[2,0]=1 (idx 4+k*2), W[0,0]=1
        # i=0, k=1
        add_term(2, 5 + k_offset*2, 1 + k_offset*n, 1, start_col+1) # U[0,2]=1, V[2,1]=1 (idx 5+k*2), W[0,1]=1
        # i=1, k=0
        add_term(5, 4 + k_offset*2, 2 + k_offset*n, 1, start_col+2) # U[1,2]=1 (idx 5), V[2,0]=1, W[1,0]=1
        # i=1, k=1
        add_term(5, 5 + k_offset*2, 3 + k_offset*n, 1, start_col+3) # U[1,2]=1, V[2,1]=1, W[1,1]=1

    # Block 1: k in 0, 1
    add_strassen_block(0, 0)
    add_trivial_block(0, 7)
    
    # Block 2: k in 2, 3
    add_strassen_block(1, 11)
    add_trivial_block(1, 18)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
