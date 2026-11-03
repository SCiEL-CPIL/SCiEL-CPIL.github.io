"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (6, 44)
    # V: (m*p, rank) -> (24, 44)
    # W: (p*n, rank) -> (16, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(rank_offset, a_rows, b_rows, b_cols):
        # a_rows: [0, 1]
        # b_rows: [0, 1]
        # b_cols: [c0, c1]
        c0, c1 = b_cols
        r0, r1 = a_rows
        b0, b1 = b_rows
        
        # M1: (A[r0,0] + A[r1,1]) * (B[b0,c0] + B[b1,c1])
        # U: A[r0,0] (idx r0*m+0), A[r1,1] (idx r1*m+1)
        # V: B[b0,c0] (idx b0*p+c0), B[b1,c1] (idx b1*p+c1)
        # W: C[c0, r0] (idx c0*n+r0), C[c1, r1] (idx c1*n+r1)
        U[r0 * m + 0][rank_offset] += 1
        U[r1 * m + 1][rank_offset] += 1
        V[b0 * p + c0][rank_offset] += 1
        V[b1 * p + c1][rank_offset] += 1
        W[c0 * n + r0][rank_offset] += 1
        W[c1 * n + r1][rank_offset] += 1
        
        # M2: (A[r0,1] + A[r1,1]) * B[b0,c0]
        # U: A[r0,1] (idx r0*m+1), A[r1,1] (idx r1*m+1)
        # V: B[b0,c0] (idx b0*p+c0)
        # W: C[c0, r1] (idx c0*n+r1)
        U[r0 * m + 1][rank_offset + 1] += 1
        U[r1 * m + 1][rank_offset + 1] += 1
        V[b0 * p + c0][rank_offset + 1] += 1
        W[c0 * n + r1][rank_offset + 1] += 1
        
        # M3: A[r0,0] * (B[b0,c1] - B[b1,c1])
        # U: A[r0,0] (idx r0*m+0)
        # V: B[b0,c1] (idx b0*p+c1), B[b1,c1] (idx b1*p+c1)
        # W: C[c1, r0] (idx c1*n+r0)
        U[r0 * m + 0][rank_offset + 2] += 1
        V[b0 * p + c1][rank_offset + 2] += 1
        V[b1 * p + c1][rank_offset + 2] -= 1
        W[c1 * n + r0][rank_offset + 2] += 1
        
        # M4: A[r1,1] * (B[b0,c0] + B[b0,c1])
        # U: A[r1,1] (idx r1*m+1)
        # V: B[b0,c0] (idx b0*p+c0), B[b0,c1] (idx b0*p+c1)
        # W: C[c0, r0] (idx c0*n+r0), C[c0, r1] (idx c0*n+r1)
        U[r1 * m + 1][rank_offset + 3] += 1
        V[b0 * p + c0][rank_offset + 3] += 1
        V[b0 * p + c1][rank_offset + 3] += 1
        W[c0 * n + r0][rank_offset + 3] += 1
        W[c0 * n + r1][rank_offset + 3] += 1
        
        # M5: (A[r0,0] + A[r0,1]) * B[b1,c1]
        # U: A[r0,0] (idx r0*m+0), A[r0,1] (idx r0*m+1)
        # V: B[b1,c1] (idx b1*p+c1)
        # W: C[c0, r0] (idx c0*n+r0) coeff -1, C[c1, r0] (idx c1*n+r0) coeff 1
        U[r0 * m + 0][rank_offset + 4] += 1
        U[r0 * m + 1][rank_offset + 4] += 1
        V[b1 * p + c1][rank_offset + 4] += 1
        W[c0 * n + r0][rank_offset + 4] -= 1
        W[c1 * n + r0][rank_offset + 4] += 1
        
        # M6: (A[r1,0] - A[r0,0]) * (B[b0,c0] + B[b0,c1])
        # U: A[r1,0] (idx r1*m+0), A[r0,0] (idx r0*m+0)
        # V: B[b0,c0] (idx b0*p+c0), B[b0,c1] (idx b0*p+c1)
        # W: C[c1, r1] (idx c1*n+r1)
        U[r1 * m + 0][rank_offset + 5] += 1
        U[r0 * m + 0][rank_offset + 5] -= 1
        V[b0 * p + c0][rank_offset + 5] += 1
        V[b0 * p + c1][rank_offset + 5] += 1
        W[c1 * n + r1][rank_offset + 5] += 1
        
        # M7: (A[r0,1] - A[r1,1]) * (B[b1,c0] + B[b1,c1])
        # U: A[r0,1] (idx r0*m+1), A[r1,1] (idx r1*m+1)
        # V: B[b1,c0] (idx b1*p+c0), B[b1,c1] (idx b1*p+c1)
        # W: C[c0, r0] (idx c0*n+r0) coeff -1
        U[r0 * m + 1][rank_offset + 6] += 1
        U[r1 * m + 1][rank_offset + 6] -= 1
        V[b1 * p + c0][rank_offset + 6] += 1
        V[b1 * p + c1][rank_offset + 6] += 1
        W[c0 * n + r0][rank_offset + 6] -= 1

    def add_schoolbook(rank_offset, a_col, b_row, b_cols):
        # a_col: column index in A (0, 1, 2)
        # b_row: row index in B (0, 1, 2)
        # b_cols: list of 4 column indices in B
        for k in b_cols:
            # C[k, 0] = A[0, a_col] * B[b_row, k]
            U[0 * m + a_col][rank_offset] += 1
            V[b_row * p + k][rank_offset] += 1
            W[k * n + 0][rank_offset] += 1
            rank_offset += 1
            
            # C[k, 1] = A[1, a_col] * B[b_row, k]
            U[1 * m + a_col][rank_offset] += 1
            V[b_row * p + k][rank_offset] += 1
            W[k * n + 1][rank_offset] += 1
            rank_offset += 1
        return rank_offset

    # Block 1: A[0:2, 0:2] * B[0:2, 0:4]
    # Split B[0:2, 0:4] into B[0:2, 0:2] and B[0:2, 2:4]
    add_strassen(0, [0, 1], [0, 1], [0, 1])
    add_strassen(7, [0, 1], [0, 1], [2, 3])
    
    # Block 2: A[0:2, 2:3] * B[2:3, 0:4]
    # A col 2, B row 2, B cols 0,1,2,3
    next_rank = add_schoolbook(14, 2, 2, [0, 1, 2, 3])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:8]
    # Split B[0:2, 4:8] into B[0:2, 4:6] and B[0:2, 6:8]
    add_strassen(next_rank, [0, 1], [0, 1], [4, 5])
    add_strassen(next_rank + 7, [0, 1], [0, 1], [6, 7])
    
    # Block 4: A[0:2, 2:3] * B[2:3, 4:8]
    # A col 2, B row 2, B cols 4,5,6,7
    next_rank = add_schoolbook(next_rank + 14, 2, 2, [4, 5, 6, 7])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
