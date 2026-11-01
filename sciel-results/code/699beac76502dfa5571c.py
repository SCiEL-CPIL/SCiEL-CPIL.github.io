"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49: 7 blocks of 2x2, each using Strassen (rank 7)
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 blocks of 2x2
    for b in range(7):
        # Global column offset for this block in B and C
        # B columns: 2*b, 2*b+1
        # C columns: 2*b, 2*b+1
        
        # Strassen terms for 2x2 multiplication
        # Local indices:
        # A: a00, a01, a10, a11
        # B: b00, b01, b10, b11
        # C: c00, c01, c10, c11
        
        # Term 1: (a00 + a11) * (b00 + b11)
        # U: a00 (0,0), a11 (1,1)
        # V: b00 (0, 2b), b11 (1, 2b+1)
        # W: c00 (2b, 0), c11 (2b+1, 1)
        U[0*2+0][col] += 1
        U[1*2+1][col] += 1
        V[0*14+2*b][col] += 1
        V[1*14+(2*b+1)][col] += 1
        W[(2*b)*2+0][col] += 1
        W[(2*b+1)*2+1][col] += 1
        col += 1
        
        # Term 2: (a10 + a11) * b00
        # U: a10 (1,0), a11 (1,1)
        # V: b00 (0, 2b)
        # W: c10 (2b, 1)
        U[1*2+0][col] += 1
        U[1*2+1][col] += 1
        V[0*14+2*b][col] += 1
        W[(2*b)*2+1][col] += 1
        col += 1
        
        # Term 3: a00 * (b01 - b11)
        # U: a00 (0,0)
        # V: b01 (0, 2b+1), b11 (1, 2b+1)
        # W: c01 (2b+1, 0)
        U[0*2+0][col] += 1
        V[0*14+(2*b+1)][col] += 1
        V[1*14+(2*b+1)][col] -= 1
        W[(2*b+1)*2+0][col] += 1
        col += 1
        
        # Term 4: a11 * (b10 - b00)
        # U: a11 (1,1)
        # V: b10 (1, 2b), b00 (0, 2b)
        # W: c00 (2b, 0), c10 (2b, 1)
        U[1*2+1][col] += 1
        V[1*14+2*b][col] += 1
        V[0*14+2*b][col] -= 1
        W[(2*b)*2+0][col] += 1
        W[(2*b)*2+1][col] += 1
        col += 1
        
        # Term 5: (a00 + a01) * b11
        # U: a00 (0,0), a01 (0,1)
        # V: b11 (1, 2b+1)
        # W: c00 (2b, 0), c01 (2b+1, 0)
        U[0*2+0][col] += 1
        U[0*2+1][col] += 1
        V[1*14+(2*b+1)][col] += 1
        W[(2*b)*2+0][col] -= 1 # c00 gets -M5
        W[(2*b+1)*2+0][col] += 1 # c01 gets +M5
        col += 1
        
        # Term 6: (a10 - a00) * (b00 + b01)
        # U: a10 (1,0), a00 (0,0)
        # V: b00 (0, 2b), b01 (0, 2b+1)
        # W: c11 (2b+1, 1)
        U[1*2+0][col] += 1
        U[0*2+0][col] -= 1
        V[0*14+2*b][col] += 1
        V[0*14+(2*b+1)][col] += 1
        W[(2*b+1)*2+1][col] += 1
        col += 1
        
        # Term 7: (a01 - a11) * (b10 + b11)
        # U: a01 (0,1), a11 (1,1)
        # V: b10 (1, 2b), b11 (1, 2b+1)
        # W: c00 (2b, 0), c11 (2b+1, 1)
        U[0*2+1][col] += 1
        U[1*2+1][col] -= 1
        V[1*14+2*b][col] += 1
        V[1*14+(2*b+1)][col] += 1
        W[(2*b)*2+0][col] += 1
        W[(2*b+1)*2+1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
