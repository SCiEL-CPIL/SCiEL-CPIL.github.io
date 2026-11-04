"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][rank_offset] += val
        V[v_idx][rank_offset] += val
        W[w_idx][rank_offset] += val
        
    # We will fill U, V, W for each of the 3 blocks of 3 columns in B
    for b in range(3):
        k0 = 3 * b
        k1 = 3 * b + 1
        k2 = 3 * b + 2
        rank_offset = 17 * b
        
        # Strassen 2x2x2 part (A cols 0,1; B rows 0,1; C cols k0, k1)
        # M1: (A00+A11)(B0k0+B1k1) -> C0k0, C1k1
        U[0][rank_offset] += 1
        U[3][rank_offset] += 1  # A11 is index 1*3+1 = 4? No.
        
        # Indexing:
        # U rows: i*m + j. n=2, m=3.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=0, j=2 -> 2
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # i=1, j=2 -> 5
        
        # V rows: j*p + k. m=3, p=9.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # ...
        # j=1, k=0 -> 9
        # j=2, k=0 -> 18
        
        # W rows: k*n + i. p=9, n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # M1: (A00+A11)(B0k0+B1k1)
        # A00: U[0], A11: U[4]
        # B0k0: V[0*9+k0], B1k1: V[1*9+k1]
        # C0k0: W[k0*2+0], C1k1: W[k1*2+1]
        
        U[0][rank_offset] += 1
        U[4][rank_offset] += 1
        V[0*9+k0][rank_offset] += 1
        V[1*9+k1][rank_offset] += 1
        W[k0*2+0][rank_offset] += 1
        W[k1*2+1][rank_offset] += 1
        
        # M2: (A10+A11)B0k0 -> C1k0
        # A10: U[3], A11: U[4]
        # B0k0: V[0*9+k0]
        # C1k0: W[k0*2+1]
        U[3][rank_offset] += 1
        U[4][rank_offset] += 1
        V[0*9+k0][rank_offset] += 1
        W[k0*2+1][rank_offset] += 1
        
        # M3: A00(B0k1-B1k1) -> C0k1
        # A00: U[0]
        # B0k1: V[0*9+k1], B1k1: V[1*9+k1]
        # C0k1: W[k1*2+0]
        U[0][rank_offset] += 1
        V[0*9+k1][rank_offset] += 1
        V[1*9+k1][rank_offset] -= 1
        W[k1*2+0][rank_offset] += 1
        
        # M4: A11(B0k0-B1k0) -> C1k0
        # A11: U[4]
        # B0k0: V[0*9+k0], B1k0: V[1*9+k0]
        # C1k0: W[k0*2+1]
        U[4][rank_offset] += 1
        V[0*9+k0][rank_offset] += 1
        V[1*9+k0][rank_offset] -= 1
        W[k0*2+1][rank_offset] += 1
        
        # M5: (A00+A01)B1k1 -> C0k1 (-1), C1k1 (+1)
        # A00: U[0], A01: U[1]
        # B1k1: V[1*9+k1]
        # C0k1: W[k1*2+0], C1k1: W[k1*2+1]
        U[0][rank_offset] += 1
        U[1][rank_offset] += 1
        V[1*9+k1][rank_offset] += 1
        W[k1*2+0][rank_offset] -= 1
        W[k1*2+1][rank_offset] += 1
        
        # M6: (A10-A00)(B0k0+B0k1) -> C1k1
        # A10: U[3], A00: U[0]
        # B0k0: V[0*9+k0], B0k1: V[0*9+k1]
        # C1k1: W[k1*2+1]
        U[3][rank_offset] += 1
        U[0][rank_offset] -= 1
        V[0*9+k0][rank_offset] += 1
        V[0*9+k1][rank_offset] += 1
        W[k1*2+1][rank_offset] += 1
        
        # M7: (A01-A11)(B1k0+B1k1) -> C0k0
        # A01: U[1], A11: U[4]
        # B1k0: V[1*9+k0], B1k1: V[1*9+k1]
        # C0k0: W[k0*2+0]
        U[1][rank_offset] += 1
        U[4][rank_offset] -= 1
        V[1*9+k0][rank_offset] += 1
        V[1*9+k1][rank_offset] += 1
        W[k0*2+0][rank_offset] += 1
        
        # Outer product part (A col 2; B row 2; C cols k0, k1)
        # M8: A02 B2k0 -> C0k0
        U[2][rank_offset] += 1
        V[2*9+k0][rank_offset] += 1
        W[k0*2+0][rank_offset] += 1
        
        # M9: A02 B2k1 -> C0k1
        U[2][rank_offset] += 1
        V[2*9+k1][rank_offset] += 1
        W[k1*2+0][rank_offset] += 1
        
        # M10: A12 B2k0 -> C1k0
        U[5][rank_offset] += 1
        V[2*9+k0][rank_offset] += 1
        W[k0*2+1][rank_offset] += 1
        
        # M11: A12 B2k1 -> C1k1
        U[5][rank_offset] += 1
        V[2*9+k1][rank_offset] += 1
        W[k1*2+1][rank_offset] += 1
        
        # Last column part (A cols 0,1,2; B col k2; C col k2)
        # M12: A00 B0k2 -> C0k2
        U[0][rank_offset] += 1
        V[0*9+k2][rank_offset] += 1
        W[k2*2+0][rank_offset] += 1
        
        # M13: A01 B1k2 -> C0k2
        U[1][rank_offset] += 1
        V[1*9+k2][rank_offset] += 1
        W[k2*2+0][rank_offset] += 1
        
        # M14: A02 B2k2 -> C0k2
        U[2][rank_offset] += 1
        V[2*9+k2][rank_offset] += 1
        W[k2*2+0][rank_offset] += 1
        
        # M15: A10 B0k2 -> C1k2
        U[3][rank_offset] += 1
        V[0*9+k2][rank_offset] += 1
        W[k2*2+1][rank_offset] += 1
        
        # M16: A11 B1k2 -> C1k2
        U[4][rank_offset] += 1
        V[1*9+k2][rank_offset] += 1
        W[k2*2+1][rank_offset] += 1
        
        # M17: A12 B2k2 -> C1k2
        U[5][rank_offset] += 1
        V[2*9+k2][rank_offset] += 1
        W[k2*2+1][rank_offset] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
