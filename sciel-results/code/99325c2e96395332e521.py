"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-42 decomposition for 2x4 times 4x6 using Strassen blocks."""
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # We split m=4 into two halves of size 2 (t=0, 1)
    # We split p=6 into three chunks of size 2 (s=0, 1, 2)
    # For each (t, s), we do a 2x2 by 2x2 multiplication using Strassen (rank 7)
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)] # 12 x 42
    
    col = 0
    for t in range(2): # A half index
        for s in range(3): # B third index
            # Define mapping for local indices to global indices
            # X (A block) local (r, c) -> global A (i, j)
            # i = r, j = 2*t + c
            # Y (B block) local (r, c) -> global B (j, k)
            # j = 2*t + r, k = 2*s + c
            # Z (C block) local (r, c) -> global C (i, k)
            # i = r, k = 2*s + c
            # W index for C(i, k) is k*n + i
            
            # Strassen steps
            # 1: (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
            # 2: (X10+X11)Y00 -> Z10+=1, Z11-=1
            # 3: X00(Y10-Y11) -> Z01+=1, Z11+=1
            # 4: X11(Y01-Y00) -> Z00+=1, Z10+=1
            # 5: (X00+X01)Y11 -> Z00-=1, Z01+=1
            # 6: (X10-X00)(Y00+Y01) -> Z11+=1
            # 7: (X01-X11)(Y10+Y11) -> Z00+=1
            
            steps = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # U_coeffs: dict of (local_i, local_j) -> coeff for A
                # V_coeffs: dict of (local_i, local_j) -> coeff for B
                # W_coeffs: dict of (local_i, local_j) -> coeff for C
                ({(0,0): 1.0, (1,1): 1.0}, {(0,0): 1.0, (1,1): 1.0}, {(0,0): 1.0, (1,1): 1.0}),
                ({(1,0): 1.0, (1,1): 1.0}, {(0,0): 1.0}, {(1,0): 1.0, (1,1): -1.0}),
                ({(0,0): 1.0}, {(1,0): 1.0, (1,1): -1.0}, {(0,1): 1.0, (1,1): 1.0}),
                ({(1,1): 1.0}, {(0,1): 1.0, (0,0): -1.0}, {(0,0): 1.0, (1,0): 1.0}),
                ({(0,0): 1.0, (0,1): 1.0}, {(1,1): 1.0}, {(0,0): -1.0, (0,1): 1.0}),
                ({(1,0): 1.0, (0,0): -1.0}, {(0,0): 1.0, (0,1): 1.0}, {(1,1): 1.0}),
                ({(0,1): 1.0, (1,1): -1.0}, {(1,0): 1.0, (1,1): 1.0}, {(0,0): 1.0}),
            ]
            
            for step in steps:
                u_coeffs, v_coeffs, w_coeffs = step
                
                # Fill U
                for (ri, rj), coeff in u_coeffs.items():
                    gi = ri
                    gj = 2 * t + rj
                    U[gi * m + gj][col] += coeff
                    
                # Fill V
                for (ri, rj), coeff in v_coeffs.items():
                    gi = 2 * t + ri
                    gj = 2 * s + rj
                    V[gi * p + gj][col] += coeff
                    
                # Fill W
                for (ri, rj), coeff in w_coeffs.items():
                    gi = ri
                    gj = 2 * s + rj
                    W[gj * n + gi][col] += coeff
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
