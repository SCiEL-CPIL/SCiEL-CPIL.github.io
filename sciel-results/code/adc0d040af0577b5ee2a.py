"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-term decomposition for each 2x2 block
    # U forms: row is index into U (0..3), col is which of 7 forms
    U_forms = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1]   # t=6: a01-a11
    ]
    
    # For each t, list of (v_row_offset, value) where v_row_offset is relative to block start
    # V row index = block_start + offset, where block_start = 2*u
    # V rows: 0..p-1 are B[0,*], p..2p-1 are B[1,*]
    # So B[0, 2u] is row 2u, B[0, 2u+1] is row 2u+1
    # B[1, 2u] is row p+2u, B[1, 2u+1] is row p+2u+1
    V_specs = {
        0: [(2*u, 1) for u in range(p//2)],  # placeholder, will build properly
    }
    
    # Let me just hardcode per t, with offsets relative to 2*u
    # t=0: B[0,2u] + B[1,2u+1] -> rows 2u, p+2u+1
    # t=1: B[0,2u] -> row 2u
    # t=2: B[0,2u+1] - B[1,2u+1] -> rows 2u+1, p+2u+1
    # t=3: B[1,2u] - B[0,2u] -> rows p+2u, 2u
    # t=4: B[1,2u+1] -> row p+2u+1
    # t=5: B[0,2u] + B[0,2u+1] -> rows 2u, 2u+1
    # t=6: B[1,2u] + B[1,2u+1] -> rows p+2u, p+2u+1
    
    # W: W[k*n + i], k=column, i=row
    # C[0,2u] = t0 + t3 - t4 + t6
    # C[0,2u+1] = t2 + t4
    # C[1,2u] = t1 + t3
    # C[1,2u+1] = t0 - t1 + t2 + t5
    # So W[(2u)*n + 0] gets t0,t3,t4(-1),t6
    # W[(2u+1)*n + 0] gets t2,t4
    # W[(2u)*n + 1] gets t1,t3
    # W[(2u+1)*n + 1] gets t0,t1(-1),t2,t5
    
    for u in range(p // 2):
        base = 2 * u
        for t in range(7):
            c = u * 7 + t
            
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            # V assignments
            if t == 0:
                V[base][c] += 1
                V[p + base + 1][c] += 1
            elif t == 1:
                V[base][c] += 1
            elif t == 2:
                V[base + 1][c] += 1
                V[p + base + 1][c] -= 1
            elif t == 3:
                V[p + base][c] += 1
                V[base][c] -= 1
            elif t == 4:
                V[p + base + 1][c] += 1
            elif t == 5:
                V[base][c] += 1
                V[base + 1][c] += 1
            elif t == 6:
                V[p + base][c] += 1
                V[p + base + 1][c] += 1
            
            # W assignments
            if t == 0:
                W[base * n + 0][c] += 1
                W[(base+1) * n + 1][c] += 1
            elif t == 1:
                W[base * n + 1][c] += 1
                W[(base+1) * n + 1][c] -= 1
            elif t == 2:
                W[(base+1) * n + 0][c] += 1
                W[(base+1) * n + 1][c] += 1
            elif t == 3:
                W[base * n + 0][c] += 1
                W[base * n + 1][c] += 1
            elif t == 4:
                W[base * n + 0][c] -= 1
                W[(base+1) * n + 0][c] += 1
            elif t == 5:
                W[(base+1) * n + 1][c] += 1
            elif t == 6:
                W[base * n + 0][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
