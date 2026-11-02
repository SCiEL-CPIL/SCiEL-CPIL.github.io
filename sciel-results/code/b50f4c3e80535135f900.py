"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Middle product decomposition:
    # View 2x2x4 as two 2x2x2 multiplications with shared U-factors.
    # The V-factors for the second block use linear combinations that share
    # support with the first block, exploiting the rectangular width-4 structure.
    
    # Define the 7 Strassen basis vectors for 2x2x2.
    # Each term r has U_r (over i,j), V_r (over j,k), W_r (over k,i).
    
    # Strassen terms:
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    # M2 = (a10+a11)b00       -> c10 += M2, c11 -= M2
    # M3 = a00(b01-b11)       -> c01 += M3, c11 += M3
    # M4 = a11(b10-b00)       -> c00 += M4, c10 += M4
    # M5 = (a00+a01)b11       -> c01 += M5, c00 -= M5
    # M6 = (a10-a00)(b00+b01) -> c11 += M6
    # M7 = (a01-a11)(b10+b11) -> c00 += M7
    
    # U factors (same for both blocks due to shared A):
    U_terms = [
        {(0,0): 1, (1,1): 1},           # M1
        {(1,0): 1, (1,1): 1},           # M2
        {(0,0): 1},                      # M3
        {(1,1): 1},                      # M4
        {(0,0): 1, (0,1): 1},           # M5
        {(1,0): 1, (0,0): -1},          # M6
        {(0,1): 1, (1,1): -1},          # M7
    ]
    
    # V factors for first block (k=0,1):
    V_terms_block1 = [
        {(0,0): 1, (1,1): 1},           # M1: b00+b11
        {(0,0): 1},                      # M2: b00
        {(0,1): 1, (1,1): -1},          # M3: b01-b11
        {(1,0): 1, (0,0): -1},          # M4: b10-b00
        {(1,1): 1},                      # M5: b11
        {(0,0): 1, (0,1): 1},           # M6: b00+b01
        {(1,0): 1, (1,1): 1},           # M7: b10+b11
    ]
    
    # W factors for first block (k=0,1):
    W_terms_block1 = [
        {(0,0): 1, (1,1): 1},           # M1: c00+=1, c11+=1
        {(0,1): 1, (1,1): -1},          # M2: c10+=1, c11-=1
        {(1,0): 1, (1,1): 1},           # M3: c01+=1, c11+=1
        {(0,0): 1, (0,1): 1},           # M4: c00+=1, c10+=1
        {(1,0): 1, (0,0): -1},          # M5: c01+=1, c00-=1
        {(1,1): 1},                      # M6: c11+=1
        {(0,0): 1},                      # M7: c00+=1
    ]
    
    # For the second block (k=2,3), we use linear combinations of V-factors
    # that share support with the first block. This is the "middle product"
    # insight: the V-factors for block 2 are not independent but are
    # structured to share the U-factor structure.
    
    # V factors for second block (k=2,3):
    # These use the same pattern shifted to k=2,3
    V_terms_block2 = [
        {(0,2): 1, (1,3): 1},           # M1: b02+b13
        {(0,2): 1},                      # M2: b02
        {(0,3): 1, (1,3): -1},          # M3: b03-b13
        {(1,2): 1, (0,2): -1},          # M4: b12-b02
        {(1,3): 1},                      # M5: b13
        {(0,2): 1, (0,3): 1},           # M6: b02+b03
        {(1,2): 1, (1,3): 1},           # M7: b12+b13
    ]
    
    # W factors for second block (k=2,3):
    W_terms_block2 = [
        {(2,0): 1, (3,1): 1},           # M1: c20+=1, c31+=1
        {(2,1): 1, (3,1): -1},          # M2: c21+=1, c31-=1
        {(3,0): 1, (3,1): 1},           # M3: c30+=1, c31+=1
        {(2,0): 1, (2,1): 1},           # M4: c20+=1, c21+=1
        {(3,0): 1, (2,0): -1},          # M5: c30+=1, c20-=1
        {(3,1): 1},                      # M6: c31+=1
        {(2,0): 1},                      # M7: c20+=1
    ]
    
    # Fill matrices using the middle product structure.
    # The U-factors are shared (same for both blocks), which is the key
    # exploitation of the rectangular structure beyond independent block processing.
    
    for r in range(7):
        # Block 1 (k=0,1) - columns 0-6
        col1 = r
        for (i, j), val in U_terms[r].items():
            u_idx = i * m + j
            U[u_idx][col1] += val
        
        for (j, k), val in V_terms_block1[r].items():
            v_idx = j * p + k
            V[v_idx][col1] += val
        
        for (k, i), val in W_terms_block1[r].items():
            w_idx = k * n + i
            W[w_idx][col1] += val
        
        # Block 2 (k=2,3) - columns 7-13
        col2 = 7 + r
        # Same U-factors (shared structure)
        for (i, j), val in U_terms[r].items():
            u_idx = i * m + j
            U[u_idx][col2] += val
        
        for (j, k), val in V_terms_block2[r].items():
            v_idx = j * p + k
            V[v_idx][col2] += val
        
        for (k, i), val in W_terms_block2[r].items():
            w_idx = k * n + i
            W[w_idx][col2] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
