"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2
    # Each term is a list of (local_i, local_j, local_k, coeff_u, coeff_v, coeff_w)
    # Actually, we define the vectors u, v, w for each term.
    # u is over (i,j), v is over (j,k), w is over (k,i).
    
    # Term 1: M1 = (a00+a11)(b00+b11)
    # u: (0,0):1, (1,1):1
    # v: (0,0):1, (1,1):1
    # w: (0,0):1, (1,1):1  [contributes to c00 and c11]
    # Wait, c00 coeff is 1, c11 coeff is 1.
    # w indices: (k,i).
    # c00 is k=0, i=0. Index 0.
    # c11 is k=1, i=1. Index 3.
    
    terms = []
    
    # Helper to add term
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    # u_entries: list of (i, j, val)
    # v_entries: list of (j, k, val)
    # w_entries: list of (k, i, val)
    
    # 1
    add_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    
    # 2: M2 = (a10+a11)b00
    # u: (1,0):1, (1,1):1
    # v: (0,0):1
    # w: c10 (k=0,i=1) coeff 1, c11 (k=1,i=1) coeff -1
    add_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)]
    )
    
    # 3: M3 = a00(b01-b11)
    # u: (0,0):1
    # v: (0,1):1, (1,1):-1
    # w: c01 (k=1,i=0) coeff 1, c11 (k=1,i=1) coeff 1
    add_term(
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)]
    )
    
    # 4: M4 = a11(b10-b00)
    # u: (1,1):1
    # v: (1,0):1, (0,0):-1
    # w: c00 (k=0,i=0) coeff 1, c10 (k=0,i=1) coeff 1
    add_term(
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)]
    )
    
    # 5: M5 = (a00+a01)b11
    # u: (0,0):1, (0,1):1
    # v: (1,1):1
    # w: c01 (k=1,i=0) coeff 1, c00 (k=0,i=0) coeff -1
    add_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(1,0,1), (0,0,-1)]
    )
    
    # 6: M6 = (a10-a00)(b00+b01)
    # u: (1,0):1, (0,0):-1
    # v: (0,0):1, (0,1):1
    # w: c11 (k=1,i=1) coeff 1
    add_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    )
    
    # 7: M7 = (a01-a11)(b10+b11)
    # u: (0,1):1, (1,1):-1
    # v: (1,0):1, (1,1):1
    # w: c00 (k=0,i=0) coeff 1
    add_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    )
    
    # Fill matrices
    for r in range(7):
        u_entries, v_entries, w_entries = terms[r]
        
        # Block 1 (k=0,1)
        col1 = r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col1] += val
            
        for j, k, val in v_entries:
            v_idx = j * p + k
            V[v_idx][col1] += val
            
        for k, i, val in w_entries:
            w_idx = k * n + i
            W[w_idx][col1] += val
            
        # Block 2 (k=2,3)
        col2 = 7 + r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col2] += val
            
        for j, k, val in v_entries:
            # Shift k by 2
            k_new = k + 2
            v_idx = j * p + k_new
            V[v_idx][col2] += val
            
        for k, i, val in w_entries:
            # Shift k by 2
            k_new = k + 2
            w_idx = k_new * n + i
            W[w_idx][col2] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
