def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2 matrix multiplication
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, val) for A[i,j]
    # V_entries: list of (j, k, val) for B[j,k]
    # W_entries: list of (i, k, val) for C[i,k]
    
    strassen_terms = [
        # P1: (a+d)(e+h)
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2: (c+d)e
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3: a(f-h)
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4: d(g-e)
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5: (a+b)h
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6: (c-a)(e+f)
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7: (b-d)(g+h)
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # Process Set 1: j in {0,1}, k_blocks in {0,1,2,3}
    for k_block in range(4):
        for term in strassen_terms:
            # Map local indices to global
            # A local (i, j_local) -> global (i, j_global)
            # j_local 0->0, 1->1
            # B local (j_local, k_local) -> global (j_global, k_global)
            # j_local 0->0, 1->1
            # k_local 0->2*k_block, 1->2*k_block+1
            # C local (i, k_local) -> global (i, k_global)
            
            for (i, jl, val) in term["U"]:
                jg = jl
                U[i*4 + jg][col] += val
            
            for (jl, kl, val) in term["V"]:
                jg = jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            
            col += 1
            
    # Process Set 2: j in {2,3}, k_blocks in {0,1,2,3}
    for k_block in range(4):
        for term in strassen_terms:
            # Map local indices to global
            # A local (i, j_local) -> global (i, j_global)
            # j_local 0->2, 1->3
            # B local (j_local, k_local) -> global (j_global, k_global)
            # j_local 0->2, 1->3
            # k_local 0->2*k_block, 1->2*k_block+1
            
            for (i, jl, val) in term["U"]:
                jg = 2 + jl
                U[i*4 + jg][col] += val
            
            for (jl, kl, val) in term["V"]:
                jg = 2 + jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
