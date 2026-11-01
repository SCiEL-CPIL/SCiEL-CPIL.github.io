"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen rank-7 U matrix (4x7), indexed by (i*m+j)
    # U[i*2+j, r]
    U = [
        [1,  1,  1,  0,  0,  0,  0],  # i=0, j=0
        [0,  0,  0,  1,  1,  0,  0],  # i=0, j=1
        [0, -1,  1,  0,  0, -1,  0],  # i=1, j=0
        [1,  1,  0,  1,  0,  0,  1],  # i=1, j=1
    ]
    
    # V matrix (32x7), indexed by (j*16+k)
    # V[j*16+k, r] = V_base[j, r] (independent of k)
    V_base = [
        [1,  1,  0,  0,  0,  0,  0],  # j=0
        [1,  0,  0,  0,  1,  0,  0],  # j=1
    ]
    V = []
    for j in range(m):
        for k in range(p):
            V.append(V_base[j][:])
    
    # W matrix (32x7), indexed by (k*2+i)
    # W[k*2+i, r] = W_base[i, r] (independent of k)
    W_base = [
        [1,  1,  1,  1,  1,  1,  1],  # i=0
        [1,  1,  1,  1,  1,  1,  1],  # i=1
    ]
    W = []
    for k in range(p):
        for i in range(n):
            W.append(W_base[i][:])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
