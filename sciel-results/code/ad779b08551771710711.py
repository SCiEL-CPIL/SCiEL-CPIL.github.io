"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 6x6 matrix multiplication.
    # View 6x6 matrices as 3x3 blocks of 2x2 matrices.
    # Use Strassen (rank 7) for each 2x2 block multiplication.
    # Total rank = 9 blocks * 7 = 63.
    n, m, p = DIMENSIONS
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # u, w: block row/col of output C (0..2)
    # v: block col of A / block row of B (0..2)
    for u in range(3):
        for w in range(3):
            for v in range(3):
                # Indices for A[u,v] block (2x2)
                # A[2u, 2v], A[2u, 2v+1], A[2u+1, 2v], A[2u+1, 2v+1]
                a_idx = [2*u*m + 2*v, 2*u*m + 2*v + 1, (2*u+1)*m + 2*v, (2*u+1)*m + 2*v + 1]
                # Indices for B[v,w] block (2x2)
                # B[2v, 2w], B[2v, 2w+1], B[2v+1, 2w], B[2v+1, 2w+1]
                b_idx = [2*v*p + 2*w, 2*v*p + 2*w + 1, (2*v+1)*p + 2*w, (2*v+1)*p + 2*w + 1]
                # Indices for C[u,w] block (2x2) in W
                # W[k*n+i] where i is row, k is col
                # C[2u, 2w] -> W[2w*n + 2u]
                # C[2u, 2w+1] -> W[(2w+1)*n + 2u]
                # C[2u+1, 2w] -> W[2w*n + 2u+1]
                # C[2u+1, 2w+1] -> W[(2w+1)*n + 2u+1]
                c_idx_11 = 2*w*n + 2*u
                c_idx_12 = (2*w+1)*n + 2*u
                c_idx_21 = 2*w*n + 2*u + 1
                c_idx_22 = (2*w+1)*n + 2*u + 1
                
                # Strassen terms
                # m1 = (a+d)(e+h)
                U[a_idx[0]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx[0]][col] += 1
                V[b_idx[3]][col] += 1
                W[c_idx_11][col] += 1
                W[c_idx_22][col] += 1
                col += 1
                
                # m2 = (c+d)e
                U[a_idx[2]][col] += 1
                U[a_idx[3]][col] += 1
                V[b_idx[0]][col] += 1
                W[c_idx_21][col] += 1
                W[c_idx_22][col] += -1
                col += 1
                
                # m3 = a(f-h)
                U[a_idx[0]][col] += 1
                V[b_idx[1]][col] += 1
                V[b_idx[3]][col] += -1
                W[c_idx_12][col] += 1
                W[c_idx_22][col] += 1
                col += 1
                
                # m4 = d(g-f)
                U[a_idx[3]][col] += 1
                V[b_idx[2]][col] += 1
                V[b_idx[1]][col] += -1
                W[c_idx_11][col] += 1
                W[c_idx_21][col] += 1
                col += 1
                
                # m5 = (a+b)h
                U[a_idx[0]][col] += 1
                U[a_idx[1]][col] += 1
                V[b_idx[3]][col] += 1
                W[c_idx_11][col] += -1
                W[c_idx_12][col] += 1
                col += 1
                
                # m6 = (c-a)(e+f)
                U[a_idx[2]][col] += 1
                U[a_idx[0]][col] += -1
                V[b_idx[0]][col] += 1
                V[b_idx[1]][col] += 1
                W[c_idx_22][col] += 1
                col += 1
                
                # m7 = (b-d)(g+h)
                U[a_idx[1]][col] += 1
                U[a_idx[3]][col] += -1
                V[b_idx[2]][col] += 1
                V[b_idx[3]][col] += 1
                W[c_idx_11][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
