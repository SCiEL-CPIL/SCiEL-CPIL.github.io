import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 component
    # A is 2x2, B is 2x2
    # U offsets: i*m + j
    # V offsets: j*p + k
    # W offsets: k*n + i
    def add_strassen(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[u_base_i + 0][col] += 1
        U[u_base_i + 1][col] += 1
        V[v_base_j + 0][col] += 1
        V[v_base_j + 1][col] += 1
        W[w_base_k + 0][col] += 1 # C00
        W[w_base_k + 1][col] += 1 # C11 (Wait, C11 is k=1, i=1. W[1*n+1]. If w_base_k is k offset, w_base_i is i offset)
        # Correction: W index is k*n + i.
        # C00: k=0, i=0. W[0*n+0].
        # C11: k=1, i=1. W[1*n+1].
        # My w_base_k, w_base_i are offsets for k and i.
        # So W[(w_base_k+0)*n + (w_base_i+0)]
        # And W[(w_base_k+1)*n + (w_base_i+1)]
        
        # Let's rewrite carefully.
        # M1
        U[u_base_i + 0][col] = 1
        U[u_base_i + 1][col] = 1
        V[v_base_j + 0][col] = 1
        V[v_base_j + 1][col] = 1
        W[(w_base_k + 0) * n + (w_base_i + 0)][col] = 1
        W[(w_base_k + 1) * n + (w_base_i + 1)][col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[u_base_i + 1][col] = 1 # A10 is i=1, j=0
        U[u_base_i + 1 + 1][col] = 1 # A11 is i=1, j=1? No.
        # U index is i*m + j.
        # A10: i=1, j=0. Index 1*m + 0.
        # A11: i=1, j=1. Index 1*m + 1.
        # u_base_i is offset for i. u_base_j is offset for j?
        # No, u_base_i should be offset for the row i.
        # Let's pass explicit indices.
        
    # Redefine helper with explicit indices
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i_off: offset for i in A and C (0 or 1)
        # j_off: offset for j in A and B (0 or 1)
        # k_off: offset for k in B and C (0..4)
        
        # M1 = (A[i,j]+A[i+1,j+1])(B[j,k]+B[j+1,k+1])
        # A indices: (i, j) and (i+1, j+1)
        # B indices: (j, k) and (j+1, k+1)
        # C contributions:
        # C[i,k] += M1
        # C[i+1,k+1] += M1
        
        # We need to sum U and V entries.
        # U[i*m+j] = 1, U[(i+1)*m+(j+1)] = 1
        # V[j*p+k] = 1, V[(j+1)*p+(k+1)] = 1
        # W[k*n+i] = 1, W[(k+1)*n+(i+1)] = 1
        
        U[i * m + j][col] = 1
        U[(i + 1) * m + (j + 1)][col] = 1
        V[j * p + k][col] = 1
        V[(j + 1) * p + (k + 1)][col] = 1
        W[k * n + i][col] = 1
        W[(k + 1) * n + (i + 1)][col] = 1
        col += 1
        
        # M2 = (A[i+1,j]+A[i+1,j+1]) B[j,k]
        # A: (i+1, j), (i+1, j+1)
        # B: (j, k)
        # C:
        # C[i+1,k] += M2
        # C[i+1,k+1] += M2
        
        U[(i + 1) * m + j][col] = 1
        U[(i + 1) * m + (j + 1)][col] = 1
        V[j * p + k][col] = 1
        W[k * n + (i + 1)][col] = 1
        W[(k + 1) * n + (i + 1)][col] = 1
        col += 1
        
        # M3 = A[i,j] (B[j,k+1]-B[j+1,k+1])
        # A: (i, j)
        # B: (j, k+1), (j+1, k+1)
        # C:
        # C[i+1,k+1] -= M3
        
        U[i * m + j][col] = 1
        V[j * p + (k + 1)][col] = 1
        V[(j + 1) * p + (k + 1)][col] = -1
        W[(k + 1) * n + (i + 1)][col] = -1
        col += 1
        
        # M4 = A[i+1,j+1] (B[j+1,k]-B[j,k])
        # A: (i+1, j+1)
        # B: (j+1, k), (j, k)
        # C:
        # C[i,k] -= M4
        # C[i+1,k] += M4
        
        U[(i + 1) * m + (j + 1)][col] = 1
        V[(j + 1) * p + k][col] = 1
        V[j * p + k][col] = -1
        W[k * n + i][col] = -1
        W[k * n + (i + 1)][col] = 1
        col += 1
        
        # M5 = (A[i,j]+A[i,j+1]) B[j+1,k+1]
        # A: (i, j), (i, j+1)
        # B: (j+1, k+1)
        # C:
        # C[i,k+1] += M5
        
        U[i * m + j][col] = 1
        U[i * m + (j + 1)][col] = 1
        V[(j + 1) * p + (k + 1)][col] = 1
        W[(k + 1) * n + i][col] = 1
        col += 1
        
        # M6 = (A[i+1,j]-A[i,j]) (B[j,k]+B[j,k+1])
        # A: (i+1, j), (i, j)
        # B: (j, k), (j, k+1)
        # C:
        # C[i+1,k] += M6
        # C[i+1,k+1] += M6
        
        U[(i + 1) * m + j][col] = 1
        U[i * m + j][col] = -1
        V[j * p + k][col] = 1
        V[j * p + (k + 1)][col] = 1
        W[k * n + (i + 1)][col] = 1
        W[(k + 1) * n + (i + 1)][col] = 1
        col += 1
        
        # M7 = (A[i,j+1]-A[i+1,j+1]) (B[j+1,k]+B[j+1,k+1])
        # A: (i, j+1), (i+1, j+1)
        # B: (j+1, k), (j+1, k+1)
        # C:
        # C[i,k] += M7
        # C[i,k+1] += M7
        
        U[i * m + (j + 1)][col] = 1
        U[(i + 1) * m + (j + 1)][col] = -1
        V[(j + 1) * p + k][col] = 1
        V[(j + 1) * p + (k + 1)][col] = 1
        W[k * n + i][col] = 1
        W[(k + 1) * n + i][col] = 1
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, 0:2] (i=0, j=0, k=0)
    add_strassen_block(0, 0, 0)
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4] (i=0, j=0, k=2)
    add_strassen_block(0, 0, 2)
    
    # 3. A[0:2, 0:2] * B[0:2, 4:5] (i=0, j=0, k=4)
    # This is 2x2 * 2x1.
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 1: A[0,0] B[0,4] -> C[0,4]
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] B[1,4] -> C[0,4]
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] B[0,4] -> C[1,4]
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] B[1,4] -> C[1,4]
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # 4. A[0:2, 2:3] * B[2:3, 0:5] (Tail)
    # A[i,2] B[2,k] -> C[i,k]
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
