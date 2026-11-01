"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    # Split m=4 into two 2-column blocks, p=6 into three 2-column blocks
    # Each sub-block is a 2x2x2 matrix multiplication, decomposed with Strassen (rank 7)
    # Total rank = (m/2) * (p/2) * 7 = 2 * 3 * 7 = 42

    # Strassen decomposition for 2x2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    #
    # Strassen uses 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, each multiplication Mk corresponds to a rank-1 term.
    # The rank-1 term for Mk has:
    #   U values (on A entries), V values (on B entries), W values (on C entries)
    #
    # Let's define the Strassen rank-1 factors for 2x2x2:
    # For sub-block with A indices (i, j) where i in {0,1}, j in {0,1}
    # and B indices (j, k) where k in {0,1}, C indices (i, k)
    #
    # M1: U = [1, 0, 0, 1] (a, b, c, d), V = [1, 0, 0, 1] (e, f, g, h), W = [1, 0, 0, 1] (C00, C01, C10, C11)
    # M2: U = [0, 0, 1, 1], V = [1, 0, 0, 0], W = [0, 0, 1, 0]
    # M3: U = [1, 0, 0, 0], V = [0, 1, 0, -1], W = [0, 1, 0, 0]
    # M4: U = [0, 0, 0, 1], V = [0, 0, 1, -1], W = [-1, 0, 0, 0]
    # M5: U = [1, 1, 0, 0], V = [0, 0, 0, 1], W = [1, 1, 0, 0]
    # M6: U = [-1, 0, 1, 0], V = [1, 1, 0, 0], W = [0, 0, 0, 1]
    # M7: U = [0, 1, 0, -1], V = [0, 0, 1, 1], W = [-1, 0, 0, 0]
    #
    # Wait, let me re-derive carefully.
    # 
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # So for each Mk, the W vector (on C entries C00, C01, C10, C11) is:
    # M1: W = [1, 0, 0, 1]
    # M2: W = [0, 0, 1, -1]
    # M3: W = [0, 1, 0, 1]
    # M4: W = [-1, 0, 1, 0]
    # M5: W = [1, 1, 0, 0]
    # M6: W = [0, 0, 0, 1]
    # M7: W = [-1, 0, 0, 0]
    #
    # U vectors (on A entries a=A00, b=A01, c=A10, d=A11):
    # M1: U = [1, 0, 0, 1]
    # M2: U = [0, 0, 1, 1]
    # M3: U = [1, 0, 0, 0]
    # M4: U = [0, 0, 0, 1]
    # M5: U = [1, 1, 0, 0]
    # M6: U = [-1, 0, 1, 0]
    # M7: U = [0, 1, 0, -1]
    #
    # V vectors (on B entries e=B00, f=B01, g=B10, h=B11):
    # M1: V = [1, 0, 0, 1]
    # M2: V = [1, 0, 0, 0]
    # M3: V = [0, 1, 0, -1]
    # M4: V = [0, 0, 1, -1]
    # M5: V = [0, 0, 0, 1]
    # M6: V = [1, 1, 0, 0]
    # M7: V = [0, 0, 1, 1]

    # Strassen factors for 2x2x2 (U on A entries [a,b,c,d], V on B entries [e,f,g,h], W on C entries [C00,C01,C10,C11])
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [-1, 0, 1, 0],  # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 0, 1],   # M6
        [-1, 0, 0, 0],  # M7
    ]

    # Build the full decomposition
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)
    
    total_rank = (m // 2) * (p // 2) * 7
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for j_block in range(m // 2):  # j_block = 0, 1
        for k_block in range(p // 2):  # k_block = 0, 1, 2
            j_base = 2 * j_block
            k_base = 2 * k_block
            for s in range(7):  # Strassen factor index
                # U indices: for i in {0,1}, j in {j_base, j_base+1}
                # U row index = i * m + j
                for i in range(n):
                    for dj in range(2):
                        j = j_base + dj
                        u_idx = i * m + j
                        # strassen_U[s] has 4 entries: [a, b, c, d] = [A00, A01, A10, A11]
                        # where 00 means i=0,dj=0; 01 means i=0,dj=1; 10 means i=1,dj=0; 11 means i=1,dj=1
                        u_val = strassen_U[s][i * 2 + dj]
                        U[u_idx][col] += u_val
                # V indices: for j in {j_base, j_base+1}, k in {k_base, k_base+1}
                # V row index = j * p + k
                for dj in range(2):
                    j = j_base + dj
                    for dk in range(2):
                        k = k_base + dk
                        v_idx = j * p + k
                        # strassen_V[s] has 4 entries: [e, f, g, h] = [B00, B01, B10, B11]
                        # where 00 means dj=0,dk=0; 01 means dj=0,dk=1; 10 means dj=1,dk=0; 11 means dj=1,dk=1
                        v_val = strassen_V[s][dj * 2 + dk]
                        V[v_idx][col] += v_val
                # W indices: for k in {k_base, k_base+1}, i in {0,1}
                # W row index = k * n + i
                for dk in range(2):
                    k = k_base + dk
                    for i in range(n):
                        w_idx = k * n + i
                        # strassen_W[s] has 4 entries: [C00, C01, C10, C11]
                        # where 00 means dk=0,i=0; 01 means dk=1,i=0; 10 means dk=0,i=1; 11 means dk=1,i=1
                        w_val = strassen_W[s][dk * 2 + i]
                        W[w_idx][col] += w_val
                col += 1

    # Verify the decomposition
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                if val != 1:
                    # Fallback to schoolbook
                    schoolbook_rank = n * m * p
                    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
                    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
                    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii*m+jj][col] = 1
                                V_fb[jj*p+kk][col] = 1
                                W_fb[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
