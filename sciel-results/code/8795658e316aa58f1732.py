import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 coefficients
    Av = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    Bv = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C01
        [0, 0, 1, 0, 0, -1, -1],# C10
        [0, -1, 1, 1, 1, 0, 0]  # C11
    ]
    
    U = [[0] * 21 for _ in range(4)]
    V = [[0] * 21 for _ in range(12)]
    W = [[0] * 21 for _ in range(12)]
    
    for block in range(3):
        for r in range(7):
            col = block * 7 + r
            for i in range(2):
                for j in range(2):
                    U[i*2+j][col] = Av[r][i*2+j]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][col] = Bv[r][j*2+k]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][col] = C_coeffs[i][k][r]
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check decomposition
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We want T_{i,j,k} = 1 if i,j,k correspond to a valid multiplication term, 0 otherwise.
    # Actually, the tensor is T_{i,j,k} = delta_{i,k} is not right.
    # The tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # The multiplication tensor T has T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In our case, every term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i in {0,1}, j in {0,1}, k in {0,..,5}.
    
    # Let's check if sum_r U_{i,m+j,r} V_{j,p+k,r} W_{k,n+i,r} == 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(21):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if val != 1:
                    # Fall back to schoolbook
                    rank = n * m * p
                    U_f = [[0] * rank for _ in range(n * m)]
                    V_f = [[0] * rank for _ in range(m * p)]
                    W_f = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i_f in range(n):
                        for j_f in range(m):
                            for k_f in range(p):
                                U_f[i_f * m + j_f][col] = 1
                                V_f[j_f * p + k_f][col] = 1
                                W_f[k_f * n + i_f][col] = 1
                                col += 1
                    return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
