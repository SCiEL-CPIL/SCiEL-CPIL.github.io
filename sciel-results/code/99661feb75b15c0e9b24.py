"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_strassen_block(i_list, j_list, k_list):
        # i_list, j_list, k_list must be of length 2
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        
        # P1 = (a+d)(e+h)
        # A: a=i0,j0; d=i1,j1. B: e=j0,k0; h=j1,k1. C: i0,k0; i1,k1; i0,k1; i1,k0
        # U: (i0,j0)=1, (i1,j1)=1. V: (j0,k0)=1, (j1,k1)=1. W: (k0,i0)=1, (k1,i1)=1, (k1,i0)=-1, (k0,i1)=1
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        W[k1 * n + i0][col] = -1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # P2 = (c+d)e
        # A: c=i1,j0; d=i1,j1. B: e=j0,k0. C: i1,k0
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # P3 = a(f-h)
        # A: a=i0,j0. B: f=j0,k1; h=j1,k1. C: i0,k1
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # P4 = d(g-e)
        # A: d=i1,j1. B: g=j1,k0; e=j0,k0. C: i1,k0
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # P5 = (a+b)h
        # A: a=i0,j0; b=i0,j1. B: h=j1,k1. C: i0,k1; i0,k0
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k1 * n + i0][col] = 1
        W[k0 * n + i0][col] = -1
        col += 1
        
        # P6 = (c-a)(e+f)
        # A: c=i1,j0; a=i0,j0. B: e=j0,k0; f=j0,k1. C: i1,k1; i1,k0
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        W[k0 * n + i1][col] = -1
        col += 1
        
        # P7 = (b-d)(g+h)
        # A: b=i0,j1; d=i1,j1. B: g=j1,k0; h=j1,k1. C: i0,k0; i1,k1
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

    # Split J into two blocks of 2
    J1 = [0, 1]
    J2 = [2, 3]
    
    for J in [J1, J2]:
        # Split I into I1 (2 rows) and I2 (1 row)
        I1 = [0, 1]
        I2 = [2]
        
        # I1 part: 2x2x5
        # Split K into K1 (2), K2 (2), K3 (1)
        K1 = [0, 1]
        K2 = [2, 3]
        K3 = [4]
        
        add_strassen_block(I1, J, K1)
        add_strassen_block(I1, J, K2)
        add_schoolbook_block(I1, J, K3)
        
        # I2 part: 1x2x5
        K_all = [0, 1, 2, 3, 4]
        add_schoolbook_block(I2, J, K_all)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
