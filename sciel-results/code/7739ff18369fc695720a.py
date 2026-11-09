import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_block(j1, j2, k1, k2):
        i1, i2 = 0, 1
        
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        add_term(
            {(i1, j1): 1, (i2, j2): 1},
            {(j1, k1): 1, (j2, k2): 1},
            {(k1, i1): 1, (k2, i2): 1}
        )
        
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        add_term(
            {(i2, j1): 1, (i2, j2): 1},
            {(j1, k1): 1},
            {(k1, i2): 1, (k2, i2): -1}
        )
        
        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        add_term(
            {(i1, j1): 1},
            {(j1, k2): 1, (j2, k2): -1},
            {(k2, i1): 1, (k2, i2): 1}
        )
        
        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        add_term(
            {(i2, j2): 1},
            {(j2, k1): 1, (j1, k1): -1},
            {(k1, i1): 1, (k1, i2): 1}
        )
        
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        add_term(
            {(i1, j1): 1, (i1, j2): 1},
            {(j2, k2): 1},
            {(k1, i1): -1, (k2, i1): 1}
        )
        
        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        add_term(
            {(i2, j1): 1, (i1, j1): -1},
            {(j1, k1): 1, (j1, k2): 1},
            {(k2, i2): 1}
        )
        
        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        add_term(
            {(i1, j2): 1, (i2, j2): -1},
            {(j2, k1): 1, (j2, k2): 1},
            {(k1, i1): 1}
        )

    def add_direct_block(j, k):
        for i in range(n):
            add_term(
                {(i, j): 1},
                {(j, k): 1},
                {(k, i): 1}
            )

    # Partition m=5 into {0,1}, {2,3}, {4}
    # Partition p=5 into {0,1}, {2,3}, {4}
    J_sets = [(0, 1), (2, 3), (4,)]
    K_sets = [(0, 1), (2, 3), (4,)]
    
    for J in J_sets:
        for K in K_sets:
            if len(J) == 2 and len(K) == 2:
                add_strassen_block(J[0], J[1], K[0], K[1])
            else:
                for j in J:
                    for k in K:
                        add_direct_block(j, k)
    
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
