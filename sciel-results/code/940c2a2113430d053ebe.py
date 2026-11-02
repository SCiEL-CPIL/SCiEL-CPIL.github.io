"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_ro, a_co, b_ro, b_co, c_co, c_ro):
        nonlocal col
        u00 = (a_ro + 0) * m + (a_co + 0)
        u01 = (a_ro + 0) * m + (a_co + 1)
        u10 = (a_ro + 1) * m + (a_co + 0)
        u11 = (a_ro + 1) * m + (a_co + 1)
        v00 = (b_ro + 0) * p + (b_co + 0)
        v01 = (b_ro + 0) * p + (b_co + 1)
        v10 = (b_ro + 1) * p + (b_co + 0)
        v11 = (b_ro + 1) * p + (b_co + 1)
        w00 = (c_co + 0) * n + (c_ro + 0)
        w01 = (c_co + 1) * n + (c_ro + 0)
        w10 = (c_co + 0) * n + (c_ro + 1)
        w11 = (c_co + 1) * n + (c_ro + 1)
        
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1
        col += 1
        
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_matvec(a_ro, a_co, b_ro, b_co, c_co, c_ro):
        nonlocal col
        u00 = (a_ro + 0) * m + (a_co + 0)
        u01 = (a_ro + 0) * m + (a_co + 1)
        u10 = (a_ro + 1) * m + (a_co + 0)
        u11 = (a_ro + 1) * m + (a_co + 1)
        v0 = (b_ro + 0) * p + b_co
        v1 = (b_ro + 1) * p + b_co
        w0 = c_co * n + (c_ro + 0)
        w1 = c_co * n + (c_ro + 1)
        
        U[u00][col] += 1; V[v0][col] += 1; W[w0][col] += 1; col += 1
        U[u01][col] += 1; V[v1][col] += 1; W[w0][col] += 1; col += 1
        U[u10][col] += 1; V[v0][col] += 1; W[w1][col] += 1; col += 1
        U[u11][col] += 1; V[v1][col] += 1; W[w1][col] += 1; col += 1
    
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 2, 0)
    add_matvec(0, 0, 0, 4, 4, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 2, 2, 2, 2, 0)
    add_matvec(0, 2, 2, 4, 4, 0)
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
