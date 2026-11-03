import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[u_idx(i, j)] = 1
                    v[v_idx(j, k)] = 1
                    w[w_idx(k, i)] = 1
                    add_term(u, v, w)

    def add_strassen_2x2x2(i0, j0, k0):
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1
        w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1
        v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
        w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
        w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1
        add_term(u, v, w)

    # Split 5 into 1+2+2
    # i-blocks: [0], [1,2], [3,4]
    # j-blocks: [0], [1,2], [3,4]
    # k-blocks: [0], [1,2], [3,4]
    
    blocks = [[0], [1, 2], [3, 4]]
    
    for bi, ib in enumerate(blocks):
        for bj, jb in enumerate(blocks):
            for bk, kb in enumerate(blocks):
                i_start, i_end = min(ib), max(ib)
                j_start, j_end = min(jb), max(jb)
                k_start, k_end = min(kb), max(kb)
                
                i_size = len(ib)
                j_size = len(jb)
                k_size = len(kb)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassen_2x2x2(i_start, j_start, k_start)
                else:
                    add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end)
    
    rank = len(U)
    
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if U[r][i] != 0:
                U_mat[i][r] = U[r][i]
        for j in range(m*p):
            if V[r][j] != 0:
                V_mat[j][r] = V[r][j]
        for k in range(p*n):
            if W[r][k] != 0:
                W_mat[k][r] = W[r][k]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
