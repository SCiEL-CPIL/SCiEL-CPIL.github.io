"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # U[i*m+j, c] for A[i,j]
    # V[j*p+k, c] for B[j,k]
    # W[k*n+i, c] for C[k,i]
    
    n, m, p = 3, 3, 7
    
    U = [[] for _ in range(n*m)]
    V = [[] for _ in range(m*p)]
    W = [[] for _ in range(p*n)]
    
    rank = 0
    
    def add_rank1_term(u_entries, v_entries, w_entries):
        nonlocal rank
        col = rank
        for idx, val in u_entries:
            U[idx].append(val)
        for idx, val in v_entries:
            V[idx].append(val)
        for idx, val in w_entries:
            W[idx].append(val)
        for i in range(n*m):
            if len(U[i]) < col + 1:
                U[i].append(0)
        for i in range(m*p):
            if len(V[i]) < col + 1:
                V[i].append(0)
        for i in range(p*n):
            if len(W[i]) < col + 1:
                W[i].append(0)
        rank += 1
    
    def strassen_2x2x2(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        a00 = (a_i0 + 0) * m + (a_j0 + 0)
        a01 = (a_i0 + 0) * m + (a_j0 + 1)
        a10 = (a_i0 + 1) * m + (a_j0 + 0)
        a11 = (a_i0 + 1) * m + (a_j0 + 1)
        
        b00 = (b_j0 + 0) * p + (b_k0 + 0)
        b01 = (b_j0 + 0) * p + (b_k0 + 1)
        b10 = (b_j0 + 1) * p + (b_k0 + 0)
        b11 = (b_j0 + 1) * p + (b_k0 + 1)
        
        c00 = (c_k0 + 0) * n + (c_i0 + 0)
        c01 = (c_k0 + 0) * n + (c_i0 + 1)
        c10 = (c_k0 + 1) * n + (c_i0 + 0)
        c11 = (c_k0 + 1) * n + (c_i0 + 1)
        
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        add_rank1_term(
            [(a00, 1), (a11, 1)],
            [(b00, 1), (b11, 1)],
            [(c00, 1), (c11, 1)]
        )
        
        # M2 = (A10+A11)B00 -> C10+=1
        add_rank1_term(
            [(a10, 1), (a11, 1)],
            [(b00, 1)],
            [(c10, 1)]
        )
        
        # M3 = A00(B01-B11) -> C01+=1
        add_rank1_term(
            [(a00, 1)],
            [(b01, 1), (b11, -1)],
            [(c01, 1)]
        )
        
        # M4 = A11(B10-B00) -> C10+=1
        add_rank1_term(
            [(a11, 1)],
            [(b10, 1), (b00, -1)],
            [(c10, 1)]
        )
        
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        add_rank1_term(
            [(a00, 1), (a01, 1)],
            [(b11, 1)],
            [(c00, -1), (c01, 1)]
        )
        
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        add_rank1_term(
            [(a10, 1), (a00, -1)],
            [(b00, 1), (b01, 1)],
            [(c11, 1)]
        )
        
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        add_rank1_term(
            [(a01, 1), (a11, -1)],
            [(b10, 1), (b11, 1)],
            [(c00, 1)]
        )
    
    # Build decomposition
    # A11 (2x2) * B1 (2x7): split B1 into three 2x2 and one 2x1
    # A11 at rows 0-1, cols 0-1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)   # B cols 0-1
    strassen_2x2x2(0, 0, 0, 2, 2, 0)   # B cols 2-3
    strassen_2x2x2(0, 0, 0, 4, 4, 0)   # B cols 4-5
    # B col 6: 2x2 * 2x1
    for di in range(2):
        for dj in range(2):
            u_idx = di * m + dj
            v_idx = dj * p + 6
            w_idx = 6 * n + di
            add_rank1_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # A12 (2x1) * B2 (1x7): A rows 0-1 col 2, B row 2
    for di in range(2):
        for dk in range(7):
            u_idx = di * m + 2
            v_idx = 2 * p + dk
            w_idx = dk * n + di
            add_rank1_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # A21 (1x2) * B1 (2x7): A row 2 cols 0-1, B rows 0-1
    for dj in range(2):
        for dk in range(7):
            u_idx = 2 * m + dj
            v_idx = dj * p + dk
            w_idx = dk * n + 2
            add_rank1_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # A22 (1x1) * B2 (1x7): A[2,2], B row 2
    for dk in range(7):
        u_idx = 2 * m + 2
        v_idx = 2 * p + dk
        w_idx = dk * n + 2
        add_rank1_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # Convert to matrix format
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    for i in range(n*m):
        for c in range(rank):
            if c < len(U[i]):
                U_mat[i][c] = U[i][c]
    for i in range(m*p):
        for c in range(rank):
            if c < len(V[i]):
                V_mat[i][c] = V[i][c]
    for i in range(p*n):
        for c in range(rank):
            if c < len(W[i]):
                W_mat[i][c] = W[i][c]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
