"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Split p=9 into 4+4+1
    p1, p2, p3 = 4, 4, 1
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a schoolbook decomposition for n x m by m x p_sub
    def add_schoolbook(p_sub, col_offset):
        nonlocal rank
        for i in range(n):
            for j in range(m):
                for k in range(p_sub):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + (col_offset + k)] = 1
                    w[(col_offset + k) * n + i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
                    rank += 1
    
    add_schoolbook(p1, 0)   # columns 0-3
    add_schoolbook(p2, 4)   # columns 4-7
    add_schoolbook(p3, 8)   # column 8
    
    rank = len(U)
    # Convert lists of columns to row-major matrices
    U_mat = [[U[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
