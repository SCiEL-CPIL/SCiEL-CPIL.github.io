"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen vectors for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Track which (i,j,k) terms are covered by Strassen
    covered = [[[False for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    # List of vectors
    U_vecs = []
    V_vecs = []
    W_vecs = []
    
    # Try to cover with 2x2x2 Strassen blocks
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                can_use = True
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if covered[i+di][j+dj][k+dk]:
                                can_use = False
                                break
                        if not can_use: break
                    if not can_use: break
                if not can_use: continue
                
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered[i+di][j+dj][k+dk] = True
                
                for c in range(7):
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    for li in range(2):
                        for lj in range(2):
                            val = u_strassen[c][li*2 + lj]
                            u_vec[(i+li)*m + (j+lj)] = val
                            
                    for lj in range(2):
                        for lk in range(2):
                            val = v_strassen[c][lj*2 + lk]
                            v_vec[(j+lj)*p + (k+lk)] = val
                            
                    for lk in range(2):
                        for li in range(2):
                            val = w_strassen[c][lk*2 + li]
                            w_vec[(k+lk)*n + (i+li)] = val
                            
                    U_vecs.append(u_vec)
                    V_vecs.append(v_vec)
                    W_vecs.append(w_vec)
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    u_vec[i*m + j] = 1
                    v_vec[j*p + k] = 1
                    w_vec[k*n + i] = 1
                    
                    U_vecs.append(u_vec)
                    V_vecs.append(v_vec)
                    W_vecs.append(w_vec)
    
    rank = len(U_vecs)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U_mat[r][c] = U_vecs[c][r]
        for r in range(m * p):
            V_mat[r][c] = V_vecs[c][r]
        for r in range(p * n):
            W_mat[r][c] = W_vecs[c][r]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
