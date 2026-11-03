"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build 4x4x4 decomposition via Kronecker product of two 2x2x2 Strassen blocks.
    Rank 49.
    U[i*4+j, col], V[j*4+k, col], W[k*4+i, col]
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0: continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0: continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0: continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build 4x4x2 decomposition using two 2x2x2 Strassen blocks.
    Rank 14.
    U[i*4+j, col], V[j*4+k, col], W[k*4+i, col]
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 14
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        # Block 1: k=0 (j1=0, k1=0)
        for i1 in range(2):
            for j1 in range(2):
                c = U2[i1*2+j1][t]
                if c != 0:
                    u_row = i1*4 + j1
                    U[u_row][t] += c
        for j1 in range(2):
            c = V2[j1*2+0][t]
            if c != 0:
                v_row = j1*4 + 0
                V[v_row][t] += c
        for i1 in range(2):
            c = W2[0*2+i1][t]
            if c != 0:
                w_row = 0*4 + i1
                W[w_row][t] += c
        
        # Block 2: k=1 (j1=0, k1=1)
        for i1 in range(2):
            for j1 in range(2):
                c = U2[i1*2+j1][t]
                if c != 0:
                    u_row = i1*4 + j1
                    U[u_row][7+t] += c
        for j1 in range(2):
            c = V2[j1*2+1][t]
            if c != 0:
                v_row = j1*4 + 1
                V[v_row][7+t] += c
        for i1 in range(2):
            c = W2[1*2+i1][t]
            if c != 0:
                w_row = 1*4 + i1
                W[w_row][7+t] += c
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140
    
    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3) — rank 49
    U4, V4, W4 = _build_4x4x4_strassen()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5) — rank 14
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5) — rank 24
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Verify correctness if time permits
    if time.time() < deadline - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Reconstruct C from U, V, W
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # More efficiently:
        # For each rank component r:
        #   u_r is a vector of length n*m (reshaped as n x m)
        #   v_r is a vector of length m*p (reshaped as m x p)
        #   w_r is a vector of length p*n (reshaped as p x n)
        #   C += u_r @ v_r * w_r (element-wise)
        rank = len(U_np[0])
        C_reconstructed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            w_r = W_np[:, r].reshape(p, n)
            C_reconstructed += (u_r @ v_r) * w_r.T
        
        if not np.allclose(C_expected, C_reconstructed, atol=1e-10):
            # Fallback to schoolbook if verification fails
            U_f, V_f, W_f = _build_schoolbook(n, m, p)
            return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
