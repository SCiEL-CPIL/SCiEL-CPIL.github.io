import time
import itertools

DIMENSIONS = [3, 5, 7]

def get_partitions(n):
    # Generate all partitions of n into 1s and 2s
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]
    if n == 2:
        return [[2], [1, 1]]
    # Recursive or iterative
    parts = []
    for p in get_partitions(n - 1):
        parts.append([1] + p)
    for p in get_partitions(n - 2):
        parts.append([2] + p)
    return parts

def block_rank(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return 1
    if n == 1 and m == 1 and p == 2:
        return 2
    if n == 1 and m == 2 and p == 1:
        return 2
    if n == 2 and m == 1 and p == 1:
        return 2
    if n == 1 and m == 2 and p == 2:
        return 4
    if n == 2 and m == 1 and p == 2:
        return 4
    if n == 2 and m == 2 and p == 1:
        return 4
    if n == 2 and m == 2 and p == 2:
        return 7
    return n * m * p # Fallback

def get_strassen_matrices():
    # Returns U, V, W for 2x2x2 Strassen
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W convention: W[k*n+i, r] -> for n=2, W[2*k+i, r]
    # C[0,0] is index 0. C[0,1] is index 1. C[1,0] is index 2. C[1,1] is index 3.
    # W[0, r] corresponds to C[0,0]
    # W[1, r] corresponds to C[0,1]
    # W[2, r] corresponds to C[1,0]
    # W[3, r] corresponds to C[1,1]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = A11 * (B11 + B22)
    # C11 += M1
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    
    # M2 = (A11 + A22) * B22
    # C11 += M2 - M1? No.
    # Standard Strassen:
    # C11 = M1 + M4 - M2 + M7
    # C12 = M3 + M6
    # C21 = M5 + M2
    # C22 = M1 - M4 + M5 - M7
    # Let's check the assignments.
    
    # M1 = A11 * (B11 + B22)
    # U: A11 (0) -> 1
    # V: B11 (0) -> 1, B22 (3) -> 1
    # W: C11 (0) -> 1
    
    # M2 = (A11 + A22) * B22
    # U: A11 (0) -> 1, A22 (3) -> 1
    # V: B22 (3) -> 1
    # W: C11 (0) -> -1, C21 (2) -> 1
    # Wait, C21 = M5 + M2. So W[2][1] = 1.
    # C11 = M1 + M4 - M2 + M7. So W[0][1] = -1.
    
    U[0][1] = 1
    U[3][1] = 1
    V[3][1] = 1
    W[0][1] = -1
    W[2][1] = 1
    
    # M3 = (A11 + A12) * B21
    # U: A11 (0) -> 1, A12 (1) -> 1
    # V: B21 (2) -> 1
    # W: C12 (1) -> 1
    # C12 = M3 + M6. So W[1][2] = 1.
    
    U[0][2] = 1
    U[1][2] = 1
    V[2][2] = 1
    W[1][2] = 1
    
    # M4 = A12 * (B21 - B11)
    # U: A12 (1) -> 1
    # V: B21 (2) -> 1, B11 (0) -> -1
    # W: C11 (0) -> 1, C12 (1) -> 1
    # C11 = ... + M4 ...
    # C12 = M3 + M6. Wait.
    # C12 = M3 + M6?
    # Let's verify Strassen formulas.
    # C11 = M1 + M4 - M2 + M7
    # C12 = M3 + M6
    # C21 = M5 + M2
    # C22 = M1 - M4 + M5 - M7
    
    # So for M4:
    # W[0][3] = 1
    # W[1][3] = 1? No, C12 does not have M4.
    # Wait, C12 = M3 + M6.
    # So M4 only contributes to C11.
    # Let's re-verify.
    # M4 = A12 * (B21 - B11)
    # A12 * B21 contributes to C12? No, A12 * B21 is (0,1) * (1,0) -> (0,0)? No.
    # A is 2x2. B is 2x2.
    # C_ij = sum_k A_ik B_kj.
    # A12 is row 0, col 1.
    # B21 is row 1, col 0.
    # A12 * B21 contributes to C_00?
    # k=1. A_01 * B_10. Yes, C_00.
    # So M4 contributes to C11.
    # Does it contribute to others?
    # A12 * B11? k=0. A_01 * B_00. C_00.
    # So M4 is only for C11.
    
    U[1][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    
    # M5 = (A21 - A11) * (B11 + B12)
    # U: A21 (2) -> 1, A11 (0) -> -1
    # V: B11 (0) -> 1, B12 (1) -> 1
    # W: C21 (2) -> 1, C22 (3) -> 1
    # C21 = M5 + M2.
    # C22 = M1 - M4 + M5 - M7.
    
    U[2][4] = 1
    U[0][4] = -1
    V[0][4] = 1
    V[1][4] = 1
    W[2][4] = 1
    W[3][4] = 1
    
    # M6 = (A11 + A22) * (B12 - B22)
    # U: A11 (0) -> 1, A22 (3) -> 1
    # V: B12 (1) -> 1, B22 (3) -> -1
    # W: C12 (1) -> 1
    # C12 = M3 + M6.
    
    U[0][5] = 1
    U[3][5] = 1
    V[1][5] = 1
    V[3][5] = -1
    W[1][5] = 1
    
    # M7 = (A21 - A22) * (B11 + B21)
    # U: A21 (2) -> 1, A22 (3) -> -1
    # V: B11 (0) -> 1, B21 (2) -> 1
    # W: C11 (0) -> 1, C22 (3) -> -1
    # C11 = M1 + M4 - M2 + M7.
    # C22 = M1 - M4 + M5 - M7.
    
    U[2][6] = 1
    U[3][6] = -1
    V[0][6] = 1
    V[2][6] = 1
    W[0][6] = 1
    W[3][6] = -1
    
    return U, V, W

def get_schoolbook_matrices(n, m, p):
    U = [[0]*(n*m*p) for _ in range(n*m)]
    V = [[0]*(n*m*p) for _ in range(m*p)]
    W = [[0]*(n*m*p) for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    parts_n = get_partitions(n)
    parts_m = get_partitions(m)
    parts_p = get_partitions(p)
    
    best_rank = float('inf')
    best_parts = None
    
    for pn in parts_n:
        if time.time() > deadline: break
        for pm in parts_m:
            if time.time() > deadline: break
            for pp in parts_p:
                rank = 0
                for ni in pn:
                    for mj in pm:
                        for pk in pp:
                            rank += block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_parts = (pn, pm, pp)
    
    if best_parts is None:
        # Fallback to schoolbook
        best_parts = ([n], [m], [p])
        best_rank = n*m*p
        
    pn, pm, pp = best_parts
    
    # Construct U, V, W
    U = [[0]*best_rank for _ in range(n*m)]
    V = [[0]*best_rank for _ in range(m*p)]
    W = [[0]*best_rank for _ in range(p*n)]
    
    # Precompute block matrices
    strassen = get_strassen_matrices()
    
    offset_u = 0
    offset_v = 0
    offset_w = 0
    rank_offset = 0
    
    for i_idx, ni in enumerate(pn):
        u_start = i_idx * m
        # Note: The partition is for n.
        # The rows of U are indexed by i (0..n-1) and j (0..m-1).
        # So for a block starting at i_start, the rows are i_start*m .. (i_start+ni-1)*m + m-1
        
        for j_idx, mj in enumerate(pm):
            v_start = j_idx * p
            for k_idx, pk in enumerate(pp):
                w_start = k_idx * n
                
                if ni == 2 and mj == 2 and pk == 2:
                    bU, bV, bW = strassen
                    b_rank = 7
                else:
                    bU, bV, bW = get_schoolbook_matrices(ni, mj, pk)
                    b_rank = ni * mj * pk
                
                # Copy bU to U
                for r in range(ni * mj):
                    u_row = u_start * m + r
                    for c in range(b_rank):
                        U[u_row][rank_offset + c] = bU[r][c]
                
                # Copy bV to V
                for r in range(mj * pk):
                    v_row = v_start * p + r
                    for c in range(b_rank):
                        V[v_row][rank_offset + c] = bV[r][c]
                
                # Copy bW to W
                for r in range(pk * ni):
                    w_row = w_start * n + r
                    for c in range(b_rank):
                        W[w_row][rank_offset + c] = bW[r][c]
                
                rank_offset += b_rank
    
    return {"U": U, "V": V, "W": W}
