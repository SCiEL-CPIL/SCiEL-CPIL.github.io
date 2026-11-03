import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 140.0

    # Helper to get Strassen 2x2x2 factors
    # U: 4x4 (i,j) -> r, V: 4x4 (j,k) -> r, W: 4x4 (k,i) -> r
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Helper for 2x2x1 factors (schoolbook)
    # U: 4x4, V: 4x4, W: 4x4
    # Only 4 non-zero columns needed, but we can just use identity-like
    # Actually, for 2x2x1, rank 4.
    # U[i*2+j, r] = delta(i,j, r) where r=i*2+j?
    # V[j*2+k, r] = delta(j,k, r) ? No, k is fixed?
    # The block is 2x2 (A) * 2x1 (B) -> 2x1 (C).
    # A indices (i,j), B indices (j,k=0..1? No, k is the single dim).
    # In our global context, the "1" dimension is a specific k value.
    # Let's construct 2x2x1 factors for a specific k.
    # U: (i,j) -> r. V: (j,k) -> r. W: (k,i) -> r.
    # Since k is fixed to a single value (say 0 locally), V depends on j.
    # W depends on i.
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C0 += A00 B0
    # r=1: U[0,1]=1, V[1,0]=1, W[0,0]=1 -> C0 += A01 B1
    # r=2: U[1,0]=1, V[0,0]=1, W[1,0]=1 -> C1 += A10 B0
    # r=3: U[1,1]=1, V[1,0]=1, W[1,0]=1 -> C1 += A11 B1
    # Note: V index is j*2+k. If k=0, V[0,0] and V[1,0] are used.
    # W index is k*2+i. If k=0, W[0,0] and W[1,0] are used.
    
    def make_221_factors(k_val):
        # Returns U_block (4x4), V_block (4x4), W_block (4x4)
        # Only 4 columns are non-zero.
        U_b = np.zeros((4, 4))
        V_b = np.zeros((4, 4))
        W_b = np.zeros((4, 4))
        
        # r=0: A00 * B0
        U_b[0, 0] = 1 # i=0, j=0
        V_b[0, 0] = 1 # j=0, k=0 (local k=0)
        W_b[0, 0] = 1 # k=0, i=0
        
        # r=1: A01 * B1
        U_b[1, 1] = 1 # i=0, j=1
        V_b[2, 1] = 1 # j=1, k=0 (local k=0, so index j*2+0 = 2)
        W_b[0, 2] = 1 # k=0, i=0
        
        # r=2: A10 * B0
        U_b[2, 2] = 1 # i=1, j=0
        V_b[0, 2] = 1 # j=0, k=0
        W_b[1, 3] = 1 # k=0, i=1
        
        # r=3: A11 * B1
        U_b[3, 3] = 1 # i=1, j=1
        V_b[2, 3] = 1 # j=1, k=0
        W_b[1, 3] = 1 # k=0, i=1
        
        return U_b, V_b, W_b

    total_rank = 50
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Partition k into [0, 2] (size 3) and [3, 6] (size 4)
    
    # Block 1: k in [0, 2]
    # Split j into [0,1] and [2,3]
    # J=0,1: 2x2x3 (rank 11)
    # J=2,3: 2x2x3 (rank 11)
    
    # Helper to add 2x2x3 block
    def add_223_block(j_offset, k_offset, col_start):
        # j_offset: 0 or 2
        # k_offset: 0
        # Returns new col
        # 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4)
        # 2x2x2 part: k in [0, 1]
        U_222, V_222, W_222 = U_s, V_s, W_s
        for i in range(2):
            for j_loc in range(2):
                U[i*m + j_offset + j_loc, col_start:col_start+7] += U_222[i*2 + j_loc]
        for j_loc in range(2):
            for k_loc in range(2):
                V[(j_offset + j_loc)*p + k_offset + k_loc, col_start:col_start+7] += V_222[j_loc*2 + k_loc]
        for k_loc in range(2):
            for i in range(2):
                W[(k_offset + k_loc)*n + i, col_start:col_start+7] += W_222[k_loc*2 + i]
        
        # 2x2x1 part: k in [2, 2] (local k=2, global k=k_offset+2)
        k_val = k_offset + 2
        U_221, V_221, W_221 = make_221_factors(k_val)
        col_221 = col_start + 7
        for i in range(2):
            for j_loc in range(2):
                U[i*m + j_offset + j_loc, col_221:col_221+4] += U_221[i*2 + j_loc]
        for j_loc in range(2):
            # V index: (j_offset + j_loc)*p + k_val
            # V_221 index: j_loc*2 + 0 (since local k=0 in the 221 factor definition)
            V[(j_offset + j_loc)*p + k_val, col_221:col_221+4] += V_221[j_loc*2]
        for i in range(2):
            # W index: k_val*n + i
            # W_221 index: 0*2 + i (since local k=0)
            W[k_val*n + i, col_221:col_221+4] += W_221[i]
            
        return col_start + 11

    col = add_223_block(0, 0, col)
    col = add_223_block(2, 0, col)
    
    # Block 2: k in [3, 6]
    # Split j into [0,1] and [2,3]
    # J=0,1: 2x2x4 (rank 14)
    # J=2,3: 2x2x4 (rank 14)
    
    # Helper to add 2x2x4 block
    def add_224_block(j_offset, k_offset, col_start):
        # 2x2x4 = 2x2x2 (rank 7) + 2x2x2 (rank 7)
        # Part 1: k in [0, 1] (local) -> global [k_offset, k_offset+1]
        for i in range(2):
            for j_loc in range(2):
                U[i*m + j_offset + j_loc, col_start:col_start+7] += U_s[i*2 + j_loc]
        for j_loc in range(2):
            for k_loc in range(2):
                V[(j_offset + j_loc)*p + k_offset + k_loc, col_start:col_start+7] += V_s[j_loc*2 + k_loc]
        for k_loc in range(2):
            for i in range(2):
                W[(k_offset + k_loc)*n + i, col_start:col_start+7] += W_s[k_loc*2 + i]
                
        # Part 2: k in [2, 3] (local) -> global [k_offset+2, k_offset+3]
        col_2 = col_start + 7
        for i in range(2):
            for j_loc in range(2):
                U[i*m + j_offset + j_loc, col_2:col_2+7] += U_s[i*2 + j_loc]
        for j_loc in range(2):
            for k_loc in range(2):
                V[(j_offset + j_loc)*p + k_offset + 2 + k_loc, col_2:col_2+7] += V_s[j_loc*2 + k_loc]
        for k_loc in range(2):
            for i in range(2):
                W[(k_offset + 2 + k_loc)*n + i, col_2:col_2+7] += W_s[k_loc*2 + i]
                
        return col_start + 14

    col = add_224_block(0, 3, col)
    col = add_224_block(2, 3, col)
    
    # Verify
    def check_valid(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    if time.time() < deadline:
        if check_valid(U, V, W):
            def clean(x):
                if abs(x - round(x)) < 1e-10:
                    return int(round(x))
                if abs(2*x - round(2*x)) < 1e-10:
                    return round(2*x) / 2
                return float(x)
            return {
                "U": [[clean(U[i, j]) for j in range(total_rank)] for i in range(n*m)],
                "V": [[clean(V[i, j]) for j in range(total_rank)] for i in range(m*p)],
                "W": [[clean(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
            }
    
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
