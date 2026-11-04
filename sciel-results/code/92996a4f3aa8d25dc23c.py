import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    block_sizes = [2, 2, 1]
    n_blocks = len(block_sizes)
    
    row_offsets = [0]
    for s in block_sizes:
        row_offsets.append(row_offsets[-1] + s)
    
    terms = []
    
    def schoolbook_block(a_start, a_size, b_start, b_size, c_start, c_size):
        for ai in range(a_size):
            for bi in range(b_size):
                for ci in range(c_size):
                    u = [0] * N
                    v = [0] * M
                    w = [0] * P
                    u[(a_start + ai) * m + (b_start + bi)] = 1
                    v[(b_start + bi) * p + (c_start + ci)] = 1
                    w[(c_start + ci) * n + (a_start + ai)] = 1
                    terms.append((u, v, w))
    
    def strassen_block(a_start, b_start, c_start):
        def a_idx(r, c):
            return (a_start + r) * m + (b_start + c)
        def b_idx(r, c):
            return (b_start + r) * p + (c_start + c)
        def c_idx(r, c):
            return (c_start + r) * n + (a_start + c)
        
        def add_to_vec(vec, idx, coeff):
            vec[idx] += coeff
        
        # M1 = (A00+A11)(B00+B11)
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(0,0), 1); add_to_vec(u, a_idx(1,1), 1)
        add_to_vec(v, b_idx(0,0), 1); add_to_vec(v, b_idx(1,1), 1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r, c), 1)
                add_to_vec(w, c_idx(r+1, c+1), 1)
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(1,0), 1); add_to_vec(u, a_idx(1,1), 1)
        add_to_vec(v, b_idx(0,0), 1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r+1, c), 1)
        terms.append((u, v, w))
        
        # M3 = A00(B10-B11)
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(0,0), 1)
        add_to_vec(v, b_idx(1,0), 1); add_to_vec(v, b_idx(1,1), -1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r, c+1), 1)
                add_to_vec(w, c_idx(r+1, c+1), 1)
        terms.append((u, v, w))
        
        # M4 = A11(B01-B00)
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(1,1), 1)
        add_to_vec(v, b_idx(0,1), 1); add_to_vec(v, b_idx(0,0), -1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r, c), 1)
                add_to_vec(w, c_idx(r+1, c), 1)
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(0,0), 1); add_to_vec(u, a_idx(0,1), 1)
        add_to_vec(v, b_idx(1,1), 1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r, c), -1)
                add_to_vec(w, c_idx(r, c+1), 1)
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(1,0), 1); add_to_vec(u, a_idx(0,0), -1)
        add_to_vec(v, b_idx(0,0), 1); add_to_vec(v, b_idx(0,1), 1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r+1, c+1), 1)
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        u = [0]*N; v = [0]*M; w = [0]*P
        add_to_vec(u, a_idx(0,1), 1); add_to_vec(u, a_idx(1,1), -1)
        add_to_vec(v, b_idx(1,0), 1); add_to_vec(v, b_idx(1,1), 1)
        for r in range(2):
            for c in range(2):
                add_to_vec(w, c_idx(r, c), 1)
        terms.append((u, v, w))
    
    for i in range(n_blocks):
        for j in range(n_blocks):
            for k in range(n_blocks):
                a_start = row_offsets[i]
                b_start = row_offsets[j]
                c_start = row_offsets[k]
                a_size = block_sizes[i]
                b_size = block_sizes[j]
                c_size = block_sizes[k]
                
                if a_size == 2 and b_size == 2 and c_size == 2:
                    strassen_block(a_start, b_start, c_start)
                else:
                    schoolbook_block(a_start, a_size, b_start, b_size, c_start, c_size)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    for col, (u_vec, v_vec, w_vec) in enumerate(terms):
        for row in range(N):
            if u_vec[row] != 0:
                U[row][col] = u_vec[row]
        for row in range(M):
            if v_vec[row] != 0:
                V[row][col] = v_vec[row]
        for row in range(P):
            if w_vec[row] != 0:
                W[row][col] = w_vec[row]
    
    # Verify using numpy
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
        # T[a,b,c] = 1 if a%m==b//p and a//m==c%n and b%p==c//n
        # For fixed (a,b) with a%m==b//p: result vector over c is e_{(b%p)*n + (a//m)}
        
        max_err = 0.0
        for a in range(N):
            for b in range(M):
                prod = U_np[a] * V_np[b]  # shape (rank,)
                result = prod @ W_np      # shape (P,)
                if a % m == b // p:
                    expected = np.zeros(P)
                    expected[(b % p) * n + (a // m)] = 1.0
                else:
                    expected = np.zeros(P)
                err = np.max(np.abs(result - expected))
                if err > max_err:
                    max_err = err
        
        if max_err > 1e-10:
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(N)]
            V_school = [[0] * rank_school for _ in range(M)]
            W_school = [[0] * rank_school for _ in range(P)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
