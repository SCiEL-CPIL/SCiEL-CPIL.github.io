import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep margin
    
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    # i_offset: row offset in A (0 or 1)
    # j_offset: col offset in A (0 or 1)
    # k_offset: col offset in B (0..7)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices:
        # A: i in {0,1}, j in {0,1} -> Global i=i_off, j=j_off+j
        # B: j in {0,1}, k in {0,1} -> Global j=j_off+j, k=k_off+k
        # C: i in {0,1}, k in {0,1} -> Global i=i_off, k=k_off+k
        
        # M1 = (a+d)(e+h)
        # U: A[0,0]=1, A[1,1]=1
        U[i_off * m + (j_off + 0)][col] = 1
        U[i_off * m + (j_off + 1)][col] = 1
        # V: B[0,0]=1, B[1,1]=1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C[0,0]+=1, C[0,1]+=0, C[1,0]+=0, C[1,1]-=1? 
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So for M1: C00+=1, C11+=1
        W[(k_off + 0) * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M2 = (c+d)e
        # U: A[1,0]=1, A[1,1]=1
        U[i_off * m + (j_off + 0)][col] = 1
        U[i_off * m + (j_off + 1)][col] = 1
        # V: B[0,0]=1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        # W: C10+=1, C11-=1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = -1
        col += 1
        
        # M3 = a(f-h)
        # U: A[0,0]=1
        U[i_off * m + (j_off + 0)][col] = 1
        # V: B[0,1]=1, B[1,1]=-1
        V[(j_off + 0) * p + (k_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        # W: C01+=1, C11+=1
        W[(k_off + 1) * n + i_off][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M4 = d(g-e)
        # U: A[1,1]=1
        U[i_off * m + (j_off + 1)][col] = 1
        # V: B[1,0]=1, B[0,0]=-1
        V[(j_off + 1) * p + (k_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = -1
        # W: C00+=1, C10+=1
        W[(k_off + 0) * n + i_off][col] = 1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M5 = (a+b)h
        # U: A[0,0]=1, A[0,1]=1
        U[i_off * m + (j_off + 0)][col] = 1
        U[i_off * m + (j_off + 1)][col] = 1
        # V: B[1,1]=1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C00-=1, C01+=1
        W[(k_off + 0) * n + i_off][col] = -1
        W[(k_off + 1) * n + i_off][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # U: A[1,0]=1, A[0,0]=-1
        U[i_off * m + (j_off + 0)][col] = -1
        U[i_off * m + (j_off + 0)][col] = -1 # Wait, c is A[1,0], a is A[0,0]
        # Correct: U[1,0]=1, U[0,0]=-1
        U[i_off * m + (j_off + 0)][col] = -1
        U[i_off * m + (j_off + 0)][col] = -1 # Overwrite? No, indices are distinct in row?
        # Row i_off+1, col j_off+0 is c. Row i_off, col j_off+0 is a.
        U[i_off * m + (j_off + 0)][col] = -1
        U[(i_off + 1) * m + (j_off + 0)][col] = 1
        # V: B[0,0]=1, B[0,1]=1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 1)][col] = 1
        # W: C11+=1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # U: A[0,1]=1, A[1,1]=-1
        U[i_off * m + (j_off + 1)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = -1
        # V: B[1,0]=1, B[1,1]=1
        V[(j_off + 1) * p + (k_off + 0)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        # W: C00+=1
        W[(k_off + 0) * n + i_off][col] = 1
        col += 1
        
    # Helper to add direct 2x1x4 block
    # i_offset: row offset in A (0 or 1)
    # j_index: col index in A (2)
    # k_offset: col offset in B (0..7)
    def add_direct_2x1x4(i_off, j_idx, k_off):
        nonlocal col
        # For each i in 0..1, k in 0..3
        for i_local in range(2):
            for k_local in range(4):
                i_global = i_off + i_local
                k_global = k_off + k_local
                # Term A[i_global, j_idx] * B[j_idx, k_global] * C[i_global, k_global]
                U[i_global * m + j_idx][col] = 1
                V[j_idx * p + k_global][col] = 1
                W[k_global * n + i_global][col] = 1
                col += 1
                if time.time() > deadline:
                    return False
        return True

    # Block 0: k=0..3
    add_strassen(0, 0, 0) # i=0..1, j=0..1, k=0..1
    add_strassen(0, 0, 2) # i=0..1, j=0..1, k=2..3
    if not add_direct_2x1x4(0, 2, 0): return {"U": U, "V": V, "W": W}

    # Block 1: k=4..7
    add_strassen(0, 0, 4) # i=0..1, j=0..1, k=4..5
    add_strassen(0, 0, 6) # i=0..1, j=0..1, k=6..7
    if not add_direct_2x1x4(0, 2, 4): return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
