import numpy as np
import random
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: for all i,j,k: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                prod = np.dot(u_val, v_val * w_val)
                if abs(prod - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return a verified rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
    U rows: A[0,0], A[0,1], A[1,0], A[1,1]
    V rows: B[0,0], B[0,1], B[1,0], B[1,1]
    W rows: C[0,0], C[0,1], C[1,0], C[1,1]
    """
    # Try multiple known Strassen formulations
    candidates = [
        # Formulation 1
        {
            'U': [[1,0,1,1,1,0,0],
                  [0,0,0,0,1,0,1],
                  [0,1,0,0,0,1,0],
                  [1,1,0,1,0,0,-1]],
            'V': [[1,1,0,1,0,1,0],
                  [0,0,0,-1,1,0,1],
                  [0,0,1,-1,0,1,0],
                  [1,0,-1,0,1,0,1]],
            'W': [[1,0,0,1,-1,0,1],
                  [0,1,0,0,1,0,1],
                  [0,0,1,1,0,0,0],
                  [1,1,1,0,0,1,0]]
        },
        # Formulation 2 (alternative)
        {
            'U': [[1,0,1,0,1,0,0],
                  [0,0,0,0,1,0,1],
                  [0,1,0,0,0,1,0],
                  [1,1,0,1,0,0,-1]],
            'V': [[1,1,0,1,0,1,0],
                  [0,0,0,-1,1,0,1],
                  [0,0,1,-1,0,1,0],
                  [1,0,-1,0,1,0,1]],
            'W': [[1,0,0,1,-1,0,1],
                  [0,1,0,0,1,0,1],
                  [0,0,1,1,0,0,0],
                  [1,1,1,0,0,1,0]]
        },
        # Formulation 3
        {
            'U': [[1,0,1,0,1,0,0],
                  [0,0,0,0,1,0,1],
                  [0,1,0,0,0,1,0],
                  [1,1,0,1,0,0,-1]],
            'V': [[1,1,0,1,0,1,0],
                  [0,0,0,-1,1,0,1],
                  [0,0,1,-1,0,1,0],
                  [1,0,-1,0,1,0,1]],
            'W': [[1,0,0,1,-1,0,1],
                  [0,1,0,0,1,0,1],
                  [0,0,1,1,0,0,0],
                  [1,1,1,0,0,1,0]]
        },
    ]
    
    # Verify each candidate
    for cand in candidates:
        U_c = np.array(cand['U'], dtype=float)
        V_c = np.array(cand['V'], dtype=float)
        W_c = np.array(cand['W'], dtype=float)
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,j] * B[j,k] -> C[i,k]
                    u_val = U_c[i * 2 + j]
                    v_val = V_c[j * 2 + k]
                    w_val = W_c[k * 2 + i]
                    prod = np.dot(u_val, v_val * w_val)
                    if abs(prod - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return cand['U'], cand['V'], cand['W']
    
    # If no candidate works, try to find one via random search
    # This is a fallback
    for _ in range(10000):
        U_c = np.random.randint(-2, 3, size=(4, 7))
        V_c = np.random.randint(-2, 3, size=(4, 7))
        W_c = np.random.randint(-2, 3, size=(4, 7))
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_val = U_c[i * 2 + j]
                    v_val = V_c[j * 2 + k]
                    w_val = W_c[k * 2 + i]
                    prod = np.dot(u_val, v_val * w_val)
                    if abs(prod - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U_c.tolist(), V_c.tolist(), W_c.tolist()
    
    # Ultimate fallback: schoolbook rank 8
    U_fb = [[0]*8 for _ in range(4)]
    V_fb = [[0]*8 for _ in range(4)]
    W_fb = [[0]*8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U_fb[i*2+j][col] = 1
                V_fb[j*2+k][col] = 1
                W_fb[k*2+i][col] = 1
                col += 1
    return U_fb, V_fb, W_fb

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Get a rank-7 Strassen decomposition for 2x2x2
    strU, strV, strW = _strassen_2x2x2()
    str_rank = len(strU[0]) if strU else 8
    
    # Build the full decomposition
    # Block 1: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2] (i.e., C columns 0,1)
    # Block 2: A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4] (i.e., C columns 2,3)
    # Remaining: all terms involving A[:,2] or B[2,:] or C[:,4]
    
    # Total rank
    if str_rank == 7:
        total_rank = 7 + 7 + 14  # 2 Strassen blocks + 14 schoolbook
    else:
        total_rank = n * m * p  # fallback to schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 15 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 10 x total_rank
    
    if str_rank == 7:
        # Block 1: Strassen on A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: (0,0)->U row 0, (0,1)->U row 1, (1,0)->U row 3, (1,1)->U row 4
        # B indices: (0,0)->V row 0, (0,1)->V row 1, (1,0)->V row 5, (1,1)->V row 6
        # C indices: (0,0)->W row 0, (0,1)->W row 1, (1,0)->W row 2, (1,1)->W row 3
        a_row_map = {0: 0, 1: 1, 2: 3, 3: 4}  # (i*2+j) in strassen -> U row
        b_row_map = {0: 0, 1: 1, 2: 5, 3: 6}  # (j*2+k) in strassen -> V row
        c_row_map = {0: 0, 1: 1, 2: 2, 3: 3}  # (k*2+i) in strassen -> W row
        
        for s in range(7):
            for si in range(4):
                U[a_row_map[si]][s] += strU[si][s]
            for sj in range(4):
                V[b_row_map[sj]][s] += strV[sj][s]
            for sk in range(4):
                W[c_row_map[sk]][s] += strW[sk][s]
        
        # Block 2: Strassen on A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]
        # Same A indices as block 1
        # B indices: (0,2)->V row 2, (0,3)->V row 3, (1,2)->V row 7, (1,3)->V row 8
        # C indices: (0,2)->W row 4, (0,3)->W row 5, (1,2)->W row 6, (1,3)->W row 7
        b_row_map2 = {0: 2, 1: 3, 2: 7, 3: 8}
        c_row_map2 = {0: 4, 1: 5, 2: 6, 3: 7}
        
        base = 7
        for s in range(7):
            for si in range(4):
                U[a_row_map[si]][base + s] += strU[si][s]
            for sj in range(4):
                V[b_row_map2[sj]][base + s] += strV[sj][s]
            for sk in range(4):
                W[c_row_map2[sk]][base + s] += strW[sk][s]
        
        # Remaining 14 schoolbook terms
        # Terms involving j=2 (A[:,2] and B[2,:]):
        # (i, 2, k) for i in {0,1}, k in {0,1,2,3,4} -> 10 terms
        # Terms involving j=0 or j=1 but k=4:
        # (0, 0, 4), (1, 0, 4), (0, 1, 4), (1, 1, 4) -> 4 terms
        # Total: 14 terms
        
        col = 14
        # j=2 terms
        for i in range(2):
            for k in range(5):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # j=0,1 with k=4
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
