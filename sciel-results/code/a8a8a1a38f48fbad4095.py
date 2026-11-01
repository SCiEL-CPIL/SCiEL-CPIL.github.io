import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7-multiplication decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-c)(e+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U: A indices [a,b,c,d] = [0,1,2,3]
    # V: B indices [e,f,g,h] = [0,1,2,3]
    # W: C indices [C11,C12,C21,C22] = [0,1,2,3] where C[k*2+i]
    #   C11: i=0,k=0 -> idx 0
    #   C12: i=0,k=1 -> idx 1
    #   C21: i=1,k=0 -> idx 2
    #   C22: i=1,k=1 -> idx 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11=1, C22=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C21=1, C22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C12=1, C22=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C11=1, C21=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C11=-1, C12=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g): U: c=1,d=-1; V: f=1,g=1; W: C22=1
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-c)(e+h): U: b=1,c=-1; V: e=1,h=1; W: C11=1
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    try:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
            
        # Check valid entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                    if np.abs(val - 1.0) > tol:
                        return False
        
        # Check invalid entries should be 0
        # We only check a subset for speed, but for this small size, full check is fast
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.dot(U_arr[a, :], np.dot(V_arr[b, :], W_arr[c, :]))
                        if np.abs(val) > tol:
                            return False
        
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # The 2x2x16 multiplication tensor is the direct sum of 8 independent 2x2x2 tensors.
    # Minimum rank is 8 * 7 = 56.
    
    U2, V2, W2 = _strassen_2x2x2()
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            # U: same for all blocks (A is 2x2)
            U[:, col] = U2[:, r]
            
            # V: B[j, block*2 + k] -> index j*p + (block*2 + k)
            for j in range(m):
                for k in range(2):
                    b_idx = j * p + (block * 2 + k)
                    v_idx = j * 2 + k
                    V[b_idx, col] = V2[v_idx, r]
                    
            # W: C[k, i] where k = block*2 + k_local, i as usual
            # W index: k*n + i
            for i in range(n):
                for k_local in range(2):
                    k = block * 2 + k_local
                    w_idx = k * n + i
                    w2_idx = k_local * n + i
                    W[w_idx, col] = W2[w2_idx, r]
    
    # Verify the decomposition
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails (should not happen)
            rank_sb = n * m * p
            U_sb = np.zeros((n * m, rank_sb))
            V_sb = np.zeros((m * p, rank_sb))
            W_sb = np.zeros((p * n, rank_sb))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, column] = 1
                        V_sb[j * p + k, column] = 1
                        W_sb[k * n + i, column] = 1
                        column += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
