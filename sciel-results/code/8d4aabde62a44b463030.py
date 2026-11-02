import time
import numpy as np

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def _remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_strassen2x2():
    # Strassen 2x2x2: rank 7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-f), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: c11, c12, c21, c22
    # M1: (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: c11=1, c22=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (c+d)e: U: c=1,d=1; V: e=1; W: c21=1, c22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3: a(f-h): U: a=1; V: f=1,h=-1; W: c12=1, c22=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4: d(g-f): U: d=1; V: g=1,f=-1; W: c11=1, c21=1
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5: (a+b)h: U: a=1,b=1; V: h=1; W: c11=-1, c12=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6: (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: c22=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: c11=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _block_multiply(U1, V1, W1, n1, m1, p1, n2, m2, p2):
    # Multiply (n1,m1,p1) by (n2,m2,p2) block structure
    # Total: n=n1*n2, m=m1*m2, p=p1*p2
    # U has shape (n1*m1*n2*m2, rank1*rank2)
    # V has shape (m1*m2*n2*p2, rank1*rank2)  wait no
    # Actually block multiply: split A into n1 x n2 blocks of size (m1 x p1), B into m2 x p2 blocks... 
    # This is getting complex. Let me use a simpler approach.
    pass

def _build_decomposition(n, m, p):
    """Build a decomposition using block Strassen where possible."""
    if _remaining() < 5.0:
        return _build_schoolbook(n, m, p)
    
    # If any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _build_schoolbook(n, m, p)
    
    # Try to use 2x2 Strassen for sub-blocks
    # Strategy: if n>=2, m>=2, p>=2, split into 2x2 blocks where possible
    # For simplicity, use the following recursive strategy:
    # If n,m,p are all even or >= 2, we can try to use 2x2 Strassen blocks
    
    # Actually, let's use a simpler approach:
    # Decompose using 2x2 Strassen for the 2x2 sub-problems
    # For n x m x p, split into blocks of size 2x2x2 where possible
    
    # Let's use a direct approach:
    # If n>=2 and m>=2 and p>=2:
    #   Use Strassen for a 2x2x2 subproblem, and schoolbook for the rest
    #   This requires careful index mapping
    
    # For now, let's try a block decomposition approach
    # Split n into n_hi + n_lo, etc.
    
    # Simplest improvement: if all dims >= 2, use 2x2 Strassen for a corner
    if n >= 2 and m >= 2 and p >= 2 and _remaining() > 10.0:
        # Build 2x2x2 Strassen for indices (0,0,0), (0,0,1), (0,1,0), (0,1,1)
        # and schoolbook for the rest
        sU, sV, sW = _build_strassen2x2()
        rank_str = 7
        rank_rest = (n * m * p) - 8  # 8 elements in the 2x2x2 block
        rank = rank_str + max(0, rank_rest)
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen part for (i in {0,1}, j in {0,1}, k in {0,1})
        # U row index: i*m + j for i in {0,1}, j in {0,1}
        # V row index: j*p + k for j in {0,1}, k in {0,1}
        # W row index: k*n + i for k in {0,1}, i in {0,1}
        
        for c in range(rank_str):
            # U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    # sU has rows 0:a, 1:b, 2:c, 3:d corresponding to (0,0),(0,1),(1,0),(1,1)
                    sU_idx = i * 2 + j
                    U[u_idx, c] = sU[sU_idx, c]
            # V
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + k
                    sV_idx = j * 2 + k
                    V[v_idx, c] = sV[sV_idx, c]
            # W
            for k in range(2):
                for i in range(2):
                    w_idx = k * n + i
                    sW_idx = k * 2 + i
                    W[w_idx, c] = sW[sW_idx, c]
        
        # Schoolbook part for remaining (i,j,k) not in {0,1}x{0,1}x{0,1}
        col = rank_str
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    return _build_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_decomposition(n, m, p)
    rank = U.shape[1]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
