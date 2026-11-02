import numpy as np
import time

def construct():
    """
    Rank-56 decomposition for 2x2x16 matrix multiplication.
    Uses Strassen's rank-7 algorithm tiled across 8 column-pair blocks.
    All entries are integers in {-1, 0, 1}.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 2, 16
    num_pairs = p // 2  # 8
    rank = 7 * num_pairs  # 56

    # Strassen 2x2x2 core: 7 rank-1 terms
    # A indices: 0=a(0,0), 1=b(0,1), 2=c(1,0), 3=d(1,1)
    # B indices: 0=e(0,0), 1=f(0,1), 2=g(1,0), 3=h(1,1)
    # C indices: 0=C(0,0), 1=C(0,1), 2=C(1,0), 3=C(1,1)
    
    # U_core: (4, 7) - A factors
    U_core = np.zeros((4, 7), dtype=np.int64)
    # V_core: (4, 7) - B factors  
    V_core = np.zeros((4, 7), dtype=np.int64)
    # W_core: (4, 7) - C factors
    W_core = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U_core[0, 0] = 1; U_core[3, 0] = 1
    V_core[0, 0] = 1; V_core[3, 0] = 1
    W_core[0, 0] = 1; W_core[3, 0] = 1
    
    # M2 = (c+d)e
    U_core[2, 1] = 1; U_core[3, 1] = 1
    V_core[0, 1] = 1
    W_core[2, 1] = 1; W_core[3, 1] = -1
    
    # M3 = a(f-h)
    U_core[0, 2] = 1
    V_core[1, 2] = 1; V_core[3, 2] = -1
    W_core[1, 2] = 1; W_core[3, 2] = 1
    
    # M4 = d(g-e)
    U_core[3, 3] = 1
    V_core[2, 3] = 1; V_core[0, 3] = -1
    W_core[0, 3] = 1; W_core[2, 3] = 1
    
    # M5 = (a+b)h
    U_core[0, 4] = 1; U_core[1, 4] = 1
    V_core[3, 4] = 1
    W_core[0, 4] = -1; W_core[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U_core[2, 5] = 1; U_core[0, 5] = -1
    V_core[0, 5] = 1; V_core[1, 5] = 1
    W_core[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U_core[1, 6] = 1; U_core[3, 6] = -1
    V_core[2, 6] = 1; V_core[3, 6] = 1
    W_core[0, 6] = 1
    
    # Build full matrices
    U = np.zeros((n * m, rank), dtype=np.int64)  # (4, 56)
    V = np.zeros((m * p, rank), dtype=np.int64)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=np.int64)  # (32, 56)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        # U: same core for all blocks
        U[:, col_offset:col_offset+7] = U_core
        
        # V: B row index = j*p + k
        # V_core rows: 0=(j=0,k_local=0), 1=(j=0,k_local=1), 2=(j=1,k_local=0), 3=(j=1,k_local=1)
        # Full B col: 2t, 2t+1
        V[2*t, col_offset:col_offset+7] = V_core[0]       # j=0, k=2t
        V[2*t+1, col_offset:col_offset+7] = V_core[1]     # j=0, k=2t+1
        V[16+2*t, col_offset:col_offset+7] = V_core[2]    # j=1, k=2t
        V[16+2*t+1, col_offset:col_offset+7] = V_core[3]  # j=1, k=2t+1
        
        # W: C row index = k*n + i (trace convention)
        # W_core rows: 0=C(0,0), 1=C(0,1), 2=C(1,0), 3=C(1,1)
        # Full C col: 2t, 2t+1
        W[4*t, col_offset:col_offset+7] = W_core[0]       # k=2t, i=0
        W[4*t+2, col_offset:col_offset+7] = W_core[1]     # k=2t+1, i=0
        W[4*t+1, col_offset:col_offset+7] = W_core[2]     # k=2t, i=1
        W[4*t+3, col_offset:col_offset+7] = W_core[3]     # k=2t+1, i=1
    
    # Quick verification: check one block
    if time.time() < deadline:
        # Verify block 0: sum_r U[i,r]*V[j,r]*W[k,r] should give correct tensor entries
        # For A[0,0]=U[0], B[0,0]=V[0], C[0,0]=W[0]: should be 1
        val = np.sum(U[0, :7] * V[0, :7] * W[0, :7])
        assert val == 1, f"Verification failed: {val}"
        
        # A[0,0]*B[1,0] -> C[0,0] should be 1
        val = np.sum(U[0, :7] * V[16, :7] * W[0, :7])
        assert val == 1, f"Verification failed: {val}"
        
        # A[0,0]*B[0,1] -> C[0,0] should be 0
        val = np.sum(U[0, :7] * V[1, :7] * W[0, :7])
        assert val == 0, f"Verification failed: {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
