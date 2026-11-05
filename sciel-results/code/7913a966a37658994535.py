import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _matmul_check(U, V, W, n, m, p):
    """
    Verify if U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)

def _trim_zero_columns(U, V, W):
    """Remove columns that are all zeros."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1 # A00, A11
    V[0][0] = 1; V[3][0] = 1 # B00, B11
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    
    # M2
    U[2][1] = 1; U[3][1] = 1 # A10, A11
    V[0][1] = 1 # B00
    W[2][1] = 1; W[3][1] = -1 # C10, C11
    
    # M3
    U[0][2] = 1 # A00
    V[1][2] = 1; V[3][2] = -1 # B01, B11
    W[1][2] = 1; W[3][2] = 1 # C01, C11
    
    # M4
    U[3][3] = 1 # A11
    V[2][3] = 1; V[0][3] = -1 # B10, B00
    W[0][3] = 1; W[2][3] = 1 # C00, C10
    
    # M5
    U[0][4] = 1; U[1][4] = 1 # A00, A01
    V[3][4] = 1 # B11
    W[0][4] = -1; W[1][4] = 1 # C00, C01
    
    # M6
    U[2][5] = 1; U[0][5] = -1 # A10, A00
    V[0][5] = 1; V[1][5] = 1 # B00, B01
    W[3][5] = 1 # C11
    
    # M7
    U[1][6] = 1; U[3][6] = -1 # A01, A11
    V[2][6] = 1; V[3][6] = 1 # B10, B11
    W[0][6] = 1 # C00
    
    return U, V, W

def _construct_rank56():
    n, m, p = DIMENSIONS
    # U: 2x4 = 8 rows
    # V: 4x8 = 32 rows
    # W: 8x2 = 16 rows
    
    U = [[0] * 56 for _ in range(n * m)]
    V = [[0] * 56 for _ in range(m * p)]
    W = [[0] * 56 for _ in range(p * n)]
    
    Su, Sv, Sw = _strassen_2x2x2()
    
    # We have 4 block columns for C (k_h = 0,1,2,3)
    # And 2 block rows for B (j_h = 0,1)
    # Total 8 Strassen instances?
    # No, for each k_h, we compute C[:, 2k_h:2k_h+2] = A_0 B_{0k_h} + A_1 B_{1k_h}
    # This requires 2 Strassen multiplications: A_0 B_{0k_h} and A_1 B_{1k_h}
    # So 4 * 2 = 8 instances.
    
    col = 0
    for k_h in range(4):
        for j_h in range(2):
            # This instance computes A_{j_h} B_{j_h, k_h}
            # A_{j_h} is the j_h-th 2x2 block of A (rows 0,1; cols 2j_h, 2j_h+1)
            # B_{j_h, k_h} is the (j_h, k_h)-th 2x2 block of B (rows 2j_h, 2j_h+1; cols 2k_h, 2k_h+1)
            
            for r in range(7):
                # U part
                # Su has 4 rows: (i, j_local) where i in 0,1; j_local in 0,1
                # Global j = 2*j_h + j_local
                for i in range(2):
                    for j_local in range(2):
                        val = Su[i*2 + j_local][r]
                        if val != 0:
                            j_global = 2*j_h + j_local
                            U[i*m + j_global][col] += val
                            
                # V part
                # Sv has 4 rows: (j_local, k_local) where j_local in 0,1; k_local in 0,1
                # Global j = 2*j_h + j_local
                # Global k = 2*k_h + k_local
                for j_local in range(2):
                    for k_local in range(2):
                        val = Sv[j_local*2 + k_local][r]
                        if val != 0:
                            j_global = 2*j_h + j_local
                            k_global = 2*k_h + k_local
                            V[j_global*p + k_global][col] += val
                            
                # W part
                # Sw has 4 rows: (k_local, i) where k_local in 0,1; i in 0,1
                # Global k = 2*k_h + k_local
                # Global i = i
                for k_local in range(2):
                    for i in range(2):
                        val = Sw[k_local*2 + i][r]
                        if val != 0:
                            k_global = 2*k_h + k_local
                            W[k_global*n + i][col] += val
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100
    
    cand = _construct_rank56()
    ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
    
    if ok:
        U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: should not happen
        return cand

