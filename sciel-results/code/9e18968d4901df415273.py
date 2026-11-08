import time
import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - C1 = A * B1 using recursive Strassen (rank 49).
    - C2 = A * B2 using recursive Strassen (rank 49).
    - C3 = A * B3 using schoolbook (rank 16).
    - Total rank: 114.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 9
    U_dim = n * m  # 16
    V_dim = m * p  # 36
    W_dim = p * n  # 36
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2x2: 7 terms
    s2_terms = [
        (({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})),
        (({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})),
        (({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})),
        (({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})),
        (({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})),
        (({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})),
        (({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})),
    ]

    def add_strassen_4x4(b_col_off):
        for oa, ob, oc in s2_terms:
            for ia, ib, ic in s2_terms:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                for (oor, ooc), c_ao in oa.items():
                    for (ior, ioc), c_ia in ia.items():
                        r = oor * 2 + ior
                        c = ooc * 2 + ioc
                        u[r * 4 + c] += c_ao * c_ia
                
                for (oor, ooc), c_bo in ob.items():
                    for (ior, ioc), c_ib in ib.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        v[r * 9 + c] += c_bo * c_ib
                
                for (oor, ooc), c_co in oc.items():
                    for (ior, ioc), c_ic in ic.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        w[c * 4 + r] += c_co * c_ic

                add_term(u, v, w)

    def add_schoolbook_4x1(b_col):
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                u[i * 4 + j] = 1.0
                v[j * 9 + b_col] = 1.0
                w[b_col * 4 + i] = 1.0
                add_term(u, v, w)

    if time.time() > deadline:
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0] * 16
                    v = [0.0] * 36
                    w = [0.0] * 36
                    u[i * 4 + j] = 1.0
                    v[j * 9 + k] = 1.0
                    w[k * 4 + i] = 1.0
                    add_term(u, v, w)
    else:
        add_strassen_4x4(0)
        add_strassen_4x4(4)
        add_schoolbook_4x1(8)

    rank = len(U_cols)
    
    # Build output matrices using NumPy for structural differentiation
    if rank == 0:
        return {"U": np.zeros((16, 0)).tolist(), "V": np.zeros((36, 0)).tolist(), "W": np.zeros((36, 0)).tolist()}
        
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Vectorized assignment for better performance
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
