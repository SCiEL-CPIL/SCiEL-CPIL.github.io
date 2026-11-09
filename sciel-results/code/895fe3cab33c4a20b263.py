import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 22 decomposition
    # Split J (size 3) into J1 (size 2) and J2 (size 1)
    # Part 1: 2x2 by 2x4 (two 2x2x2 Strassen blocks) -> Rank 14
    # Part 2: 2x1 by 1x4 (schoolbook) -> Rank 8
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    strassen_222 = [
        # M1 = (A00+A11)(B00+B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2 = (A10+A11)B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]),
        # M3 = A00(B01-B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1) ]),
        # M4 = A11(B10-B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (3, 1) ]),
        # M5 = (A00+A01)B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1) ]),
        # M6 = (A10-A00)(B00+B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (2, 1) ]),
        # M7 = (A01-A11)(B10+B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (3, 1) ]),
    ]
    
    def add_strassen_block(v_offset, w_offset, col_start):
        for t_idx, (u_c, v_c, w_c) in enumerate(strassen_222):
            col = col_start + t_idx
            for idx, coeff in u_c:
                i = idx // 2
                j = idx % 2
                full_u_idx = i * m + j
                U[full_u_idx][col] += coeff
            for idx, coeff in v_c:
                j = idx // 2
                k = idx % 2
                full_v_idx = j * p + (k + v_offset)
                V[full_v_idx][col] += coeff
            for idx, coeff in w_c:
                k = idx // 2
                i = idx % 2
                full_w_idx = (k + w_offset) * n + i
                W[full_w_idx][col] += coeff

    # Block 1: k in {0, 1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: k in {2, 3}
    add_strassen_block(2, 2, 7)
    
    # Part 2: 2x1 by 1x4
    # A[:, 2] * B[2, :]
    col_start = 14
    for i in range(n):
        for k in range(p):
            col = col_start + i * p + k
            full_u_idx = i * m + 2
            U[full_u_idx][col] = 1
            full_v_idx = 2 * p + k
            V[full_v_idx][col] = 1
            full_w_idx = k * n + i
            W[full_w_idx][col] = 1

    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    term = W_np[k*n+i, r]
                    for j in range(m):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * term
                C_decomposed[i, k] = s
        
        error = np.max(np.abs(C_expected - C_decomposed))
        if error > 1e-9:
            # Fallback
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
    except Exception:
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
