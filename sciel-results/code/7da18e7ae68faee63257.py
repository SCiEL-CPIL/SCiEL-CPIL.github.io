import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 2, 2, 13

    # Schoolbook fallback (Rank 52)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        rank_strassen = 6 * 7 + 4
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))

        # Strassen terms for 2x2x2x2
        # A indices: [a00, a01, a10, a11]
        # B indices: [b00, b01, b10, b11]
        # C indices: [c00, c10, c01, c11] (W index = j*n+i, so c00 is j=0,i=0; c10 is j=1,i=0? 
        # Wait. W index is j*n + i.
        # Block j0, j1.
        # c00 corresponds to j=j0, i=0 -> index j0*n + 0
        # c10 corresponds to j=j0, i=1? No.
        # Standard C is 2x2. C[i,j].
        # Here C is 2x13. C[i,j].
        # W index is j*n + i.
        # So for block (j0, j1):
        # C[i, j0] is at W[j0*n + i]
        # C[i, j1] is at W[j1*n + i]
        
        # Strassen output C = [[c00, c01], [c10, c11]]
        # c00 = M1 - M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 - M6
        
        # Let's map Strassen C indices to W indices.
        # Strassen C[0,0] -> global C[0, j0] -> W[j0*n + 0]
        # Strassen C[0,1] -> global C[0, j1] -> W[j1*n + 0]
        # Strassen C[1,0] -> global C[1, j0] -> W[j0*n + 1]
        # Strassen C[1,1] -> global C[1, j1] -> W[j1*n + 1]

        strassen_terms = [
            # M1: (A00+A11)(B00+B11) -> C00, C11
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), 
             np.array([1, 0, 0, 1])), # w: c00=1, c10=0, c01=0, c11=1? 
             # My previous code:
             # W[j0*n+0] = w[0] (c00)
             # W[j0*n+1] = w[1] (c10)
             # W[j1*n+0] = w[2] (c01)
             # W[j1*n+1] = w[3] (c11)
             
            # Let's re-verify the W mapping in the previous code.
            # Previous:
            # W[j0 * n + 0, col] = w_local[0] # c00
            # W[j0 * n + 1, col] = w_local[1] # c10
            # W[j1 * n + 0, col] = w_local[2] # c01
            # W[j1 * n + 1, col] = w_local[3] # c11
            
            # Strassen formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B00-B10)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 - M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 - M6
        ]
        
        # Correct Strassen terms (u, v, w)
        # u: A-coeffs, v: B-coeffs, w: C-coeffs [c00, c10, c01, c11]
        
        terms = [
            # M1
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            # M2
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            # M3
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            # M4
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            # M5
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            # M6
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            # M7
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1

            for t in range(7):
                u_local, v_local, w_local = terms[t]

                # A: global index = i*m + k
                U[0, col] = u_local[0]
                U[1, col] = u_local[1]
                U[2, col] = u_local[2]
                U[3, col] = u_local[3]

                # B: global index = k*p + j
                V[0 * p + j0, col] = v_local[0]
                V[0 * p + j1, col] = v_local[1]
                V[1 * p + j0, col] = v_local[2]
                V[1 * p + j1, col] = v_local[3]

                # C: global index = j*n + i
                W[j0 * n + 0, col] = w_local[0]
                W[j0 * n + 1, col] = w_local[1]
                W[j1 * n + 0, col] = w_local[2]
                W[j1 * n + 1, col] = w_local[3]

                col += 1

        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1

        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0

        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))

        if error < 1e-10:
            best_rank = rank_strassen
            best_U, best_V, best_W = U, V, W

    except Exception:
        pass

    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
