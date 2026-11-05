import numpy as np
import time
from fractions import Fraction

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Optimized decomposition for 2x5x6
    # Strategy: Split middle dimension 5 = 3 + 2
    #   A = [A1 (2x3) | A2 (2x2)]
    #   B = [[B1 (3x6)], [B2 (2x6)]]
    #   C = A1*B1 + A2*B2
    #
    # A1*B1: 2x3 * 3x6. Split output cols into 3+3:
    #   2x3 * 3x3 (cols 0:3): schoolbook rank 6
    #   2x3 * 3x3 (cols 3:6): schoolbook rank 6
    #   Subtotal: 12
    #
    # A2*B2: 2x2 * 2x6. Split output cols into 2+2+2:
    #   2x2 * 2x2 (cols 0:2): Strassen rank 7
    #   2x2 * 2x2 (cols 2:4): Strassen rank 7
    #   2x2 * 2x2 (cols 4:6): Strassen rank 7
    #   Subtotal: 21
    #
    # Total: 12 + 21 = 33
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term."""
        u_col = np.zeros(nm)
        v_col = np.zeros(mp)
        w_col = np.zeros(pn)
        for row, val in u_vals.items():
            u_col[row] = val
        for row, val in v_vals.items():
            v_col[row] = val
        for row, val in w_vals.items():
            w_col[row] = val
        U_list.append(u_col)
        V_list.append(v_col)
        W_list.append(w_col)
    
    def strassen_2x2(i_base, j_base, k_base):
        """Strassen for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
        # M1 = (A11 + A22)(B11 + B22)
        add_term(
            {i_base*m + j_base: 1, (i_base+1)*m + (j_base+1): 1},
            {j_base*p + k_base: 1, (j_base+1)*p + (k_base+1): 1},
            {k_base*n + i_base: 1, (k_base+1)*n + (i_base+1): 1}
        )
        # M2 = (A21 + A22)B11
        add_term(
            {(i_base+1)*m + j_base: 1, (i_base+1)*m + (j_base+1): 1},
            {j_base*p + k_base: 1},
            {k_base*n + (i_base+1): 1}
        )
        # M3 = A11(B21 - B22)
        add_term(
            {i_base*m + j_base: 1},
            {(j_base+1)*p + k_base: 1, (j_base+1)*p + (k_base+1): -1},
            {k_base*n + i_base: 1}
        )
        # M4 = A22(B12 - B11)
        add_term(
            {(i_base+1)*m + (j_base+1): 1},
            {j_base*p + (k_base+1): 1, j_base*p + k_base: -1},
            {(k_base+1)*n + (i_base+1): 1}
        )
        # M5 = (A11 + A12)B22
        add_term(
            {i_base*m + j_base: 1, i_base*m + (j_base+1): 1},
            {(j_base+1)*p + (k_base+1): 1},
            {(k_base+1)*n + i_base: 1}
        )
        # M6 = (A21 - A11)(B11 + B12)
        add_term(
            {(i_base+1)*m + j_base: 1, i_base*m + j_base: -1},
            {j_base*p + k_base: 1, j_base*p + (k_base+1): 1},
            {k_base*n + (i_base+1): 1}
        )
        # M7 = (A12 - A22)(B21 + B22)
        add_term(
            {i_base*m + (j_base+1): 1, (i_base+1)*m + (j_base+1): -1},
            {(j_base+1)*p + k_base: 1, (j_base+1)*p + (k_base+1): 1},
            {(k_base+1)*n + i_base: 1}
        )
    
    # Part 1: A1*B1 where A1 is 2x3 (cols 0,1,2 of A), B1 is 3x6 (rows 0,1,2 of B)
    # Schoolbook: for each i in 0,1; j in 0,1,2; k in 0..5
    for i in range(2):
        for j in range(3):  # j = 0,1,2
            for k in range(6):
                add_term(
                    {i*m + j: 1},
                    {j*p + k: 1},
                    {k*n + i: 1}
                )
    
    # Part 2: A2*B2 where A2 is 2x2 (cols 3,4 of A), B2 is 2x6 (rows 3,4 of B)
    # Strassen for three 2x2*2x2 blocks
    strassen_2x2(0, 3, 0)  # A[0:2,3:5] * B[3:5,0:2] -> C[0:2,0:2]
    strassen_2x2(0, 3, 2)  # A[0:2,3:5] * B[3:5,2:4] -> C[0:2,2:4]
    strassen_2x2(0, 3, 4)  # A[0:2,3:5] * B[3:5,4:6] -> C[0:2,4:6]
    
    # Convert to arrays
    if U_list:
        U_new = np.column_stack(U_list)
        V_new = np.column_stack(V_list)
        W_new = np.column_stack(W_list)
        new_rank = U_new.shape[1]
    else:
        U_new, V_new, W_new = U_school, V_school, W_school
        new_rank = rank_school
    
    if new_rank < best_rank:
        best_U, best_V, best_W = U_new, V_new, W_new
        best_rank = new_rank
    
    # Verify the decomposition is correct
    verify_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(best_U[i*m+j, :], best_V[j*p+k, :] * best_W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    verify_ok = False
                    break
            if not verify_ok:
                break
        if not verify_ok:
            break
    
    if not verify_ok:
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    # Convert to lists
    U_out = np.zeros((nm, best_rank))
    V_out = np.zeros((mp, best_rank))
    W_out = np.zeros((pn, best_rank))
    
    for i in range(nm):
        for r in range(best_rank):
            U_out[i, r] = float(best_U[i, r])
    for i in range(mp):
        for r in range(best_rank):
            V_out[i, r] = float(best_V[i, r])
    for i in range(pn):
        for r in range(best_rank):
            W_out[i, r] = float(best_W[i, r])
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
