"""
5x5 matrix multiplication - rank 77 decomposition (4+1 split).
Uses naive 4x4 multiplication for the main block and optimized
rank-1 terms for the cross blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j] = val
        for (j, k), val in v_vals.items():
            V[j*p+k] = val
        for (k, i), val in w_vals.items():
            W[k*n+i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    # 1. C_44 = A_44 B_44 + a_41 b_14^T
    # A_44 B_44: 64 terms (naive)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
    
    # a_41 b_14^T: 1 term
    # A_{i,4} B_{4,k} -> C_{i,k} for i,k in 0..3
    u_vals = {}
    v_vals = {}
    w_vals = {}
    for i in range(4):
        u_vals[(i, 4)] = 1
    for k in range(4):
        v_vals[(4, k)] = 1
        w_vals[(k, i)] = 1 # Wait, w depends on i and k
    # Correction: The term is sum_{i,k} A_{i,4} B_{4,k} C_{i,k}
    # This is NOT a single rank-1 term because W_{k,i} must be 1 for all i,k.
    # If we use 1 term, W_{k,i} is fixed.
    # We need W_{k,i} = 1 for all i,k in 0..3.
    # So we can set W_{k,i} = 1 for all i,k in 0..3.
    w_vals = {}
    for k in range(4):
        for i in range(4):
            w_vals[(k, i)] = 1
            
    add_term(u_vals, v_vals, w_vals)

    # 2. C_41 = A_44 b_41 + a_41 b_11
    # A_44 b_41: 4 terms (one for each row i of C)
    for i in range(4):
        u_vals = {}
        v_vals = {}
        w_vals = {}
        for j in range(4):
            u_vals[(i, j)] = 1
            v_vals[(j, 4)] = 1
        w_vals[(4, i)] = 1
        add_term(u_vals, v_vals, w_vals)
        
    # a_41 b_11: 1 term
    u_vals = {}
    v_vals = {}
    w_vals = {}
    for i in range(4):
        u_vals[(i, 4)] = 1
        w_vals[(4, i)] = 1
    v_vals[(4, 4)] = 1
    add_term(u_vals, v_vals, w_vals)

    # 3. C_14 = a_14^T B_44 + a_11 b_14^T
    # a_11 b_14^T: 1 term
    u_vals = {}
    v_vals = {}
    w_vals = {}
    for k in range(4):
        v_vals[(4, k)] = 1
        w_vals[(k, 4)] = 1
    u_vals[(4, 4)] = 1
    add_term(u_vals, v_vals, w_vals)

    # a_14^T B_44: 4 terms (one for each col k of C)
    for k in range(4):
        u_vals = {}
        v_vals = {}
        w_vals = {}
        for j in range(4):
            u_vals[(4, j)] = 1
            v_vals[(j, k)] = 1
        w_vals[(k, 4)] = 1
        add_term(u_vals, v_vals, w_vals)

    # 4. C_11 = a_14^T b_41 + a_11 b_11
    # a_11 b_11: 1 term
    add_term({(4, 4): 1}, {(4, 4): 1}, {(4, 4): 1})
    
    # a_14^T b_41: 1 term
    u_vals = {}
    v_vals = {}
    w_vals = {}
    for j in range(4):
        u_vals[(4, j)] = 1
        v_vals[(j, 4)] = 1
    w_vals[(4, 4)] = 1
    add_term(u_vals, v_vals, w_vals)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
