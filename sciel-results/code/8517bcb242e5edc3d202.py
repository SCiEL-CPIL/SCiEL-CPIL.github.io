import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: { (i, j): coeff } for A
        v_dict: { (j, k): coeff } for B
        w_dict: { (k, i): coeff } for C
        """
        u_vec = [0.0] * (n * m)
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
            
        v_vec = [0.0] * (m * p)
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
            
        w_vec = [0.0] * (p * n)
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def strassen_222(a_j_offset, b_j_offset, c_k_offset):
        """
        Adds 7 terms for multiplying A[:, j0:j0+2] by B[j0:j0+2, k0:k0+2]
        a_j_offset: starting column index for A (0 or 2)
        b_j_offset: starting row index for B (0 or 2)
        c_k_offset: starting column index for C (0, 2, or 4)
        
        A indices: i in {0,1}, j in {j0, j0+1}
        B indices: j in {j0, j0+1}, k in {k0, k0+1}
        C indices: k in {k0, k0+1}, i in {0,1}
        """
        j0 = a_j_offset
        k0 = c_k_offset
        
        # A elements:
        # a = A[0, j0], b = A[0, j0+1]
        # c = A[1, j0], d = A[1, j0+1]
        
        # B elements:
        # e = B[j0, k0], f = B[j0, k0+1]
        # g = B[j0+1, k0], h = B[j0+1, k0+1]
        
        # C elements:
        # c00 = C[k0, 0], c01 = C[k0, 1]
        # c10 = C[k0+1, 0], c11 = C[k0+1, 1]
        
        # m1 = (a+d)(e+h)
        u1 = {(0, j0): 1, (1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        w1 = {(k0, 0): 1, (k0+1, 1): 1}
        add_term(u1, v1, w1)
        
        # m2 = (c+d)(g+h)
        u2 = {(1, j0): 1, (1, j0+1): 1}
        v2 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w2 = {(k0+1, 0): 1, (k0+1, 1): 1}
        add_term(u2, v2, w2)
        
        # m3 = a(f-h)
        u3 = {(0, j0): 1}
        v3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w3 = {(k0, 1): 1, (k0+1, 1): -1}
        add_term(u3, v3, w3)
        
        # m4 = d(g-e)
        u4 = {(1, j0+1): 1}
        v4 = {(j0+1, k0): 1, (j0, k0): -1}
        w4 = {(k0, 0): 1, (k0+1, 0): 1}
        add_term(u4, v4, w4)
        
        # m5 = (a+b)e
        u5 = {(0, j0): 1, (0, j0+1): 1}
        v5 = {(j0, k0): 1}
        w5 = {(k0, 1): 1, (k0+1, 1): -1}
        add_term(u5, v5, w5)
        
        # m6 = (c+d)h
        u6 = {(1, j0): 1, (1, j0+1): 1}
        v6 = {(j0+1, k0+1): 1}
        w6 = {(k0, 0): -1}
        add_term(u6, v6, w6)
        
        # m7 = (a-b)(f+g)
        u7 = {(0, j0): 1, (0, j0+1): -1}
        v7 = {(j0, k0+1): 1, (j0+1, k0): 1}
        w7 = {(k0, 0): 1}
        add_term(u7, v7, w7)

    def schoolbook_221(a_j_offset, b_j_offset, c_k_offset):
        """
        Adds 4 terms for multiplying A[:, j0:j0+2] by B[j0:j0+2, k0:k0+1]
        a_j_offset: starting column index for A (0 or 2)
        b_j_offset: starting row index for B (0 or 2)
        c_k_offset: starting column index for C (4)
        """
        j0 = a_j_offset
        k0 = c_k_offset
        
        # m1 = A[0, j0] * B[j0, k0] -> C[k0, 0]
        add_term({(0, j0): 1}, {(j0, k0): 1}, {(k0, 0): 1})
        
        # m2 = A[0, j0+1] * B[j0+1, k0] -> C[k0, 0]
        add_term({(0, j0+1): 1}, {(j0+1, k0): 1}, {(k0, 0): 1})
        
        # m3 = A[1, j0] * B[j0, k0] -> C[k0, 1]
        add_term({(1, j0): 1}, {(j0, k0): 1}, {(k0, 1): 1})
        
        # m4 = A[1, j0+1] * B[j0+1, k0] -> C[k0, 1]
        add_term({(1, j0+1): 1}, {(j0+1, k0): 1}, {(k0, 1): 1})

    # Block 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, :] is split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    strassen_222(0, 0, 0)
    strassen_222(0, 0, 2)
    schoolbook_221(0, 0, 4)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # B[2:4, :] is split into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    strassen_222(2, 2, 0)
    strassen_222(2, 2, 2)
    schoolbook_221(2, 2, 4)
    
    rank = len(U_cols)
    
    # Convert lists of columns to matrices
    # U is (n*m, rank)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
