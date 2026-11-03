import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by partitioning
    # i: {0,1} | {2}
    # j: {0,1} | {2,3}
    # k: {0,1} | {2,3} | {4,5} | {6,7}
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec has length n*m=12, v_vec has length m*p=32, w_vec has length p*n=24."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2: local i in {0,1}, j in {0,1}, k in {0,1}
    # Global positions: i_off, j_off, k_off
    def strassen_2x2x2(i_off, j_off, k_off):
        # 7 terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + j_off] = 1      # A00
        u[(i_off+1)*m + (j_off+1)] = 1  # A11
        v[j_off*p + k_off] = 1      # B00
        v[(j_off+1)*p + (k_off+1)] = 1  # B11
        w[k_off*n + i_off] = 1      # C00
        w[(k_off+1)*n + (i_off+1)] = 1  # C11
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[(i_off+1)*m + j_off] = 1      # A10
        u[(i_off+1)*m + (j_off+1)] = 1  # A11
        v[j_off*p + k_off] = 1      # B00
        w[k_off*n + (i_off+1)] = 1  # C10
        add_term(u, v, w)
        
        # M3 = A00(B10-B11)
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + j_off] = 1      # A00
        v[(j_off+1)*p + k_off] = 1      # B10
        v[(j_off+1)*p + (k_off+1)] = -1  # B11
        w[k_off*n + i_off] = 1      # C00
        add_term(u, v, w)
        
        # M4 = A11(B01-B00)
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[(i_off+1)*m + (j_off+1)] = 1  # A11
        v[j_off*p + (k_off+1)] = 1      # B01
        v[j_off*p + k_off] = -1      # B00
        w[(k_off+1)*n + (i_off+1)] = 1  # C11
        add_term(u, v, w)
        
        # M5 = (A00+A10)B11
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + j_off] = 1      # A00
        u[(i_off+1)*m + j_off] = 1  # A10
        v[(j_off+1)*p + (k_off+1)] = 1  # B11
        w[(k_off+1)*n + i_off] = 1  # C01
        add_term(u, v, w)
        
        # M6 = (A11-A01)(B00+B10)
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[(i_off+1)*m + (j_off+1)] = 1  # A11
        u[i_off*m + (j_off+1)] = -1  # A01
        v[j_off*p + k_off] = 1      # B00
        v[(j_off+1)*p + k_off] = 1  # B10
        w[k_off*n + (i_off+1)] = 1  # C10
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B01+B11)
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + (j_off+1)] = 1  # A01
        u[(i_off+1)*m + (j_off+1)] = -1  # A11
        v[j_off*p + (k_off+1)] = 1  # B01
        v[(j_off+1)*p + (k_off+1)] = 1  # B11
        w[(k_off+1)*n + i_off] = 1  # C01
        add_term(u, v, w)
    
    # 1x2x2 rank-2: local i={0}, j in {0,1}, k in {0,1}
    def rank2_1x2x2(i_off, j_off, k_off):
        # Two terms: one for j=0, one for j=1
        # Term 1: A00 * B0k
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + j_off] = 1
        v[j_off*p + k_off] = 1
        v[j_off*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        w[(k_off+1)*n + i_off] = 1
        add_term(u, v, w)
        
        # Term 2: A01 * B1k
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        u[i_off*m + (j_off+1)] = 1
        v[(j_off+1)*p + k_off] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        w[(k_off+1)*n + i_off] = 1
        add_term(u, v, w)
    
    # Now apply to all blocks
    # i_parts: [(0,2), (2,3)]
    # j_parts: [(0,2), (2,4)]
    # k_parts: [(0,2), (2,4), (4,6), (6,8)]
    
    i_parts = [(0,2), (2,3)]
    j_parts = [(0,2), (2,4)]
    k_parts = [(0,2), (2,4), (4,6), (6,8)]
    
    for i_off, i_end in i_parts:
        for j_off, j_end in j_parts:
            for k_off, k_end in k_parts:
                i_size = i_end - i_off
                j_size = j_end - j_off
                k_size = k_end - k_off
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_2x2x2(i_off, j_off, k_off)
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    rank2_1x2x2(i_off, j_off, k_off)
    
    rank = len(U_cols)
    U = [U_cols[c][r] for r in range(n*m) for c in range(rank)]  # This is wrong
    # Actually U should be shape (n*m, rank), so U[r][c]
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
