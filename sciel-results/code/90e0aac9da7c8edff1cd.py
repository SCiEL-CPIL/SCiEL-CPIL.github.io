import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the 2x2x2 Strassen blocks
    # i in {0, 1}, j in {0, 1, 2, 3}, k in {0, 1, 2, 3, 4, 5}
    # We form blocks of size 2x2x2
    # i_start: 0
    # j_start: 0, 2
    # k_start: 0, 2, 4
    
    blocks = []
    for i_s in [0]:
        for j_s in [0, 2]:
            for k_s in [0, 2, 4]:
                blocks.append((i_s, j_s, k_s))
                
    # Set of all (i, j, k) covered by blocks
    covered = set()
    for i_s, j_s, k_s in blocks:
        for i in range(i_s, i_s + 2):
            for j in range(j_s, j_s + 2):
                for k in range(k_s, k_s + 2):
                    covered.add((i, j, k))
                    
    # Helper to add a column
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Add Strassen terms for each block
    for i_s, j_s, k_s in blocks:
        # Local indices 0, 1 map to global i_s, i_s+1 etc.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (0,0), (1,1)
        u[i_s * m + j_s] = 1
        u[(i_s+1) * m + (j_s+1)] = 1
        # v: (0,0), (1,1)
        v[j_s * p + k_s] = 1
        v[(j_s+1) * p + (k_s+1)] = 1
        # w: (0,0), (1,1) -> C00, C11
        w[k_s * n + i_s] = 1
        w[(k_s+1) * n + (i_s+1)] = 1
        add_col(u, v, w)
        
        # Term 2: M2 = (A10+A11)B00
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (1,0), (1,1)
        u[(i_s+1) * m + j_s] = 1
        u[(i_s+1) * m + (j_s+1)] = 1
        # v: (0,0)
        v[j_s * p + k_s] = 1
        # w: (1,0) +, (1,1) -
        w[k_s * n + (i_s+1)] = 1
        w[(k_s+1) * n + (i_s+1)] = -1
        add_col(u, v, w)

        # Term 3: M3 = A00(B10-B11)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (0,0)
        u[i_s * m + j_s] = 1
        # v: (1,0), (1,1)
        v[(j_s+1) * p + k_s] = 1
        v[(j_s+1) * p + (k_s+1)] = -1
        # w: (0,1) +, (1,1) +
        w[(k_s+1) * n + i_s] = 1
        w[(k_s+1) * n + (i_s+1)] = 1
        add_col(u, v, w)

        # Term 4: M4 = A11(B01-B00)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (1,1)
        u[(i_s+1) * m + (j_s+1)] = 1
        # v: (0,1), (0,0)
        v[j_s * p + (k_s+1)] = 1
        v[j_s * p + k_s] = -1
        # w: (1,0) +, (1,1) +
        w[k_s * n + (i_s+1)] = 1
        w[(k_s+1) * n + (i_s+1)] = 1
        add_col(u, v, w)

        # Term 5: M5 = (A00+A01)B11
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (0,0), (0,1)
        u[i_s * m + j_s] = 1
        u[i_s * m + (j_s+1)] = 1
        # v: (1,1)
        v[(j_s+1) * p + (k_s+1)] = 1
        # w: (0,0) -, (0,1) +
        w[k_s * n + i_s] = -1
        w[(k_s+1) * n + i_s] = 1
        add_col(u, v, w)

        # Term 6: M6 = (A10-A00)(B00+B01)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (1,0), (0,0)
        u[(i_s+1) * m + j_s] = 1
        u[i_s * m + j_s] = -1
        # v: (0,0), (0,1)
        v[j_s * p + k_s] = 1
        v[j_s * p + (k_s+1)] = 1
        # w: (1,1) +
        w[(k_s+1) * n + (i_s+1)] = 1
        add_col(u, v, w)

        # Term 7: M7 = (A01-A11)(B10+B11)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        # u: (0,1), (1,1)
        u[i_s * m + (j_s+1)] = 1
        u[(i_s+1) * m + (j_s+1)] = -1
        # v: (1,0), (1,1)
        v[(j_s+1) * p + k_s] = 1
        v[(j_s+1) * p + (k_s+1)] = 1
        # w: (0,0) +, (1,1) +
        w[k_s * n + i_s] = 1
        w[(k_s+1) * n + (i_s+1)] = 1
        add_col(u, v, w)

    # Add schoolbook terms for remaining (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    add_col(u, v, w)

    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res['U'].shape[1])
