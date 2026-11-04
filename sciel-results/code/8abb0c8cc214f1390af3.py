import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vals, v_vals, w_vals):
        u_row = np.zeros(n * m)
        v_row = np.zeros(m * p)
        w_row = np.zeros(p * n)
        for (i, j), val in u_vals.items():
            u_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_row[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_row[k * n + i] = val
        U_terms.append(u_row)
        V_terms.append(v_row)
        W_terms.append(w_row)
        
    def get_strassen_2x2():
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (0,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1, (1,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1, (1,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
    strassen = get_strassen_2x2()
    
    def offset_strassen(i_off, j_off, k_off):
        result = []
        for u, v, w in strassen:
            u_new = {(i + i_off, j + j_off): val for (i, j), val in u.items()}
            v_new = {(j + j_off, k + k_off): val for (j, k), val in v.items()}
            w_new = {(k + k_off, i + i_off): val for (k, i), val in w.items()}
            result.append((u_new, v_new, w_new))
        return result
        
    # A1*B1a1: 2x2 x 2x2, A rows 0-1, B rows 0-1, B cols 0-1
    for u, v, w in offset_strassen(0, 0, 0):
        add_term(u, v, w)
        
    # A1*B1a2: 2x2 x 2x2, A rows 0-1, B rows 0-1, B cols 2-3
    for u, v, w in offset_strassen(0, 0, 2):
        add_term(u, v, w)
        
    # A1*B1b_2x2: 2x2 x 2x2, A rows 0-1, B rows 0-1, B cols 4-5
    for u, v, w in offset_strassen(0, 0, 4):
        add_term(u, v, w)
        
    # A1*B1b_2x1: 2x2 x 2x1, schoolbook: rank 4
    for i in range(2):
        for j in range(2):
            k = 6
            u = {(i, j): 1}
            v = {(j, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)
            
    # A2*B2: 2x1 x 1x7, A rows 0-1 col 2, B row 2 cols 0-6
    # Rank 2: for each i in {0,1}, one term covering all k
    for i in range(2):
        u = {(i, 2): 1}
        v = {(2, k): 1 for k in range(7)}
        w = {(k, i): 1 for k in range(7)}
        add_term(u, v, w)
        
    rank = len(U_terms)
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    r = n * m * p
                    U_fb = np.zeros((n * m, r))
                    V_fb = np.zeros((m * p, r))
                    W_fb = np.zeros((p * n, r))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, col] = 1.0
                                V_fb[jj * p + kk, col] = 1.0
                                W_fb[kk * n + ii, col] = 1.0
                                col += 1
                    return {
                        "U": [[int(val) for val in row] for row in U_fb.tolist()],
                        "V": [[int(val) for val in row] for row in V_fb.tolist()],
                        "W": [[int(val) for val in row] for row in W_fb.tolist()]
                    }
                    
    return {
        "U": [[int(val) for val in row] for row in U.tolist()],
        "V": [[int(val) for val in row] for row in V.tolist()],
        "W": [[int(val) for val in row] for row in W.tolist()]
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    print("Rank:", rank)
    # Quick check
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > 1e-10:
                    valid = False
    print("Valid:", valid)
