"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    terms = [
        ([1,0,0,0], [1,0,0,1], [1,0,0,1]),
        ([1,0,0,1], [0,0,0,1], [0,0,1,0]),
        ([1,0,1,0], [1,0,0,0], [0,1,0,0]),
        ([0,0,1,1], [0,0,0,1], [1,0,0,0]),
        ([1,0,0,1], [1,0,0,1], [0,1,0,0]),
        ([1,0,-1,0], [1,0,0,1], [0,0,1,0]),
        ([0,-1,0,1], [1,1,0,0], [0,0,0,1]),
    ]
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    i_pairs = [(0, 1)]
    i_singles = [2]
    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    k_pairs = [(0, 1), (2, 3)]
    k_singles = [4]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen_terms = _strassen_2x2x2()
    
    for (i0, i1) in i_pairs:
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                for (u, v, w) in strassen_terms:
                    u_full = [0] * (n * m)
                    v_full = [0] * (m * p)
                    w_full = [0] * (p * n)
                    
                    u_full[i0 * m + j0] += u[0]
                    u_full[i0 * m + j1] += u[1]
                    u_full[i1 * m + j0] += u[2]
                    u_full[i1 * m + j1] += u[3]
                    
                    v_full[j0 * p + k0] += v[0]
                    v_full[j0 * p + k1] += v[1]
                    v_full[j1 * p + k0] += v[2]
                    v_full[j1 * p + k1] += v[3]
                    
                    w_full[k0 * n + i0] += w[0]
                    w_full[k0 * n + i1] += w[1]
                    w_full[k1 * n + i0] += w[2]
                    w_full[k1 * n + i1] += w[3]
                    
                    U_terms.append(u_full)
                    V_terms.append(v_full)
                    W_terms.append(w_full)
    
    # Schoolbook for all remaining (i,j,k)
    strassen_ik = set()
    for (i0, i1) in i_pairs:
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            strassen_ik.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_ik:
                    continue
                u_full = [0] * (n * m)
                v_full = [0] * (m * p)
                w_full = [0] * (p * n)
                u_full[i * m + j] = 1
                v_full[j * p + k] = 1
                w_full[k * n + i] = 1
                U_terms.append(u_full)
                V_terms.append(v_full)
                W_terms.append(w_full)
    
    rank = len(U_terms)
    
    # Verify
    import numpy as np
    U_np = np.array(U_terms, dtype=float)
    V_np = np.array(V_terms, dtype=float)
    W_np = np.array(W_terms, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        for i in range(n):
            for k in range(p):
                inner = sum(U_np[i*m+j, c] * V_np[j*p+k, c] for j in range(m))
                C_computed[i, k] += inner * W_np[k*n+i, c]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        rank = n * m * p
        U_terms = [[0] * rank for _ in range(n * m)]
        V_terms = [[0] * rank for _ in range(m * p)]
        W_terms = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_terms[i*m+j][col] = 1
                    V_terms[j*p+k][col] = 1
                    W_terms[k*n+i][col] = 1
                    col += 1
    
    return {"U": U_terms, "V": V_terms, "W": W_terms}
# EVOLVE-BLOCK-END

