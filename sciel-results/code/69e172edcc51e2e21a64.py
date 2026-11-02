import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Schoolbook decomposition for 2x2x3 (rank 12)
    def schoolbook_223():
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = {(i,j): 1}
                    b = {(j,k): 1}
                    c = {(k,i): 1}
                    terms.append((a, b, c))
        return terms

    def build_kronecker(n1, m1, p1, terms1, n2, m2, p2, terms2):
        """Build Kronecker product of two tensor decompositions."""
        rank = len(terms1) * len(terms2)
        U = np.zeros((n1*n2, m1*m2, rank))
        V = np.zeros((m1*m2, p1*p2, rank))
        W = np.zeros((p1*p2, n1*n2, rank))
        
        col = 0
        for t1 in terms1:
            a1, b1, c1 = t1
            for t2 in terms2:
                a2, b2, c2 = t2
                for (i1, j1), c_a1 in a1.items():
                    for (i2, j2), c_a2 in a2.items():
                        U[i1*n2+i2, j1*m2+j2, col] += c_a1 * c_a2
                for (j1, k1), c_b1 in b1.items():
                    for (j2, k2), c_b2 in b2.items():
                        V[j1*m2+j2, k1*p2+k2, col] += c_b1 * c_b2
                for (k1, i1), c_c1 in c1.items():
                    for (k2, i2), c_c2 in c2.items():
                        W[k1*p2+k2, i1*n2+i2, col] += c_c1 * c_c2
                col += 1
        return U, V, W

    def verify_decomp(U, V, W, n, m, p):
        """Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def to_list(mat):
        """Convert numpy array to list of lists."""
        return mat.tolist()

    # Strategy: Kronecker product of 2x2x2 (Strassen, rank 7) and 2x2x3 (schoolbook, rank 12)
    # This gives 4x4x6 with rank 7*12 = 84
    terms_223 = schoolbook_223()
    U_k, V_k, W_k = build_kronecker(
        2, 2, 2, strassen_terms,
        2, 2, 3, terms_223
    )
    
    U_flat = U_k.reshape(-1, U_k.shape[2])
    V_flat = V_k.reshape(-1, V_k.shape[2])
    W_flat = W_k.reshape(-1, W_k.shape[2])
    
    if verify_decomp(U_flat, V_flat, W_flat, n, m, p):
        return {
            "U": to_list(U_flat),
            "V": to_list(V_flat),
            "W": to_list(W_flat)
        }
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
