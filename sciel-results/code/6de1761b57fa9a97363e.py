import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 28 decomposition
    rank = 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Part 1: 1x2 x 2x4 decomposition (Rank 4)
    # U1: (2, 4) -> indices j1, r1
    # V1: (8, 4) -> indices (j1, k1), r1
    # W1: (4, 4) -> indices k1, r1
    U1 = np.ones((2, 4))
    V1 = np.zeros((8, 4))
    W1 = np.zeros((4, 4))
    
    for j1 in range(2):
        for k1 in range(4):
            V1[j1 * 4 + k1, k1] = 1.0
        U1[j1, :] = 1.0
    for k1 in range(4):
        W1[k1, k1] = 1.0
        
    # Part 2: 2x2 x 2x2 decomposition (Strassen, Rank 7)
    # U2: (4, 7) -> indices (i, j2), r2
    # V2: (4, 7) -> indices (j2, k2), r2
    # W2: (4, 7) -> indices (k2, i), r2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms for 2x2
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for r2 in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_terms[r2]
        for local_idx, coeff in u_coeffs:
            U2[local_idx, r2] += coeff
        for local_idx, coeff in v_coeffs:
            V2[local_idx, r2] += coeff
        for local_idx, coeff in w_coeffs:
            W2[local_idx, r2] += coeff
            
    # Combine using Kronecker product logic
    # U total: (n*m, rank) = (8, 28)
    # Index: i in 0..1, j in 0..3. j = 2*j1 + j2.
    # U[i, j, r1*7 + r2] = U1[j1, r1] * U2[i, j2, r2]
    for i in range(2):
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for r1 in range(4):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U[i * 4 + j, r] = U1[j1, r1] * U2[i * 2 + j2, r2]
                    
    # V total: (m*p, rank) = (32, 28)
    # Index: j in 0..3, k in 0..7.
    # j = 2*j1 + j2. k = 2*k1 + k2.
    # V[j, k, r1*7 + r2] = V1[j1, k1, r1] * V2[j2, k2, r2]
    for j in range(4):
        j1 = j // 2
        j2 = j % 2
        for k in range(8):
            k1 = k // 2
            k2 = k % 2
            for r1 in range(4):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V[j * 8 + k, r] = V1[j1 * 4 + k1, r1] * V2[j2 * 2 + k2, r2]
                    
    # W total: (p*n, rank) = (16, 28)
    # Index: k in 0..7, i in 0..1.
    # k = 2*k1 + k2.
    # W[k, i, r1*7 + r2] = W1[k1, r1] * W2[i, k2, r2]
    # W index convention: W[k*n + i, rank].
    for k in range(8):
        k1 = k // 2
        k2 = k % 2
        for i in range(2):
            for r1 in range(4):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W[k * 2 + i, r] = W1[k1, r1] * W2[i * 2 + k2, r2]

    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to rank 56 or 64 if something went wrong
        # For safety, return the previous known good 56 or just 64
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
