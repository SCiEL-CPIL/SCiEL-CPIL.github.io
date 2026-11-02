import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen decomposition for 2x2x2
    # We decompose the 2x4 by 4x8 multiplication into 8 independent 2x2x2 blocks.
    # A is 2x4, split into columns (2a, 2a+1) for a in 0..1
    # B is 4x8, split into rows (2a, 2a+1) for a in 0..1 and columns (2b, 2b+1) for b in 0..3
    # This gives 2 * 4 = 8 blocks.
    # Each block is a 2x2x2 multiplication, which has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen terms for 2x2x2
    # Indices for A: (i', j'), B: (j', k'), C: (i', k')
    # i', j', k' in {0, 1}
    strassen_terms = [
        # t=0: M1 = (A00+A11)(B00+B11)
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1: M2 = (A10+A11)B00
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2: M3 = A00(B01-B11)
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3: M4 = A11(B10-B00)
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4: M5 = (A00+A01)B11
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5: M6 = (A10-A00)(B00+B01)
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6: M7 = (A01-A11)(B10+B11)
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    col = 0
    for a in range(2):  # A block column index
        for b in range(4):  # B block column index
            for t in range(7):
                term = strassen_terms[t]
                
                # U part: A[i, 2a+j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx, col] += coeff
                    
                # V part: B[2a+j', 2b+k']
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx, col] += coeff
                    
                # W part: C[2b+k', i'] -> W index is k*n + i
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx, col] += coeff
                    
                col += 1

    # Fast verification: check a sample of entries
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        # Fallback to schoolbook rank 64
                        rank_fb = n * m * p
                        U_fb = np.zeros((nm, rank_fb))
                        V_fb = np.zeros((mp, rank_fb))
                        W_fb = np.zeros((pn, rank_fb))
                        for jj in range(m):
                            for kk in range(p):
                                idx = jj * p + kk
                                U_fb[jj * n + 0, idx] = 0  # placeholder
                        # Actually build proper schoolbook
                        U_fb = np.zeros((nm, rank_fb))
                        V_fb = np.zeros((mp, rank_fb))
                        W_fb = np.zeros((pn, rank_fb))
                        idx = 0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    U_fb[ii * m + jj, idx] = 1.0
                                    V_fb[jj * p + kk, idx] = 1.0
                                    W_fb[kk * n + ii, idx] = 1.0
                                    idx += 1
                        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
