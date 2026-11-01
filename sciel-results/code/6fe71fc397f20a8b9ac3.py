import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition for 2x2x2
    # We are multiplying A (2x4) by B (4x8).
    # Split middle dimension (4) into two 2s.
    # A = [A_0, A_1] (2x2 each)
    # B = [B_0; B_1] (2x8 each)
    # C = A_0 B_0 + A_1 B_1
    # B_0 is 2x8 -> 4 blocks of 2x2.
    # B_1 is 2x8 -> 4 blocks of 2x2.
    # Total 8 blocks of 2x2 * 2x2.
    # Each block uses 7 rank-1 terms.
    # Total rank 56.

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen terms definition: (u_indices, v_indices, w_indices)
    # u_indices: list of (index, coeff)
    # v_indices: list of (index, coeff)
    # w_indices: list of (index, coeff)
    
    def get_strassen_terms(a_idx, b_idx, c_idx):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        return [
            # M1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
            ([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)]),
            # M2 = (a10+a11)b00 -> c10+=1, c11-=1
            ([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)]),
            # M3 = a00(b01-b11) -> c01+=1, c11+=1
            ([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)]),
            # M4 = a11(b10-b00) -> c00+=1, c10+=1
            ([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)]),
            # M5 = (a00+a01)b11 -> c00-=1, c01+=1
            ([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)]),
            # M6 = (a10-a00)(b00+b01) -> c11+=1
            ([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)]),
            # M7 = (a01-a11)(b10+b11) -> c00+=1
            ([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])
        ]

    col_idx = 0
    for u in range(2):  # Row block of B / Col block of A
        for v in range(4): # Col block of B (each is 2x2)
            # A indices
            a_idx = [
                0 * m + 2 * u,
                0 * m + 2 * u + 1,
                1 * m + 2 * u,
                1 * m + 2 * u + 1
            ]
            # B indices
            b_idx = [
                (2 * u) * p + 2 * v,
                (2 * u) * p + 2 * v + 1,
                (2 * u + 1) * p + 2 * v,
                (2 * u + 1) * p + 2 * v + 1
            ]
            # C indices (W): k*n + i
            c_idx = [
                (2 * v) * n + 0,
                (2 * v + 1) * n + 0,
                (2 * v) * n + 1,
                (2 * v + 1) * n + 1
            ]
            
            terms = get_strassen_terms(a_idx, b_idx, c_idx)
            
            for u_terms, v_terms, w_terms in terms:
                for idx, coeff in u_terms:
                    U[idx, col_idx] += coeff
                for idx, coeff in v_terms:
                    V[idx, col_idx] += coeff
                for idx, coeff in w_terms:
                    W[idx, col_idx] += coeff
                col_idx += 1

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        U, V, W, rank = schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
