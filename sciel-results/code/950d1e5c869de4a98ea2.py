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
    # We split the middle dimension (4) into two 2s.
    # A = [A_0, A_1] where A_0, A_1 are 2x2.
    # B = [B_0; B_1] where B_0, B_1 are 2x8.
    # C = A_0 B_0 + A_1 B_1.
    # B_0 is 2x8, which is 4 2x2 blocks: B_00, B_01, B_02, B_03.
    # B_1 is 2x8, which is 4 2x2 blocks: B_10, B_11, B_12, B_13.
    # So we have 8 independent 2x2 * 2x2 multiplications.
    # Each uses 7 rank-1 terms. Total rank 56.

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen terms for 2x2 * 2x2
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
    # M2 = (c+d)e    -> C10 += M2, C11 -= M2
    # M3 = a(f-h)    -> C01 += M3, C11 += M3
    # M4 = d(g-e)    -> C00 += M4, C10 += M4
    # M5 = (a+b)h    -> C00 -= M5, C01 += M5
    # M6 = (c-a)(e+f)-> C11 += M6
    # M7 = (b-d)(g+h)-> C00 += M7
    
    # Indices in U (A part): i*m + j
    # Indices in V (B part): j*p + k
    # Indices in W (C part): k*n + i
    
    # We iterate over the 8 blocks.
    # Block (u, v) where u is the row-block of B (0 or 1) and v is the col-block of B (0 to 3).
    # Wait, A is 2x4. The split is on the columns of A (which are rows of B).
    # A = [A_left, A_right] (2x2 each).
    # B = [B_top; B_bottom] (2x8 each).
    # So for u in [0, 1] (row block of B, col block of A):
    #   For v in [0, 1, 2, 3] (col block of B, 2x2 each):
    #     We compute A_u * B_{u,v}
    
    col_idx = 0
    for u in range(2):  # Row block of B / Col block of A
        for v in range(4): # Col block of B (each is 2x2)
            # A_u is 2x2.
            # A indices:
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a00 = 0 * m + 2 * u
            a01 = 0 * m + 2 * u + 1
            a10 = 1 * m + 2 * u
            a11 = 1 * m + 2 * u + 1
            
            # B_{u,v} is 2x2.
            # B indices:
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b00 = (2 * u) * p + 2 * v
            b01 = (2 * u) * p + 2 * v + 1
            b10 = (2 * u + 1) * p + 2 * v
            b11 = (2 * u + 1) * p + 2 * v + 1
            
            # C output is 2x2 block at column 2v, 2v+1.
            # C indices (W): k*n + i
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            c00 = (2 * v) * n + 0
            c01 = (2 * v + 1) * n + 0
            c10 = (2 * v) * n + 1
            c11 = (2 * v + 1) * n + 1
            
            # Term 1: (a00 + a11)(b00 + b11)
            U[a00, col_idx] += 1
            U[a11, col_idx] += 1
            V[b00, col_idx] += 1
            V[b11, col_idx] += 1
            W[c00, col_idx] += 1
            W[c11, col_idx] += 1
            col_idx += 1
            
            # Term 2: (a10 + a11)b00
            U[a10, col_idx] += 1
            U[a11, col_idx] += 1
            V[b00, col_idx] += 1
            W[c10, col_idx] += 1
            W[c11, col_idx] -= 1
            col_idx += 1
            
            # Term 3: a00(b01 - b11)
            U[a00, col_idx] += 1
            V[b01, col_idx] += 1
            V[b11, col_idx] -= 1
            W[c01, col_idx] += 1
            W[c11, col_idx] += 1
            col_idx += 1
            
            # Term 4: a11(b10 - b00)
            U[a11, col_idx] += 1
            V[b10, col_idx] += 1
            V[b00, col_idx] -= 1
            W[c00, col_idx] += 1
            W[c10, col_idx] += 1
            col_idx += 1
            
            # Term 5: (a00 + a01)b11
            U[a00, col_idx] += 1
            U[a01, col_idx] += 1
            V[b11, col_idx] += 1
            W[c00, col_idx] -= 1
            W[c01, col_idx] += 1
            col_idx += 1
            
            # Term 6: (a10 - a00)(b00 + b01)
            U[a10, col_idx] += 1
            U[a00, col_idx] -= 1
            V[b00, col_idx] += 1
            V[b01, col_idx] += 1
            W[c11, col_idx] += 1
            col_idx += 1
            
            # Term 7: (a01 - a11)(b10 + b11)
            U[a01, col_idx] += 1
            U[a11, col_idx] -= 1
            V[b10, col_idx] += 1
            V[b11, col_idx] += 1
            W[c00, col_idx] += 1
            col_idx += 1

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if Strassen construction failed
        U, V, W, rank = schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
