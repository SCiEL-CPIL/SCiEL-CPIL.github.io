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

    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in 2x2 block:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Strassen patterns: (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # u_indices are indices into the 4-element A vector [A11, A12, A21, A22]
    # v_indices are indices into the 4-element B vector [B11, B12, B21, B22]
    # w_indices are indices into the 4-element C vector [C11, C12, C21, C22]
    
    strassen_blocks = [
        # M1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # M2
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # M3
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        # M4
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        # M5
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        # M6
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # M7
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Define the 8 (u, v) block pairs
    # u is the column block of A (2x2), v is the row/col block of B?
    # In the original logic:
    # u_idx in 0..1 corresponds to A column block.
    # v_idx in 0..3 corresponds to B column block.
    # But B rows are determined by u_idx.
    # A block u: cols 2u, 2u+1.
    # B block for u, v: rows 2u, 2u+1; cols 2v, 2v+1.
    
    block_configs = []
    for u in range(2):
        for v in range(4):
            # A indices in global U (size nm=8)
            # A is 2x4. Row 0,1. Cols 2u, 2u+1.
            a_ids = [0 * m + 2 * u, 0 * m + 2 * u + 1, 1 * m + 2 * u, 1 * m + 2 * u + 1]
            
            # B indices in global V (size mp=32)
            # B is 4x8. Rows 2u, 2u+1. Cols 2v, 2v+1.
            b_ids = [(2 * u) * p + 2 * v, (2 * u) * p + 2 * v + 1, (2 * u + 1) * p + 2 * v, (2 * u + 1) * p + 2 * v + 1]
            
            # C indices in global W (size pn=16)
            # C is 2x8. Cols 2v, 2v+1.
            # W index = k * n + i.
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            c_ids = [(2 * v) * n + 0, (2 * v + 1) * n + 0, (2 * v) * n + 1, (2 * v + 1) * n + 1]
            
            base_col = 7 * (4 * u + v)
            block_configs.append((a_ids, b_ids, c_ids, base_col))
            
    for a_ids, b_ids, c_ids, base_col in block_configs:
        for s_idx, pattern in enumerate(strassen_blocks):
            col = base_col + s_idx
            
            for idx, coeff in pattern['u']:
                U[a_ids[idx], col] += coeff
            for idx, coeff in pattern['v']:
                V[b_ids[idx], col] += coeff
            for idx, coeff in pattern['w']:
                W[c_ids[idx], col] += coeff
                
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
        U, V, W, rank = schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
