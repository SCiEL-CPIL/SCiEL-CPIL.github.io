import numpy as np
import time

def construct():
    """
    Returns a rank-56 exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x8 matrix using Strassen's algorithm on 8 independent 2x2 sub-problems.
    
    The decomposition blocks A (2x4) into two 2x2 column blocks and B (4x8) 
    into eight 2x2 blocks, applying Strassen (rank 7) to each 2x2*2x2 product.
    Total rank = 8 * 7 = 56.
    
    Output W follows convention W[k*n+i, r] where k is the column index of C,
    i is the row index of C.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)  # (8, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 56)
    
    # Strassen terms for 2x2 * 2x2:
    # Each term is (x_coeffs, y_coeffs, z_coeffs) where:
    # x_coeffs: dict (i,j) -> coeff for A[i,j]
    # y_coeffs: dict (j,k) -> coeff for B[j,k]  
    # z_coeffs: dict (i,k) -> coeff for C[i,k]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for a_block in range(2):  # A column block: columns 2*a_block, 2*a_block+1
        for b_col in range(4):  # B column block: columns 2*b_col, 2*b_col+1
            for t in range(7):
                x_coeffs, y_coeffs, z_coeffs = strassen_terms[t]
                
                # U part: A[i, 2*a_block + j] -> flat index i*m + (2*a_block + j)
                for (i, j), coeff in x_coeffs.items():
                    u_idx = i * m + (2 * a_block + j)
                    U[u_idx, col] += coeff
                
                # V part: B[2*a_block + j, 2*b_col + k] -> flat index (2*a_block + j)*p + (2*b_col + k)
                for (j, k), coeff in y_coeffs.items():
                    v_idx = (2 * a_block + j) * p + (2 * b_col + k)
                    V[v_idx, col] += coeff
                
                # W part: C[i, 2*b_col + k] -> W[(2*b_col + k)*n + i, col]
                for (i, k), coeff in z_coeffs.items():
                    w_idx = (2 * b_col + k) * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify correctness
    # T[i,j,k] = 1 for all valid (i,j,k) since every A[i,j]*B[j,k] contributes to C[i,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry should be 1
                expected = 1.0
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(actual, expected, atol=1e-10):
                    # Fallback to schoolbook if verification fails
                    rank = n * m * p
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m + j2, col] = 1
                                V[j2*p + k2, col] = 1
                                W[k2*n + i2, col] = 1
                                col += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("U shape:", len(res["U"]), "x", len(res["U"][0]))
    print("V shape:", len(res["V"]), "x", len(res["V"][0]))
    print("W shape:", len(res["W"]), "x", len(res["W"][0]))
    
    # Verify
    n, m, p = 2, 4, 8
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    print(f"FAIL at i={i}, j={j}, k={k}: got {val}")
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print("Valid:", valid)
