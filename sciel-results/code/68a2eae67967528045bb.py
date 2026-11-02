import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Rank 56 solution based on 8 blocks of Strassen (2x2x2)
    # A is split into 2 blocks of 2 columns.
    # B is split into 4 blocks of 2 columns.
    # Each block multiplication is 2x2 * 2x2 -> 2x2, rank 7.
    # Total rank 2 * 4 * 7 = 56.
    
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11       -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22)       -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11)       -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22       -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # Indices in local 2x2 block:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: (i, j) -> index i*2+j
        # v: (j, k) -> index j*2+k
        # w: (k, i) -> index k*2+i  (Trace convention for C[i,k] is W[k*n+i])
        # In local 2x2 block, C index for output [i, k] is k*2+i.
        
        # M1: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        
        # M2: (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # C10 +=, C11 -=
        
        # M3: A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # C01 +=, C11 +=
        
        # M4: A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # C00 +=, C10 +=
        
        # M5: (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # C00 -=, C01 +=
        
        # M6: (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # C11 +=
        
        # M7: (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # C00 +=
    ]
    
    col = 0
    for u_block in range(2):  # A column block index
        for v_block in range(4):  # B column block index
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                
                # U: A[i, 2*u_block + j'] -> index i*m + (2*u_block + j')
                for (i_prime, j_prime), coeff in u_coeffs.items():
                    U[i_prime * m + (2*u_block + j_prime), col] += coeff
                
                # V: B[2*u_block + j', 2*v_block + k'] -> index (2*u_block + j')*p + (2*v_block + k')
                for (j_prime, k_prime), coeff in v_coeffs.items():
                    V[(2*u_block + j_prime) * p + (2*v_block + k_prime), col] += coeff
                
                # W: C[i', 2*v_block + k'] -> W index (2*v_block + k')*n + i'
                for (i_prime, k_prime), coeff in w_coeffs.items():
                    W[(2*v_block + k_prime) * n + i_prime, col] += coeff
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Verification
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
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
