import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    Uses Strassen's algorithm applied to three 2x2 blocks of the 2x6 matrix.
    Rank = 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 21
    rank = 21
    
    # Initialize matrices
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen coefficients for 2x2 multiplication
    # u_r: coefficients of A (indices 00, 01, 10, 11 -> 0, 1, 2, 3)
    # v_r: coefficients of B (indices 00, 01, 10, 11 -> 0, 1, 2, 3)
    # w_r: coefficients of C (indices 00, 01, 10, 11 -> 0, 1, 2, 3)
    # Note: W index in tensor is k*n+i.
    # C_00 -> k=0, i=0 -> idx 0
    # C_01 -> k=1, i=0 -> idx 2
    # C_10 -> k=0, i=1 -> idx 1
    # C_11 -> k=1, i=1 -> idx 3
    
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [A00, A01, A10, A11]
        # v: [B00, B01, B10, B11]
        # w: [C00, C01, C10, C11] mapped to tensor W indices [0, 2, 1, 3]
        
        # r=1: P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # w[0]=1(C00), w[1]=0(C10), w[2]=0(C01), w[3]=1(C11)
        # r=2: P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # w[2]=1(C10)
        # r=3: P3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # w[2]=1(C01)
        # r=4: P4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # w[0]=1(C00), w[2]=1(C10)
        # r=5: P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # w[0]=-1(C00), w[2]=1(C01)
        # r=6: P6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # w[3]=1(C11)
        # r=7: P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # w[0]=1(C00)
    ]
    
    # Fill U, V, W for 3 blocks
    for q in range(3):
        for r_local in range(7):
            r_global = q * 7 + r_local
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[r_local]
            
            # Fill U (independent of block q)
            for i in range(2):
                for j in range(2):
                    idx_u = i * 2 + j
                    U[idx_u][r_global] = u_coeffs[i * 2 + j]
            
            # Fill V (depends on block q)
            for j in range(2):
                for k_local in range(2):
                    idx_v = j * 6 + (2 * q + k_local)
                    V[idx_v][r_global] = v_coeffs[j * 2 + k_local]
            
            # Fill W (depends on block q)
            for k_local in range(2):
                for i in range(2):
                    # W index is k*n + i
                    # k_global = 2*q + k_local
                    idx_w = (2 * q + k_local) * 2 + i
                    # w_coeffs index is k_local*2 + i? 
                    # My w_coeffs list was [C00, C01, C10, C11]
                    # C00: k=0, i=0 -> idx 0
                    # C01: k=1, i=0 -> idx 2
                    # C10: k=0, i=1 -> idx 1
                    # C11: k=1, i=1 -> idx 3
                    # So w_coeffs[k_local*2 + i] is NOT correct mapping to list [C00, C01, C10, C11]
                    # List index 0: C00 (k=0, i=0)
                    # List index 1: C01 (k=1, i=0) -> Wait, I defined w_coeffs as [C00, C01, C10, C11]
                    # But in the tuple above, I put values at indices 0,1,2,3 corresponding to C00, C10, C01, C11?
                    # Let's re-verify the w_coeffs tuples in strassen_terms.
                    
                    # r=1: [1, 0, 0, 1]
                    # C00=1, C10=0, C01=0, C11=1.
                    # My tuple was [1, 0, 0, 1].
                    # If index 0 is C00, index 1 is C10, index 2 is C01, index 3 is C11.
                    # Then for k_local=0, i=0 (C00), coeff is w_coeffs[0]=1.
                    # For k_local=0, i=1 (C10), coeff is w_coeffs[1]=0.
                    # For k_local=1, i=0 (C01), coeff is w_coeffs[2]=0.
                    # For k_local=1, i=1 (C11), coeff is w_coeffs[3]=1.
                    
                    # So the mapping is:
                    # w_coeffs index = k_local * 2 + i?
                    # k_local=0, i=0 -> 0.
                    # k_local=0, i=1 -> 1.
                    # k_local=1, i=0 -> 2.
                    # k_local=1, i=1 -> 3.
                    # Yes, this matches the order [C00, C10, C01, C11].
                    
                    W[idx_w][r_global] = w_coeffs[k_local * 2 + i]

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 2, 6
    rank = U.shape[1]
    
    # Build T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Check
    max_err = 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.dot(U[a], V[b] * W[c])
                err = abs(val - T[a,b,c])
                if err > max_err:
                    max_err = err
    print(f"Rank: {rank}, Max Error: {max_err}")
