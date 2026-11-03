"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Laderman's rank-15 algorithm for 3x3 matrix multiplication.
    # Returns (a_factors, b_factors, c_coeffs)
    # a_factors: list of 15 flattened 3x3 matrices (length 9)
    # b_factors: list of 15 flattened 3x3 matrices (length 9)
    # c_coeffs: 3x3 matrix where c_coeffs[i,k] is list of (term_idx, coeff)
    def laderman_3x3():
        # Define 15 products
        # a_t[i,j] and b_t[j,k] for t=0..14
        a = []
        b = []
        
        # M1 = (a11+a12+a13)(b11+b12+b13)
        a.append([1,1,1, 0,0,0, 0,0,0])
        b.append([1,1,1, 0,0,0, 0,0,0])
        # M2 = a11*b11
        a.append([1,0,0, 0,0,0, 0,0,0])
        b.append([1,0,0, 0,0,0, 0,0,0])
        # M3 = (a21+a22+a23)*b21
        a.append([0,0,0, 1,1,1, 0,0,0])
        b.append([0,0,0, 1,0,0, 0,0,0])
        # M4 = (a31+a32+a33)*b31
        a.append([0,0,0, 0,0,0, 1,1,1])
        b.append([0,0,0, 0,0,0, 1,0,0])
        # M5 = (a11-a21+a31)(b21-b31)
        a.append([1,0,0, -1,0,0, 1,0,0])
        b.append([0,0,0, 1,0,0, -1,0,0])
        # M6 = (a11+a21-a31)(b21+b31)
        a.append([1,0,0, 1,0,0, -1,0,0])
        b.append([0,0,0, 1,0,0, 1,0,0])
        # M7 = (a11-a12)(b22-b23)
        a.append([1,-1,0, 0,0,0, 0,0,0])
        b.append([0,0,0, 0,1,-1, 0,0,0])
        # M8 = (a22-a23)(b12-b13)
        a.append([0,0,0, 0,1,-1, 0,0,0])
        b.append([0,1,-1, 0,0,0, 0,0,0])
        # M9 = (a32-a33)(b12-b13)
        a.append([0,0,0, 0,0,0, 0,1,-1])
        b.append([0,1,-1, 0,0,0, 0,0,0])
        # M10 = (a12+a22+a32)*b13
        a.append([0,1,0, 0,1,0, 0,1,0])
        b.append([0,0,1, 0,0,0, 0,0,0])
        # M11 = (a13+a23+a33)*b12
        a.append([0,0,1, 0,0,1, 0,0,1])
        b.append([0,1,0, 0,0,0, 0,0,0])
        # M12 = (a21-a11)*b12
        a.append([-1,0,0, 1,0,0, 0,0,0])
        b.append([0,1,0, 0,0,0, 0,0,0])
        # M13 = (a31-a11)*b13
        a.append([-1,0,0, 0,0,0, 1,0,0])
        b.append([0,0,1, 0,0,0, 0,0,0])
        # M14 = (a12-a22)*b23
        a.append([0,1,0, 0,-1,0, 0,0,0])
        b.append([0,0,0, 0,0,1, 0,0,0])
        # M15 = (a13-a33)*b32
        a.append([0,0,1, 0,0,0, 0,0,-1])
        b.append([0,0,0, 0,0,0, 0,1,0])
        
        # Output coefficients: C[i,k] = sum_t c_coeffs[i,k][t] * M_t
        # C11 = M1 - M2 + M5 - M6
        # C12 = M7 + M8 + M11 - M12
        # C13 = M9 + M10 + M13 - M14
        # C21 = M3 - M5 + M12 - M15
        # C22 = M6 - M1 + M14 - M15
        # C23 = M7 + M8 - M11 + M12
        # C31 = M4 - M5 + M13 - M14
        # C32 = M6 - M1 + M15 - M13
        # C33 = M9 + M10 - M14 + M15
        
        c_coeffs = [
            [(0,1),(1,-1),(4,1),(5,-1)],       # C11
            [(6,1),(7,1),(10,1),(11,-1)],      # C12
            [(8,1),(9,1),(12,1),(13,-1)],      # C13
            [(2,1),(4,-1),(11,1),(14,-1)],     # C21
            [(5,1),(0,-1),(13,1),(14,-1)],     # C22
            [(6,1),(7,1),(10,-1),(11,1)],      # C23
            [(3,1),(4,-1),(12,1),(13,-1)],     # C31
            [(5,1),(0,-1),(14,1),(12,-1)],     # C32
            [(8,1),(9,1),(13,-1),(14,1)],      # C33
        ]
        
        return a, b, c_coeffs
    
    # Verify Laderman
    la, lb, lc = laderman_3x3()
    laderman_ok = True
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # T[i,j,k] should be 1 (each A[i,j]*B[j,k] contributes to C[i,k])
                # sum_t la[t][i*3+j] * lb[t][j*3+k] * (sum of coeffs for C[i,k])
                # Actually: C[i,k] = sum_j A[i,j]B[j,k] = sum_t (sum_j la[t][i*3+j]*A[i,j]) * (sum_k' lb[t][j*3+k']*B[j,k'])
                # The tensor equation is: for all i,j,k: sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
                # where U[i,j,t] = la[t][i*3+j], V[j,k,t] = lb[t][j*3+k], W[k,i,t] = coeff for C[i,k] from term t
                actual = 0
                for t in range(15):
                    coeff = 0
                    for (ti, c) in lc[i*3+k]:
                        if ti == t:
                            coeff = c
                    actual += la[t][i*3+j] * lb[t][j*3+k] * coeff
                if actual != 1:
                    laderman_ok = False
                    break
            if not laderman_ok:
                break
        if not laderman_ok:
            break
    
    # Strassen's rank-7 algorithm for 2x2
    def strassen_2x2():
        a = []
        b = []
        # M1 = (a11+a22)(b11+b22)
        a.append([1,0, 0,1])
        b.append([1,0, 0,1])
        # M2 = (a21+a22)*b11
        a.append([0,0, 1,1])
        b.append([1,0, 0,0])
        # M3 = a11*(b12-b22)
        a.append([1,0, 0,0])
        b.append([0,1, -1,0])
        # M4 = a22*(b21-b11)
        a.append([0,0, 0,1])
        b.append([-1,0, 1,0])
        # M5 = (a11+a12)*b22
        a.append([1,1, 0,0])
        b.append([0,0, 0,1])
        # M6 = (a21-a11)*(b11+b12)
        a.append([-1,0, 1,0])
        b.append([1,1, 0,0])
        # M7 = (a12-a22)*(b21+b22)
        a.append([0,1, 0,-1])
        b.append([0,0, 1,1])
        
        c_coeffs = [
            [(0,1),(3,1),(4,-1),(6,1)],   # C11
            [(2,1),(4,1)],                # C12
            [(1,1),(3,1)],                # C21
            [(0,1),(1,-1),(2,1),(5,1)],   # C22
        ]
        return a, b, c_coeffs
    
    sa, sb, sc = strassen_2x2()
    strassen_ok = True
    for i in range(2):
        for k in range(2):
            for j in range(2):
                actual = 0
                for t in range(7):
                    coeff = 0
                    for (ti, c) in sc[i*2+k]:
                        if ti == t:
                            coeff = c
                    actual += sa[t][i*2+j] * sb[t][j*2+k] * coeff
                if actual != 1:
                    strassen_ok = False
                    break
            if not strassen_ok:
                break
        if not strassen_ok:
            break
    
    # Build the full 5x5 decomposition
    terms = []  # (u_dict, v_dict, w_idx)
    
    def add_term(u, v, w_idx):
        terms.append((dict(u), dict(v), w_idx))
    
    # Block structure:
    # A = [[A11(3x3) | A12(3x2)],
    #      [A21(2x3) | A22(2x2)]]
    # B = [[B11(3x3) | B12(3x2)],
    #      [B21(2x3) | B22(2x2)]]
    # C = [[C11(3x3) | C12(3x2)],
    #      [C21(2x3) | C22(2x2)]]
    
    # Global indices:
    # A[i,j] at row i*5+j
    # B[j,k] at row j*5+k
    # C[i,k] at row k*5+i (trace convention W[k*n+i])
    
    # C11 = A11*B11 + A12*B21
    if laderman_ok:
        # A11*B11 using Laderman
        for t in range(15):
            u = {}
            v = {}
            for i in range(3):
                for j in range(3):
                    if la[t][i*3+j] != 0:
                        u[i*5+j] = la[t][i*3+j]
            for j in range(3):
                for k in range(3):
                    if lb[t][j*3+k] != 0:
                        v[j*5+k] = lb[t][j*3+k]
            # For each (i,k) in C11, add contribution
            for i in range(3):
                for k in range(3):
                    coeff = 0
                    for (ti, c) in lc[i*3+k]:
                        if ti == t:
                            coeff = c
                    if coeff != 0:
                        add_term(u, v, k*5+i)
    else:
        # Schoolbook for A11*B11
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = {i*5+j: 1}
                    v = {j*5+k: 1}
                    add_term(u, v, k*5+i)
    
    # A12*B21 (3x2 * 2x3, schoolbook 18 terms)
    for i in range(3):
        for s in range(2):
            for k in range(3):
                u = {i*5+(3+s): 1}
                v = {(3+s)*5+k: 1}
                add_term(u, v, k*5+i)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12 (3x3 * 3x2, schoolbook 18 terms)
    for i in range(3):
        for s in range(3):
            for k in range(2):
                u = {i*5+s: 1}
                v = {s*5+(3+k): 1}
                add_term(u, v, (3+k)*5+i)
    
    # A12*B22 (3x2 * 2x2)
    if strassen_ok:
        for t in range(7):
            u = {}
            v = {}
            for i in range(3):
                for s in range(2):
                    if sa[t][i%2*2+s] != 0:  # This is wrong, need to map properly
                        pass
            # Actually A12 is 3x2, not 2x2. Strassen is for 2x2*2x2.
            # A12*B22: A12 is 3x2, B22 is 2x2. This is not a square matrix multiplication.
            # So we can't directly use Strassen here. Use schoolbook.
            pass
    # Schoolbook for A12*B22 (3*2*2 = 12 terms)
    for i in range(3):
        for s in range(2):
            for k in range(2):
                u = {i*5+(3+s): 1}
                v = {(3+s)*5+(3+k): 1}
                add_term(u, v, (3+k)*5+i)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11 (2x3 * 3x3, schoolbook 18 terms)
    for i in range(2):
        for s in range(3):
            for k in range(3):
                u = {(3+i)*5+s: 1}
                v = {s*5+k: 1}
                add_term(u, v, k*5+(3+i))
    
    # A22*B21 (2x2 * 2x3)
    # A22 is 2x2, B21 is 2x3. Not square, use schoolbook.
    for i in range(2):
        for s in range(2):
            for k in range(3):
                u = {(3+i)*5+(3+s): 1}
                v = {(3+s)*5+k: 1}
                add_term(u, v, k*5+(3+i))
    
    # C22 = A21*B12 + A22*B22
    # A21*B12 (2x3 * 3x2, schoolbook 12 terms)
    for i in range(2):
        for s in range(3):
            for k in range(2):
                u = {(3+i)*5+s: 1}
                v = {s*5+(3+k): 1}
                add_term(u, v, (3+k)*5+(3+i))
    
    # A22*B22 (2x2 * 2x2)
    if strassen_ok:
        for t in range(7):
            u = {}
            v = {}
            for i in range(2):
                for s in range(2):
                    if sa[t][i*2+s] != 0:
                        u[(3+i)*5+(3+s)] = sa[t][i*2+s]
            for s in range(2):
                for k in range(2):
                    if sb[t][s*2+k] != 0:
                        v[(3+s)*5+(3+k)] = sb[t][s*2+k]
            for i in range(2):
                for k in range(2):
                    coeff = 0
                    for (ti, c) in sc[i*2+k]:
                        if ti == t:
                            coeff = c
                    if coeff != 0:
                        add_term(u, v, (3+k)*5+(3+i))
    else:
        for i in range(2):
            for s in range(2):
                for k in range(2):
                    u = {(3+i)*5+(3+s): 1}
                    v = {(3+s)*5+(3+k): 1}
                    add_term(u, v, (3+k)*5+(3+i))
    
    rank = len(terms)
    
    # Assemble U, V, W
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col, (u, v, w_idx) in enumerate(terms):
        for idx, val in u.items():
            U[idx][col] += val
        for idx, val in v.items():
            V[idx][col] += val
        W[w_idx][col] = 1
    
    # Verify
    if time.time() < deadline:
        try:
            Ua = np.array(U, dtype=np.int64)
            Va = np.array(V, dtype=np.int64)
            Wa = np.array(W, dtype=np.int64)
            M1 = (Ua * Va.T).sum(axis=1)
            M2 = (M1 * Wa.T).sum(axis=1)
            ok = True
            for i in range(n):
                for k in range(p):
                    for s in range(m):
                        if M2[i, k, s] != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
