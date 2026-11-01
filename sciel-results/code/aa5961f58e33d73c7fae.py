import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 115.0

def construct():
    n, m, p = 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Rank: 4 column-pairs * 2 Strassen blocks * 7 factors = 56
    rank = 56
    
    U = np.zeros((A_dim, rank))
    V = np.zeros((B_dim, rank))
    W = np.zeros((C_dim, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x11+x22)*(y11+y22)
    # M2 = (x21+x22)*y11
    # M3 = x11*(y12-y22)
    # M4 = x22*(y21-y11)
    # M5 = (x11+x12)*y22
    # M6 = (x21-x11)*(y11+y12)
    # M7 = (x12-x22)*(y21+y22)
    # 
    # Result[0,0] = M1 + M4 - M5 + M7
    # Result[0,1] = M3 + M5
    # Result[1,0] = M2 + M4
    # Result[1,1] = M1 - M2 + M3 + M6
    
    # Strassen factor definitions: (x_factors, y_factors, result_coeffs)
    # Each factor is a linear combination of x elements and y elements
    # x = [[x00, x01], [x10, x11]] (row-major: x00, x01, x10, x11)
    # y = [[y00, y01], [y10, y11]]
    
    strassen_factors = [
        # (x_coeffs, y_coeffs, result_coeffs for [r0c0, r0c1, r1c0, r1c1])
        # M1 = (x00+x11)*(y00+y11)
        (([1,0,0,1], [1,0,0,1]), (0,0,0,1)),  # placeholder, will define properly
    ]
    
    # Let me define each factor explicitly
    # Factor 0: M1 = (x00+x11)*(y00+y11)
    # x part: x00 + x11 -> coeffs [1, 0, 0, 1] on [x00,x01,x10,x11]
    # y part: y00 + y11 -> coeffs [1, 0, 0, 1] on [y00,y01,y10,y11]
    # Result contribution: [1, 0, 0, 1] for [r0c0, r0c1, r1c0, r1c1]
    # Actually result: r0c0 += 1, r1c1 += 1
    
    # Factor 1: M2 = (x10+x11)*y00
    # x: x10 + x11 -> [0, 0, 1, 1]
    # y: y00 -> [1, 0, 0, 0]
    # Result: r1c0 += 1
    
    # Factor 2: M3 = x00*(y01-y11)
    # x: x00 -> [1, 0, 0, 0]
    # y: y01 - y11 -> [0, 1, 0, -1]
    # Result: r0c1 += 1
    
    # Factor 3: M4 = x11*(y10-y00)
    # x: x11 -> [0, 0, 0, 1]
    # y: y10 - y00 -> [-1, 0, 1, 0]
    # Result: r0c0 += 1, r1c0 += 1
    
    # Factor 4: M5 = (x00+x01)*y11
    # x: x00 + x01 -> [1, 1, 0, 0]
    # y: y11 -> [0, 0, 0, 1]
    # Result: r0c0 -= 1, r0c1 += 1
    
    # Factor 5: M6 = (x10-x00)*(y00+y01)
    # x: x10 - x00 -> [-1, 0, 1, 0]
    # y: y00 + y01 -> [1, 1, 0, 0]
    # Result: r1c1 += 1
    
    # Factor 6: M7 = (x01-x11)*(y10+y11)
    # x: x01 - x11 -> [0, 1, 0, -1]
    # y: y10 + y11 -> [0, 0, 1, 1]
    # Result: r0c0 += 1
    
    # Define factors: (x_coeffs[4], y_coeffs[4], result_coeffs[4])
    # result_coeffs for [r0c0, r0c1, r1c0, r1c1]
    factors = [
        # M1
        (np.array([1, 0, 0, 1], dtype=float), 
         np.array([1, 0, 0, 1], dtype=float), 
         np.array([1, 0, 0, 1], dtype=float)),
        # M2
        (np.array([0, 0, 1, 1], dtype=float), 
         np.array([1, 0, 0, 0], dtype=float), 
         np.array([0, 0, 1, 0], dtype=float)),
        # M3
        (np.array([1, 0, 0, 0], dtype=float), 
         np.array([0, 1, 0, -1], dtype=float), 
         np.array([0, 1, 0, 0], dtype=float)),
        # M4
        (np.array([0, 0, 0, 1], dtype=float), 
         np.array([-1, 0, 1, 0], dtype=float), 
         np.array([1, 0, 1, 0], dtype=float)),
        # M5
        (np.array([1, 1, 0, 0], dtype=float), 
         np.array([0, 0, 0, 1], dtype=float), 
         np.array([-1, 1, 0, 0], dtype=float)),
        # M6
        (np.array([-1, 0, 1, 0], dtype=float), 
         np.array([1, 1, 0, 0], dtype=float), 
         np.array([0, 0, 0, 1], dtype=float)),
        # M7
        (np.array([0, 1, 0, -1], dtype=float), 
         np.array([0, 0, 1, 1], dtype=float), 
         np.array([1, 0, 0, 0], dtype=float)),
    ]
    
    # For each column-pair k (0..3) and each Strassen block (A1 or A2):
    # A1 is columns 0-1 of A, A2 is columns 2-3 of A
    # B1k is rows 0-1, cols 2k-2k+1 of B
    # B2k is rows 2-3, cols 2k-2k+1 of B
    
    for k in range(4):  # 4 column-pairs
        for block in range(2):  # block 0 = A1*B1k, block 1 = A2*B2k
            for f in range(7):  # 7 Strassen factors
                r_idx = k * 2 + block * 7 + f  # rank index
                
                x_coeffs, y_coeffs, res_coeffs = factors[f]
                
                # x is the 2x2 block of A
                if block == 0:
                    # A1: rows 0-1, cols 0-1
                    # A[i, j] where i in 0..1, j in 0..1
                    a_base_i = 0
                    a_base_j = 0
                else:
                    # A2: rows 0-1, cols 2-3
                    a_base_i = 0
                    a_base_j = 2
                
                # y is the 2x2 block of B
                if block == 0:
                    # B1k: rows 0-1, cols 2k-2k+1
                    b_base_i = 0
                    b_base_j = 2 * k
                else:
                    # B2k: rows 2-3, cols 2k-2k+1
                    b_base_i = 2
                    b_base_j = 2 * k
                
                # Fill U: A has 2 rows, 4 cols, indexed as [i*4 + j]
                for di in range(2):
                    for dj in range(2):
                        val = x_coeffs[di * 2 + dj]
                        if val != 0:
                            a_idx = (a_base_i + di) * 4 + (a_base_j + dj)
                            U[a_idx, r_idx] += val
                
                # Fill V: B has 4 rows, 8 cols, indexed as [i*8 + j]
                for di in range(2):
                    for dj in range(2):
                        val = y_coeffs[di * 2 + dj]
                        if val != 0:
                            b_idx = (b_base_i + di) * 8 + (b_base_j + dj)
                            V[b_idx, r_idx] += val
                
                # Fill W: C has 2 rows, 8 cols, indexed as [k*2 + i] per convention W[k*n+i]
                # C[i, col] where col = 2k + dj
                for di in range(2):
                    for dj in range(2):
                        val = res_coeffs[di * 2 + dj]
                        if val != 0:
                            c_row = di
                            c_col = 2 * k + dj
                            c_idx = c_col * 2 + c_row  # W[k*n+i] = W[col*2 + row]
                            W[c_idx, r_idx] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
