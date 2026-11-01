"""Matrix multiplication 2x4 x 4x4 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def build_strassen_28():
        """Build rank-28 decomposition using Strassen on 2x2 blocks."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_U = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):1},
            2: {(0,0):1},
            3: {(1,1):1},
            4: {(0,0):1, (0,1):1},
            5: {(1,0):1, (0,0):-1},
            6: {(0,1):1, (1,1):-1},
        }
        strassen_V = {
            0: {(0,0):1, (1,1):1},
            1: {(0,0):1},
            2: {(1,0):1, (1,1):-1},
            3: {(0,1):1, (0,0):-1},
            4: {(1,1):1},
            5: {(0,0):1, (0,1):1},
            6: {(1,0):1, (1,1):1},
        }
        strassen_W = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):-1},
            2: {(0,1):1, (1,1):1},
            3: {(0,0):1, (1,0):1},
            4: {(0,0):-1, (0,1):1},
            5: {(1,1):1},
            6: {(0,0):1},
        }
        
        col = 0
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff
        col += 7
        
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + (ccol + 2)][c] += coeff
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff
        col += 7
        
        return {"U": U, "V": V, "W": W}

    def build_rank_27_algebraic():
        """
        Build a rank-27 decomposition using algebraic structure.
        
        Key insight: 2x4*4x4 can be decomposed as:
        C = A * B where A is 2x4, B is 4x4
        
        Write A = [a0 a1 a2 a3] (row 0), [b0 b1 b2 b3] (row 1)
        Write B as columns: B = [B0 B1 B2 B3] where Bj is 4x1
        
        C[0,k] = a0*B0[k] + a1*B1[k] + a2*B2[k] + a3*B3[k]
        C[1,k] = b0*B0[k] + b1*B1[k] + b2*B2[k] + b3*B3[k]
        
        We can use a structured decomposition that exploits the fact that
        we only have 2 rows of A. 
        
        Strategy: Use 14 terms for a "half-Strassen" on the combined structure,
        plus 13 more terms, with one term shared/merged.
        
        Actually, let's use a direct construction:
        Group B into pairs: (B0,B1) and (B2,B3)
        
        For the left half (B0,B1):
        C[0,0] = a0*B0[0] + a1*B1[0] + a2*B2[0] + a3*B3[0]
        C[0,1] = a0*B0[1] + a1*B1[1] + a2*B2[1] + a3*B3[1]
        
        Use Strassen-like on the 2x2 block formed by (a0,a1) and (B0,B1):
        M1 = (a0+a1)(B0+B1)  [gives a0*B0 + a0*B1 + a1*B0 + a1*B1]
        M2 = a0*B0
        M3 = a1*B1
        M4 = (a0-a1)(B0-B1) [gives a0*B0 - a0*B1 - a1*B0 + a1*B1]
        
        From these: a0*B0 = (M1+M2+M3+M4)/4... this gets complicated.
        
        Let me try a different approach: use the fact that for 2x4*4x4,
        we can achieve rank 27 by a specific algebraic identity.
        
        Construction:
        Let's define 27 bilinear forms:
        
        Terms 1-7: Strassen for A_left * B_topleft (2x2 * 2x2)
        Terms 8-14: Strassen for A_right * B_bottomleft (2x2 * 2x2)  
        Terms 15-21: Strassen for A_left * B_topright (2x2 * 2x2)
        Terms 22-27: Modified Strassen for A_right * B_bottomright (2x2 * 2x2)
                    with one term merged/adjusted
        
        The key is that the last Strassen block can share one term with
        the first block due to the specific structure.
        
        Let me implement this carefully.
        """
        rank = 27
        U = [[0] * rank for _ in range(n * m)]  # 8 x 27
        V = [[0] * rank for _ in range(m * p)]  # 16 x 27
        W = [[0] * rank for _ in range(p * n)]  # 8 x 27
        
        # Strassen coefficients for 2x2 * 2x2
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y00-Y10)
        # M5 = (X00+X01)Y11
        # M6 = (X01-X10)(Y00+Y10)
        # M7 = (X10-X11)(Y11+Y01)
        
        # For each 2x2 block product, we need:
        # U side: coefficients for the 4 A entries
        # V side: coefficients for the 4 B entries
        # W side: coefficients for the 4 C entries
        
        # Block 1: A_left (rows 0,1; cols 0,1) * B_topleft (rows 0,1; cols 0,1)
        # A entries: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx4, A[1,1]=idx5
        # B entries: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx4, B[1,1]=idx5
        # C entries: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
        
        strassen_defs = [
            # (U_coeffs: {(row,col):coeff}, V_coeffs, W_coeffs)
            # M1 = (X00+X11)(Y00+Y11)
            (
                {(0,0):1, (1,1):1},  # X00 + X11
                {(0,0):1, (1,1):1},  # Y00 + Y11
                {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}  # C00+ C11 - C01 - C10
            ),
            # M2 = (X10+X11)Y00
            (
                {(1,0):1, (1,1):1},  # X10 + X11
                {(0,0):1},            # Y00
                {(1,0):1}             # C10
            ),
            # M3 = X00(Y10-Y11)
            (
                {(0,0):1},            # X00
                {(1,0):1, (1,1):-1},  # Y10 - Y11
                {(0,0):1, (0,1):-1}   # C00 - C01
            ),
            # M4 = X11(Y00-Y10)
            (
                {(1,1):1},            # X11
                {(0,0):1, (1,0):-1},  # Y00 - Y10
                {(1,1):1, (1,0):-1}   # C11 - C10
            ),
            # M5 = (X00+X01)Y11
            (
                {(0,0):1, (0,1):1},   # X00 + X01
                {(1,1):1},            # Y11
                {(0,1):1, (1,1):1}    # C01 + C11
            ),
            # M6 = (X01-X10)(Y00+Y10)
            (
                {(0,1):1, (1,0):-1},  # X01 - X10
                {(0,0):1, (1,0):1},   # Y00 + Y10
                {(0,1):1, (0,0):-1, (1,1):1, (1,0):-1}  # C01-C00+C11-C10
            ),
            # M7 = (X10-X11)(Y11+Y01)
            (
                {(1,0):1, (1,1):-1},  # X10 - X11
                {(1,1):1, (0,1):1},   # Y11 + Y01
                {(1,0):1, (1,1):-1, (0,0):1, (0,1):-1}  # C10-C11+C00-C01
            ),
        ]
        
        # Block 1: A_left * B_topleft, cols 0-6
        for s in range(7):
            c = s
            u_coeff, v_coeff, w_coeff = strassen_defs[s]
            for (r, cc), val in u_coeff.items():
                U[r*4 + cc][c] += val  # A row r, col cc
            for (r, cc), val in v_coeff.items():
                V[r*4 + cc][c] += val  # B row r, col cc
            for (r, cc), val in w_coeff.items():
                W[cc*2 + r][c] += val  # C col cc, row r
        
        # Block 2: A_right * B_bottomleft, cols 7-13
        # A entries: A[0,2]=idx2, A[0,3]=idx3, A[1,2]=idx6, A[1,3]=idx7
        # B entries: B[2,0]=idx8, B[2,1]=idx9, B[3,0]=idx12, B[3,1]=idx13
        # C entries: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
        for s in range(7):
            c = 7 + s
            u_coeff, v_coeff, w_coeff = strassen_defs[s]
            for (r, cc), val in u_coeff.items():
                U[r*4 + (cc+2)][c] += val  # A row r, col cc+2
            for (r, cc), val in v_coeff.items():
                V[(r+2)*4 + cc][c] += val  # B row r+2, col cc
            for (r, cc), val in w_coeff.items():
                W[cc*2 + r][c] += val  # C col cc, row r
        
        # Block 3: A_left * B_topright, cols 14-20
        # A entries: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx4, A[1,1]=idx5
        # B entries: B[0,2]=idx2, B[0,3]=idx3, B[1,2]=idx6, B[1,3]=idx7
        # C entries: C[0,2]=idx4, C[0,3]=idx5, C[1,2]=idx6, C[1,3]=idx7
        for s in range(7):
            c = 14 + s
            u_coeff, v_coeff, w_coeff = strassen_defs[s]
            for (r, cc), val in u_coeff.items():
                U[r*4 + cc][c] += val  # A row r, col cc
            for (r, cc), val in v_coeff.items():
                V[r*4 + (cc+2)][c] += val  # B row r, col cc+2
            for (r, cc), val in w_coeff.items():
                W[(cc+2)*2 + r][c] += val  # C col cc+2, row r
        
        # Block 4: A_right * B_bottomright, cols 21-26 (only 6 terms!)
        # A entries: A[0,2]=idx2, A[0,3]=idx3, A[1,2]=idx6, A[1,3]=idx7
        # B entries: B[2,2]=idx10, B[2,3]=idx11, B[3,2]=idx14, B[3,3]=idx15
        # C entries: C[0,2]=idx4, C[0,3]=idx5, C[1,2]=idx6, C[1,3]=idx7
        
        # For this block, we use only 6 of the 7 Strassen terms.
        # The key insight: one of the Strassen terms can be expressed
        # as a linear combination of the others plus terms from other blocks.
        
        # Use Strassen terms 0,1,2,3,4,5 (skip term 6)
        for s in range(6):
            c = 21 + s
            u_coeff, v_coeff, w_coeff = strassen_defs[s]
            for (r, cc), val in u_coeff.items():
                U[r*4 + (cc+2)][c] += val  # A row r, col cc+2
            for (r, cc), val in v_coeff.items():
                V[(r+2)*4 + (cc+2)][c] += val  # B row r+2, col cc+2
            for (r, cc), val in w_coeff.items():
                W[(cc+2)*2 + r][c] += val  # C col cc+2, row r
        
        # Now we need to add the missing contribution from Strassen term 6
        # to the last 6 columns (21-26). We'll add it as a correction.
        # Strassen term 6: (X10-X11)(Y11+Y01)
        # For block 4: X = A_right, Y = B_bottomright
        # X10 = A[1,2]=idx6, X11 = A[1,3]=idx7
        # Y11 = B[3,3]=idx15, Y01 = B[2,3]=idx11
        # So: (A[1,2]-A[1,3])(B[3,3]+B[2,3])
        
        # This needs to contribute to C[1,2], C[1,3], C[0,2], C[0,3]
        # W coeffs: {(1,0):1, (1,1):-1, (0,0):1, (0,1):-1}
        # In C indexing: C[1,2]=idx6, C[1,3]=idx7, C[0,2]=idx4, C[0,3]=idx5
        # W[6] += 1, W[7] += -1, W[4] += 1, W[5] += -1
        
        # But we can't add a new column. Instead, we need to modify
        # the existing 6 columns to include this term.
        
        # Alternative: adjust the last column (col 26) to include
        # a combination that accounts for the missing term.
        
        # This approach is getting complex. Let me try a different strategy.
        
        return {"U": U, "V": V, "W": W}

    def build_rank_27_direct():
        """
        Direct construction of rank-27 decomposition.
        
        Use a systematic approach: start with 28-term Strassen and find
        one column that can be eliminated by adjusting others.
        """
        # First build the 28-term version
        result_28 = build_strassen_28()
        U28 = np.array(result_28["U"], dtype=float)
        V28 = np.array(result_28["V"], dtype=float)
        W28 = np.array(result_28["W"], dtype=float)
        
        # Try to find a column that is linearly dependent on others
        # in the tensor decomposition sense.
        
        # For each column c, check if removing it and adjusting others
        # can still give the correct tensor.
        
        # This is a linear algebra problem. For a fixed set of columns,
        # the tensor contribution is:
        # T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
        
        # We need T to equal the target tensor.
        
        # If we drop column c0, we need to find corrections delta_U, delta_V, delta_W
        # for the remaining columns such that the tensor is still correct.
        
        # This is nonlinear in general, but we can try a simpler approach:
        # check if any two columns can be merged.
        
        # For now, return the 28-term result as fallback
        return result_28

    def validate_fast(U, V, W, rank):
        """Faster validation using matrix operations."""
        U_arr = np.array(U, dtype=float)  # (nm, rank)
        V_arr = np.array(V, dtype=float)  # (mp, rank)
        W_arr = np.array(W, dtype=float)  # (pn, rank)
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposition tensor
        UV = U_arr[:, np.newaxis, :] * V_arr[np.newaxis, :, :]  # (nm, mp, rank)
        T = np.einsum('abr,cr->abc', UV, W_arr)
        
        return np.allclose(T, target, atol=1e-9)

    # Build and validate the rank-28 Strassen decomposition
    result_28 = build_strassen_28()
    valid_28 = validate_fast(result_28["U"], result_28["V"], result_28["W"], 28)
    
    if valid_28:
        best_result = result_28
        best_rank = 28
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try rank 27 with systematic search
    if time.time() < deadline - 10:
        U28 = np.array(best_result["U"], dtype=float)
        V28 = np.array(best_result["V"], dtype=float)
        W28 = np.array(best_result["W"], dtype=float)
        
        random.seed(42)
        
        # Strategy 1: Try dropping each column and see if we can fix it
        for drop_col in range(28):
            if time.time() > deadline - 5:
                break
            
            U27 = np.delete(U28, drop_col, axis=1)
            V27 = np.delete(V28, drop_col, axis=1)
            W27 = np.delete(W28, drop_col, axis=1)
            
            if validate_fast(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                best_rank = 27
                break
        
        # Strategy 2: Random search with structure
        if best_rank > 27:
            for attempt in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Start from Strassen-28, modify one column
                U27 = U28[:, :27].copy()
                V27 = V28[:, :27].copy()
                W27 = W28[:, :27].copy()
                
                # Add random small perturbations
                for _ in range(3):
                    ri = np.random.randint(0, n*m)
                    ci = np.random.randint(0, 27)
                    U27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                    ri = np.random.randint(0, m*p)
                    ci = np.random.randint(0, 27)
                    V27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                    ri = np.random.randint(0, p*n)
                    ci = np.random.randint(0, 27)
                    W27[ri, ci] += 0.5 * np.random.choice([-1, 1])
                
                if validate_fast(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                    best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    best_rank = 27
                    break
    
    return best_result

# EVOLVE-BLOCK-END
