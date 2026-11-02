import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W decompose the matmul tensor exactly (and are half-integers)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # half-integer check
    for M in (U, V, W):
        if np.max(np.abs(M * 2 - np.round(M * 2))) > tol:
            return False
    # tensor check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U[i * m + j], V[j * p + k], W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Rank n*m*p decomposition (always valid)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_int_lists(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    def conv(M):
        M = np.round(M * 2) / 2.0
        out = []
        for row in M:
            out.append([float(x) for x in row])
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _strassen_2x2_terms():
    """
    Returns the 7 terms for 2x2x2 Strassen algorithm.
    Each term is a tuple (u_vec, v_vec, w_vec) of length 4, 4, 4 respectively.
    u_vec corresponds to A (2x2 flattened), v_vec to B (2x2 flattened), w_vec to C (2x2 flattened).
    """
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # M2 = (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    
    # M3 = a00(b01-b11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0]
    
    # M4 = a11(b10-b00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [0, 0, 0, 1]
    
    # M5 = (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 1, 0, 0]
    
    # M6 = (a10-a00)(b00+b01)
    u6 = [0, 0, 1, -1]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 1, 0]
    
    # M7 = (a01-a11)(b10+b11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 0, 0, 1]
    
    return [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]

def _build_strassen_4x4x4():
    """
    Constructs the rank-49 decomposition for 4x4x4 matrix multiplication.
    Returns U, V, W arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    terms_2x2 = _strassen_2x2_terms()
    
    # Top-level Strassen terms for 4x4 blocks (2x2 blocks of 2x2 matrices)
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2 matrices
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product Xk * Yk = Zk
    # We need to decompose Xk*Yk=Zk using 2x2 Strassen.
    # Xk is a linear combination of A blocks.
    # Yk is a linear combination of B blocks.
    # Zk contributes to C blocks.
    
    # Let's define the top-level structure:
    # For each top-level term k (0..6):
    #   Xk = sum_{a,b} xa_k[a,b] * A_ab
    #   Yk = sum_{a,b} ya_k[a,b] * B_ab
    #   Zk = sum_{a,b} za_k[a,b] * C_ab (with signs)
    
    # Actually, it's easier to think of the 49 scalar multiplications directly.
    # Let's iterate through the 7 top-level terms and the 7 bottom-level terms.
    
    # Top-level definitions (coefficients for A blocks and B blocks)
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # Top-level term k defines Xk and Yk.
    # Xk is a 2x2 matrix formed from A blocks.
    # Yk is a 2x2 matrix formed from B blocks.
    
    top_level = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # X = [[A00, 0], [0, A11]] + ... ? No, Strassen uses sums.
        # Let's use the standard 2x2 Strassen formulas applied to the 4 2x2 blocks.
        # M1 = (A00+A11)(B00+B11)
        #   X1 = A00+A11, Y1 = B00+B11
        #   Z1 = C00+C11 (partially)
        # Actually, let's just map the 2x2 Strassen terms directly to the 4x4 blocks.
        
        # The 7 terms of the top-level Strassen are:
        # 1. (A00+A11) * (B00+B11)
        # 2. (A10+A11) * B00
        # 3. A00 * (B01-B11)
        # 4. A11 * (B10-B00)
        # 5. (A00+A01) * B11
        # 6. (A10-A00) * (B00+B01)
        # 7. (A01-A11) * (B10+B11)
        
        # Each of these is a product of two 2x2 matrices.
        # Let P = X * Y.
        # We decompose P = X*Y using the 2x2 Strassen algorithm, which gives 7 scalar multiplications.
        # So for each of the 7 top-level terms, we have 7 scalar multiplications.
        # Total 49.
        
        # We need to express X and Y in terms of the scalar entries of A and B.
        # And P in terms of the scalar entries of C.
    ]
    
    # Let's define the mapping for each top-level term.
    # Term 1: X = A00+A11, Y = B00+B11
    # X is 2x2. Its entries are sums of entries of A00 and A11.
    # A00 is rows 0-1, cols 0-1. A11 is rows 2-3, cols 2-3.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # We will build U, V, W by iterating.
    
    # List of top-level terms: (X_block_coeffs, Y_block_coeffs, Z_block_coeffs)
    # X_block_coeffs: dict mapping (br, bc) to coefficient. br, bc in {0,1}
    # Y_block_coeffs: dict mapping (br, bc) to coefficient.
    # Z_block_coeffs: dict mapping (br, bc) to coefficient.
    
    top_terms = [
        # 1
        ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1} ),
        # 2
        ( {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1} ),
        # 3
        ( {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1} ),
        # 4
        ( {(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1} ),
        # 5
        ( {(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1} ),
        # 6
        ( {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1} ),
        # 7
        ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1} ),
    ]
    
    idx = 0
    for x_coeff, y_coeff, z_coeff in top_terms:
        # For this top-level term, we have X = sum_x, Y = sum_y, Z = sum_z
        # We need to compute X*Y = Z using 2x2 Strassen.
        # The 2x2 Strassen algorithm computes 7 products.
        # Let's generate the 7 sub-terms for this specific X and Y.
        
        # The 2x2 Strassen terms are defined on the entries of X and Y.
        # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]].
        # The 7 products are:
        # P1 = (x00+x11)(y00+y11)
        # P2 = (x10+x11)y00
        # P3 = x00(y01-y11)
        # P4 = x11(y10-y00)
        # P5 = (x00+x01)y11
        # P6 = (x10-x00)(y00+y01)
        # P7 = (x01-x11)(y10+y11)
        
        # And the results Z are:
        # Z00 = P1 - P4 + P5 - P7
        # Z01 = P3 + P7
        # Z10 = P2 + P6
        # Z11 = P1 - P3 + P4 + P6
        
        # We need to express each Pk as a linear combination of A entries and B entries.
        # Pk = (linear combo of X entries) * (linear combo of Y entries)
        # X entries are linear combos of A entries.
        # Y entries are linear combos of B entries.
        
        # So Pk = (sum_{a,b} xa_k[a,b] A_ab) * (sum_{a,b} ya_k[a,b] B_ab)
        # This is a single scalar multiplication term in the tensor decomposition.
        # The term is: U_term * V_term * W_term
        # Where U_term is the coefficient of A in the first factor,
        # V_term is the coefficient of B in the second factor,
        # W_term is the coefficient of C in the result.
        
        # Wait, the tensor decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
        # Here, for a specific scalar multiplication Pk, we have:
        # Pk = (sum_{i,j} u_{ij} A_{ij}) * (sum_{j,k} v_{jk} B_{jk})
        # And Pk contributes to Z_{kl} with some weight w_{kl}.
        # So the term in the sum is:
        # (sum u_{ij} A_{ij}) * (sum v_{jk} B_{jk}) * (sum w_{kl} C_{kl})? No.
        
        # The tensor is T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}.
        # For a specific c, U_{ij,c} is the coefficient of A_{ij} in the first factor.
        # V_{jk,c} is the coefficient of B_{jk} in the second factor.
        # W_{ki,c} is the coefficient of C_{ki} in the result.
        
        # So for each of the 49 terms, we need to fill U, V, W.
        
        # Let's define the 7 sub-terms for the current top-level term.
        # We need the coefficients of X and Y entries in each sub-term.
        
        sub_terms = [
            # (x_coeffs, y_coeffs, z_coeffs)
            # x_coeffs: dict mapping (r,c) to coeff for X entry
            # y_coeffs: dict mapping (r,c) to coeff for Y entry
            # z_coeffs: dict mapping (r,c) to coeff for Z entry
            ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1, (0,1):-1, (1,0):1} ), # P1
            ( {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1} ), # P2
            ( {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1} ), # P3
            ( {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,1):1} ), # P4
            ( {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):-1} ), # P5
            ( {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1} ), # P6
            ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):-1, (1,1):-1} ), # P7
        ]
        
        for x_sub, y_sub, z_sub in sub_terms:
            # For this sub-term c:
            # Factor 1: sum_{r,c} x_sub[r,c] * X[r,c]
            # Factor 2: sum_{r,c} y_sub[r,c] * Y[r,c]
            # Result: sum_{r,c} z_sub[r,c] * Z[r,c]
            
            # X[r,c] is a linear combination of A entries.
            # X is formed from A blocks.
            # X[r,c] = sum_{br,bc} x_coeff[br,bc] * A_block[br,bc][r,c]
            
            # A_block[br,bc] has entries A[2*br+r, 2*bc+c].
            
            # So Factor 1 = sum_{r,c} x_sub[r,c] * sum_{br,bc} x_coeff[br,bc] * A[2*br+r, 2*bc+c]
            # Factor 1 = sum_{i,j} (sum_{r,c,br,bc} x_sub[r,c] x_coeff[br,bc] delta_{i,2*br+r} delta_{j,2*bc+c}) A[i,j]
            
            # Similarly for Factor 2 with B.
            # And Result with C.
            
            # We can compute the coefficients for U, V, W directly.
            
            u_row = np.zeros(16)
            v_row = np.zeros(16)
            w_row = np.zeros(16)
            
            for r in range(2):
                for c in range(2):
                    x_val = x_sub.get((r,c), 0)
                    y_val = y_sub.get((r,c), 0)
                    z_val = z_sub.get((r,c), 0)
                    
                    if x_val == 0 and y_val == 0 and z_val == 0:
                        continue
                    
                    # X[r,c] contributes to Factor 1 with weight x_val
                    for br in range(2):
                        for bc in range(2):
                            xc = x_coeff.get((br,bc), 0)
                            if xc != 0:
                                i = 2*br + r
                                j = 2*bc + c
                                u_row[i*4 + j] += x_val * xc
                    
                    # Y[r,c] contributes to Factor 2 with weight y_val
                    for br in range(2):
                        for bc in range(2):
                            yc = y_coeff.get((br,bc), 0)
                            if yc != 0:
                                j = 2*br + r
                                k = 2*bc + c
                                v_row[j*4 + k] += y_val * yc
                                
                    # Z[r,c] contributes to Result with weight z_val
                    for br in range(2):
                        for bc in range(2):
                            zc = z_coeff.get((br,bc), 0)
                            if zc != 0:
                                k = 2*br + r
                                i = 2*bc + c
                                w_row[k*4 + i] += z_val * zc
            
            U[:, idx] = u_row
            V[:, idx] = v_row
            W[:, idx] = w_row
            idx += 1
            
    return U, V, W

def _build_strassen_4x4x5():
    """
    Constructs the rank-65 decomposition for 4x4x5 matrix multiplication.
    Rank 49 for 4x4x4 part + Rank 16 for 4x4x1 part.
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 4x4x4 (k=0..3)
    U44, V44, W44 = _build_strassen_4x4x4()
    U[:, :49] = U44
    V[:16, :49] = V44
    W[:16, :49] = W44
    
    # Part 2: 4x4x1 (k=4)
    # Schoolbook: C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    # 16 terms.
    for i in range(4):
        for j in range(4):
            c = 49 + i*4 + j
            U[i*4 + j, c] = 1.0
            V[j*5 + 4, c] = 1.0
            W[4*4 + i, c] = 1.0
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    # Fallback: schoolbook (rank 80)
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen 4x4x5 (rank 65)
    if time.time() < deadline - 5:
        try:
            Uf, Vf, Wf = _build_strassen_4x4x5()
            if _verify(Uf, Vf, Wf, n, m, p):
                r = Uf.shape[1]
                if r < best_rank:
                    best = _to_int_lists(Uf, Vf, Wf)
                    best_rank = r
        except Exception as e:
            pass
            
    return best

# EVOLVE-BLOCK-END
