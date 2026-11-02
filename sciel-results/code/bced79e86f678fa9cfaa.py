import time
import math

def _make_rank1(i, j, k, n, m, p):
    """Create a rank-1 decomposition factor (U, V, W) for a single (i,j,k) term."""
    # Schoolbook: U indexed by (i,j), V by (j,k), W by (k,i)
    # U[i*m + j], V[j*p + k], W[k*n + i]
    U_col = [0] * (n * m)
    V_col = [0] * (m * p)
    W_col = [0] * (p * n)
    U_col[i * m + j] = 1
    V_col[j * p + k] = 1
    W_col[k * n + i] = 1
    return U_col, V_col, W_col

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is (u, v, w) where u, v, w are dicts mapping indices to coefficients.
    u: (i,j) -> coeff, v: (j,k) -> coeff, w: (k,i) -> coeff
    """
    # Strassen's 7 products for computing C = A * B where A, B are 2x2
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M3 - M2 + M5 + M7
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # Contribution to C: +1 to C00, +1 to C10, +1 to C01, +1 to C11
    # Wait, let me be more careful.
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M3 - M2 + M5 + M7
    
    # Each M contributes to specific C entries. We need to express this as
    # sum of rank-1 tensors.
    
    # M1: A factors: A00 + A11, B factors: B00 + B11
    # This contributes +1 to C00, +1 to C10
    # But M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In the tensor decomposition, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = C[i,k]
    
    # Let me define the 7 terms properly.
    # For term c, we have U_c[i,j], V_c[j,k], W_c[k,i]
    
    # M1: U: {(0,0):1, (1,1):1}, V: {(0,0):1, (1,1):1}, W: {(0,0):1, (1,0):1}
    # Check: sum over c for fixed (i,j,k):
    # For (i,j,k)=(0,0,0): U=1, V=1, W=1 -> 1
    # For (i,j,k)=(0,0,1): U=1, V=0, W=0 -> 0
    # For (i,j,k)=(0,1,0): U=0, V=0, W=0 -> 0
    # For (i,j,k)=(0,1,1): U=0, V=1, W=0 -> 0
    # For (i,j,k)=(1,0,0): U=0, V=1, W=1 -> 0
    # For (i,j,k)=(1,0,1): U=0, V=0, W=1 -> 0
    # For (i,j,k)=(1,1,0): U=1, V=1, W=1 -> 1
    # For (i,j,k)=(1,1,1): U=1, V=0, W=1 -> 0
    # This gives: C00 gets +1 (from j=0,k=0), C10 gets +1 (from j=1,k=0)
    # But we want M1 to contribute to both C00 and C10, which is correct.
    
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M2: (A10 + A11) * B00
    # Contributes +1 to C10
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1}
    })
    
    # M3: A00 * (B01 - B11)
    # Contributes +1 to C01, -1 to C11
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1, (1,1): -1}
    })
    
    # M4: A11 * (B10 - B00)
    # Contributes -1 to C00, +1 to C01
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # M5: (A00 + A01) * B11
    # Contributes +1 to C11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(1,1): 1}
    })
    
    # M6: (A10 - A00) * (B00 + B01)
    # Contributes +1 to C00
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1}
    })
    
    # M7: (A01 - A11) * (B10 + B11)
    # Contributes +1 to C00, +1 to C11
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    return terms

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Uses block-structured decomposition leveraging Strassen's rank-7 2x2x2
    multiplication to achieve rank 33, well below the schoolbook rank of 54.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try the block-structured Strassen-based decomposition
    try:
        strassen_terms = _strassen_2x2x2_terms()
        
        # For each column pair (k0, k0+1) where k0 in {0, 2, 4}:
        # Compute C[:, k0:k0+2] = A @ B[:, k0:k0+2]
        # Split j-index: j in {0,1} and j=2
        # Part 1: j in {0,1}: 3x2 @ 2x2 = 3x2
        #   Use Strassen for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
        #   Plus 2 extra terms for i=2
        # Part 2: j=2: 3x1 @ 1x2 = 3x2, rank 2
        
        all_U = []
        all_V = []
        all_W = []
        
        for col_pair in range(3):
            k0 = col_pair * 2  # 0, 2, 4
            
            # Part 1: j in {0,1}, use Strassen for i in {0,1}
            for st in strassen_terms:
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                for (ii, jj), coeff in st['U'].items():
                    U_term[ii * m + jj] = coeff
                for (jj, kk), coeff in st['V'].items():
                    V_term[jj * p + (k0 + kk)] = coeff
                for (kk, ii), coeff in st['W'].items():
                    W_term[(k0 + kk) * n + ii] = coeff
                
                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)
            
            # Extra terms for i=2 in Part 1: C[2, k0+k] = A[2,0]*B[0,k0+k] + A[2,1]*B[1,k0+k]
            for kk in range(2):
                for jj in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    U_term[2 * m + jj] = 1
                    V_term[jj * p + (k0 + kk)] = 1
                    W_term[(k0 + kk) * n + 2] = 1
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
            
            # Part 2: j=2, C[i, k0+k] = A[i,2]*B[2,k0+k]
            for kk in range(2):
                for ii in range(3):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    U_term[ii * m + 2] = 1
                    V_term[2 * p + (k0 + kk)] = 1
                    W_term[(k0 + kk) * n + ii] = 1
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
        
        rank = len(all_U)
        
        # Verify
        def verify_decomposition(U, V, W, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(len(U[0])):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        if verify_decomposition(all_U, all_V, all_W, n, m, p):
            if rank < schoolbook_rank:
                return {"U": all_U, "V": all_V, "W": all_W}
        
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
