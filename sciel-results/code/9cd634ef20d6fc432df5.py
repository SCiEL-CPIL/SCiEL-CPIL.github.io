"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_decomp(n, m, p):
    """
    Build a rank-44 decomposition for 2x3x8 matrix multiplication.
    Strategy:
    1. Split A (2x3) into A1 (2x2) and A2 (2x1).
    2. Split B (3x8) into B1 (2x8) and B2 (1x8).
    3. C = A1*B1 + A2*B2.
    4. A2*B2 is rank 16 (each of 8 cols requires 2 mults).
    5. A1*B1 is 2x2 * 2x8. Split into 4 blocks of 2x2x2.
       Each 2x2x2 block uses Strassen (rank 7). Total 4*7=28.
    Total rank = 16 + 28 = 44.
    """
    n, m, p = 2, 3, 8
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    # A2: A[0,2] (idx 2), A[1,2] (idx 5)
    # B2: B[2,k] (idx 16+k)
    # C: C[i,k] (idx k*2+i)
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    # A1: A[0,0](0), A[0,1](1), A[1,0](3), A[1,1](4)
    # B1: B[0,k](k), B[1,k](8+k)
    # Split B1 into 4 blocks of 2 columns.
    for g in range(4):
        # Block g handles columns 2g, 2g+1
        b11 = 0 + 2 * g      # B[0, 2g]
        b12 = 0 + 2 * g + 1  # B[0, 2g+1]
        b21 = 8 + 2 * g      # B[1, 2g]
        b22 = 8 + 2 * g + 1  # B[1, 2g+1]
        
        a11 = 0  # A[0,0]
        a12 = 1  # A[0,1]
        a21 = 3  # A[1,0]
        a22 = 4  # A[1,1]
        
        # Strassen products for 2x2 * 2x2
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[(2 * g) * 2 + 0][col] = 1      # C[0, 2g]
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[(2 * g) * 2 + 1][col] = 1  # C[1, 2g]
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[(2 * g + 1) * 2 + 0][col] = 1   # C[0, 2g+1]
        W[(2 * g + 1) * 2 + 1][col] = -1  # C[1, 2g+1]
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        W[(2 * g) * 2 + 0][col] = -1  # C[0, 2g]
        W[(2 * g) * 2 + 1][col] = 1   # C[1, 2g]
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 0][col] = 1  # C[0, 2g+1]
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[(2 * g) * 2 + 0][col] = 1  # C[0, 2g]
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err

def _to_rational(U, V, W, tol=1e-6):
    """Convert float entries to exact rationals if they're close to simple fractions."""
    from fractions import Fraction
    def conv(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                # Try to find a simple fraction
                f = Fraction(v).limit_denominator(100)
                if abs(v - float(f)) < tol:
                    new_row.append(float(f))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    return conv(U), conv(V), conv(W)

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=100):
    """Try to reduce rank by solving a least-squares problem."""
    import numpy as np
    import time
    start = time.time()
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    if cur_rank <= target_rank:
        return None
    
    # We want to find U' (nm x target_rank), V' (mp x target_rank), W' (pn x target_rank)
    # such that the tensor is preserved.
    # Strategy: fix W and V, solve for U, then fix U and W, solve for V, etc.
    # Or: use the fact that we can try to merge columns.
    
    # Try: keep first target_rank columns and try to express the rest as linear combinations
    # that can be absorbed.
    
    # Alternative: use alternating least squares
    # Initialize with first target_rank columns
    U_init = U[:, :target_rank]
    V_init = V[:, :target_rank]
    W_init = W[:, :target_rank]
    
    # Build the tensor constraint matrix
    # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is nonlinear. Let's use a different approach.
    # Fix two of the three matrices and solve for the third.
    
    # If we fix V and W, then for each (i,j,k):
    # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # This is linear in U.
    
    # Let's define for each r: c_r[i,j,k] = V[j*p+k, r] * W[k*n+i, r]
    # Then we need: sum_r U[i*m+j, r] * c_r[i,j,k] = 1
    
    # This is a linear system in U.
    
    # Let's set up the system.
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    # Number of equations: nm * mp * pn / (some factor) - actually nm*mp*pn is too many
    # Each equation is for a specific (i,j,k)
    n_eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                n_eq += 1
    
    # Build A_matrix (n_eq x (nm * r)) and b (n_eq)
    # Variable: U_flat = [U[0,0], U[0,1], ..., U[0,r-1], U[1,0], ..., U[nm-1,r-1]]
    # For equation (i,j,k): sum_{r_idx} U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] = 1
    
    A_rows = []
    b_vec = []
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = np.zeros(nm * r)
                for r_idx in range(r):
                    coeff = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if coeff != 0:
                        row[(i*m+j) * r + r_idx] = coeff
                A_rows.append(row)
                b_vec.append(1.0)
                eq_idx += 1
    
    A_mat = np.array(A_rows)
    b_mat = np.array(b_vec)
    
    # Solve least squares
    try:
        U_flat, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_mat, rcond=None)
        U_sol = U_flat.reshape(nm, r)
    except:
        return None
    
    # Check error
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_sol[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                err = max(err, abs(val - 1.0))
    
    if err < 1e-8:
        # Now try to reduce V
        # Fix U_sol and W_init, solve for V
        A_rows2 = []
        b_vec2 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = np.zeros(mp * r)
                    for r_idx in range(r):
                        coeff = U_sol[i*m+j, r_idx] * W_init[k*n+i, r_idx]
                        if coeff != 0:
                            row[(j*p+k) * r + r_idx] = coeff
                    A_rows2.append(row)
                    b_vec2.append(1.0)
        
        A_mat2 = np.array(A_rows2)
        b_mat2 = np.array(b_vec2)
        
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A_mat2, b_mat2, rcond=None)
            V_sol = V_flat.reshape(mp, r)
        except:
            return None
        
        # Check
        err2 = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_sol[i*m+j] * V_sol[j*p+k] * W_init[k*n+i])
                    err2 = max(err2, abs(val - 1.0))
        
        if err2 < 1e-8:
            # Now try to reduce W
            A_rows3 = []
            b_vec3 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = np.zeros(pn * r)
                        for r_idx in range(r):
                            coeff = U_sol[i*m+j, r_idx] * V_sol[j*p+k, r_idx]
                            if coeff != 0:
                                row[(k*n+i) * r + r_idx] = coeff
                        A_rows3.append(row)
                        b_vec3.append(1.0)
            
            A_mat3 = np.array(A_rows3)
            b_mat3 = np.array(b_vec3)
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat3, b_mat3, rcond=None)
                W_sol = W_flat.reshape(pn, r)
            except:
                return None
            
            # Final check
            err3 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_sol[i*m+j] * V_sol[j*p+k] * W_sol[k*n+i])
                        err3 = max(err3, abs(val - 1.0))
            
            if err3 < 1e-8:
                return [U_sol.tolist(), V_sol.tolist(), W_sol.tolist()]
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use the rank-44 decomposition
    U, V, W = _build_decomp(n, m, p)
    
    # Verify
    err = _verify(U, V, W, n, m, p)
    if err < 1e-10:
        # Try to reduce rank
        for target in range(43, 30, -1):
            if time.time() - start_time > 120:
                break
            result = _try_reduce_rank(U, V, W, n, m, p, target)
            if result is not None:
                U_red, V_red, W_red = result
                # Convert to list of lists with integer/half-integer if possible
                return {"U": U_red, "V": V_red, "W": W_red}
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
