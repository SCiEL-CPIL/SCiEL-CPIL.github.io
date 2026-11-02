"""Matrix multiplication 2,2,11 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n=2, m=2, p=11):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_strict(U, V, W, n=2, m=2, p=11):
    """Verify with stricter tolerance using exact rational arithmetic."""
    from fractions import Fraction
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for c in range(rank):
                    val += Fraction(U[i*m+j][c]).limit_denominator(10**6) * \
                           Fraction(V[j*p+k][c]).limit_denominator(10**6) * \
                           Fraction(W[k*n+i][c]).limit_denominator(10**6)
                if val != Fraction(1, 1):
                    return False
    return True

def _construct_rank39():
    """
    Known rank-39 construction for (2,2,11) matrix multiplication.
    
    This uses the fact that 2x2x11 can be done with rank 39.
    The construction is based on grouping the 11 columns into blocks.
    """
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]  # 22 x 39
    
    # We use a systematic construction.
    # For 2x2x11, we can think of it as computing:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # ... etc for all 11 columns
    # 
    # A known approach: use the fact that we need 44 scalar products total,
    # but we can share intermediate products.
    #
    # Strategy: For each of the 11 columns of B, we need to compute
    # A*B_col. This is a 2x2 times 2x1 multiplication, which needs
    # 4 multiplications per column = 44 total.
    # 
    # We can save 5 by sharing. One known decomposition:
    # Use Strassen-like ideas on the 2x2 part per column, but that only
    # gives 3 per column = 33, which isn't enough for 11 columns sharing.
    #
    # Actually, rank 39 is achievable. Let me use a known construction.
    #
    # Construction: Split the 11 columns into groups.
    # For columns 0-9 (10 columns), use a pattern that shares some products.
    # For column 10, use a simpler pattern.
    
    # Alternative: Use the fact that M(2,2,11) <= 39.
    # One construction:
    # Think of it as: we need to compute C = A * B where A is 2x2, B is 2x11.
    # C is 2x11.
    # 
    # Standard: 4 * 11 = 44 multiplications.
    # 
    # Sharing strategy: For each pair of rows in A, we can compute products
    # that are shared across columns of B.
    
    # Let me use a direct known construction for rank 39.
    # Based on the observation that 2*2*11 = 44 and we need to save 5.
    
    # I'll use a construction based on bilinear maps.
    # For the 2x2x2 sub-problem (first 2 columns), use Strassen (rank 3 instead of 4).
    # That saves 1. Then for the remaining 9 columns, each needs 4, but we can
    # share some.
    
    # Actually, let me just use a verified rank-44 schoolbook and try to find
    # a better one through search. But first, let me hardcode a known rank-39.
    
    # Known: The tensor rank of matrix multiplication M(n,m,p) for (2,2,11)
    # is at most 39. Here's a construction:
    
    # We'll use 11 "column blocks" but with sharing.
    # 
    # Let's denote the entries:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, ..., b0,10],
    #      [b10, b11, ..., b1,10]]
    # 
    # C[i,j] = a_i0*b_0j + a_i1*b_1j for i=0,1 and j=0..10
    
    # For each j, we need:
    # C[0,j] = a00*b0j + a01*b1j
    # C[1,j] = a10*b0j + a11*b1j
    
    # That's 4 multiplications per j, 44 total.
    # 
    # Sharing: Note that a00*b0j and a00*b0k share a00. But in bilinear
    # decomposition, each "product" is a separate term unless we factor.
    
    # In tensor decomposition, each term is (u_i, v_i, w_i) where:
    # u is a linear form in A entries
    # v is a linear form in B entries  
    # w is a linear form in C entries
    # and the product u*v contributes to w's coefficient.
    
    # So we want: sum_c alpha_c(A) * beta_c(B) = C entries (linear combination)
    
    # For rank 39, we need 39 such terms.
    
    # Let me try a construction:
    # For j = 0..10, define:
    # Term 4j+1: alpha = a00, beta = b0j, contributes to C[0,j]
    # Term 4j+2: alpha = a01, beta = b1j, contributes to C[0,j]
    # Term 4j+3: alpha = a10, beta = b0j, contributes to C[1,j]
    # Term 4j+4: alpha = a11, beta = b1j, contributes to C[1,j]
    # That's 44 terms.
    
    # To save 5, we need to combine 5 pairs of terms into single terms.
    
    # One approach: For some columns, use Strassen-like combinations.
    # E.g., for column j, instead of 4 terms, use 3:
    # (a00+a10)(b0j+b1j) -> contributes to C[0,j] + C[1,j] + ...
    # This doesn't directly work because the C entries are different.
    
    # Let me try a different approach: use the known result that
    # M(2,2,11) <= 39 via a specific construction.
    
    # I'll construct it as follows:
    # Use 3 terms for the first column (Strassen-like) and 4 terms for the rest,
    # but with some sharing across columns.
    
    # Actually, 3 + 4*10 = 43, still not 39.
    # Need more sharing.
    
    # Let me use a block approach:
    # Group columns into pairs: (0,1), (2,3), (4,5), (6,7), (8,9), and column 10.
    # For each pair (2j, 2j+1), we compute A * B[:, 2j:2j+2] which is 2x2 times 2x2.
    # That's a 2x2x2 matrix multiplication, rank 7 (Strassen) instead of 8.
    # Save 1 per pair.
    # 5 pairs: save 5. 1 single column: 4 terms.
    # Total: 5*7 + 4 = 35 + 4 = 39. Yes!
    
    # Construction:
    # For each pair (j1, j2) = (0,1), (2,3), (4,5), (6,7), (8,9):
    # Use Strassen's 7 terms for the 2x2x2 multiplication of A and B[:, j1:j2+1].
    
    # For column 10: use 4 standard terms.
    
    # Strassen's 7 terms for 2x2x2:
    # Given X (2x2) and Y (2x2), compute Z = X*Y (2x2).
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y01+Y11)
    #
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M4
    # Z10 = M2 + M7
    # Z11 = M1 - M3 + M6 - M7
    
    # Here X = A (2x2), Y = B[:, j1:j2+1] (2x2).
    # Z = C[:, j1:j2+1] (2x2).
    
    # U entries (linear forms in A):
    # M1: a00 + a11
    # M2: a10 + a11
    # M3: a00
    # M4: a11
    # M5: a00 + a01
    # M6: a10 - a00
    # M7: a01 - a11
    
    # V entries (linear forms in B):
    # M1: b0,j1 + b1,j2  (Y00 + Y11)
    # M2: b0,j1          (Y00)
    # M3: b1,j1 - b1,j2  (Y10 - Y11)
    # M4: b0,j2 - b0,j1  (Y01 - Y00)
    # M5: b1,j2          (Y11)
    # M6: b0,j1 + b1,j1  (Y00 + Y10)
    # M7: b0,j2 + b1,j2  (Y01 + Y11)
    
    # W entries (linear forms in C):
    # M1: c0,j1 + c1,j2  (Z00 + Z11)
    # M2: c1,j1          (Z10)
    # M3: c0,j2          (Z01)
    # M4: c1,j2 - c1,j1  (Z11 - Z10) -- wait, let me recheck
    #
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M4
    # Z10 = M2 + M7
    # Z11 = M1 - M3 + M6 - M7
    #
    # So W for each M:
    # M1: Z00 + Z11 -> w has 1 at (j1, 0) and 1 at (j2, 1) in C indices
    #     C is 2x11, so C[i, j] with i=0,1 and j=0..10
    #     W index: k*n + i where k is the column of B (which becomes row in W?)
    #
    # Wait, the convention is W[k*n+i, c] where k is the column index of B/C.
    # C has shape (2, 11), so C[i, k] for i=0,1 and k=0..10.
    # W row index = k*n + i = 2*k + i.
    
    # Let me redo:
    # Z = C[:, j1:j2+1], so Z[0,0]=C[0,j1], Z[0,1]=C[0,j2], Z[1,0]=C[1,j1], Z[1,1]=C[1,j2]
    #
    # M1 contributes to Z00 and Z11: W[2*j1+0] += 1, W[2*j2+1] += 1
    # M2 contributes to Z10: W[2*j1+1] += 1
    # M3 contributes to Z01: W[2*j2+0] += 1
    # M4 contributes to -Z00 and Z01: W[2*j1+0] += -1, W[2*j2+0] += 1
    # M5 contributes to Z00: W[2*j1+0] += 1
    # M6 contributes to Z11: W[2*j2+1] += 1
    # M7 contributes to -Z00, Z10, -Z11: W[2*j1+0] += -1, W[2*j1+1] += 1, W[2*j2+1] += -1
    
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    col = 0
    for (j1, j2) in pairs:
        # M1
        U[0][col] = 1.0  # a00
        U[1][col] = 0.0  # a01
        U[2][col] = 0.0  # a10
        U[3][col] = 1.0  # a11
        # U is indexed by i*m+j: row 0 = a00, row 1 = a01, row 2 = a10, row 3 = a11
        # U[i*m+j] where i=0,1 and j=0,1:
        # U[0] = a00, U[1] = a01, U[2] = a10, U[3] = a11
        
        # Actually U has shape (n*m, rank) = (4, rank)
        # U[0] corresponds to (i=0, j=0) = a00
        # U[1] corresponds to (i=0, j=1) = a01
        # U[2] corresponds to (i=1, j=0) = a10
        # U[3] corresponds to (i=1, j=1) = a11
        
        U[0][col] = 1.0
        U[1][col] = 0.0
        U[2][col] = 0.0
        U[3][col] = 1.0
        
        # V has shape (m*p, rank) = (22, rank)
        # V[j*p+k] where j=0,1 (row of B) and k=0..10 (column of B)
        # V[0] = b00, V[1] = b01, ..., V[10] = b0,10
        # V[11] = b10, V[12] = b11, ..., V[21] = b1,10
        
        V[0][col] = 0.0
        for t in range(p):
            if t == j1:
                V[0 * p + t][col] = 1.0  # b0,j1
            elif t == j2:
                V[0 * p + t][col] = 0.0
            else:
                V[0 * p + t][col] = 0.0
        for t in range(p):
            if t == j1:
                V[1 * p + t][col] = 0.0
            elif t == j2:
                V[1 * p + t][col] = 1.0  # b1,j2
            else:
                V[1 * p + t][col] = 0.0
        
        # W has shape (p*n, rank) = (22, rank)
        # W[k*n+i] where k=0..10 (column) and i=0,1 (row)
        W[2*j1+0][col] = 1.0   # C[0, j1]
        W[2*j2+1][col] = 1.0   # C[1, j2]
        for t in range(p):
            for r in range(2):
                if not (t == j1 and r == 0) and not (t == j2 and r == 1):
                    W[2*t+r][col] = 0.0
        
        col += 1
        
        # M2
        U[0][col] = 0.0
        U[1][col] = 0.0
        U[2][col] = 1.0
        U[3][col] = 1.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[0 * p + j1][col] = 1.0  # b0,j1
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j1+1][col] = 1.0  # C[1, j1]
        
        col += 1
        
        # M3
        U[0][col] = 1.0
        U[1][col] = 0.0
        U[2][col] = 0.0
        U[3][col] = 0.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[1 * p + j1][col] = 1.0   # b1,j1
        V[1 * p + j2][col] = -1.0  # -b1,j2
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j2+0][col] = 1.0  # C[0, j2]
        
        col += 1
        
        # M4
        U[0][col] = 0.0
        U[1][col] = 0.0
        U[2][col] = 0.0
        U[3][col] = 1.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[0 * p + j2][col] = 1.0   # b0,j2
        V[0 * p + j1][col] = -1.0  # -b0,j1
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j1+0][col] = -1.0  # -C[0, j1]
        W[2*j2+0][col] = 1.0   # C[0, j2]
        
        col += 1
        
        # M5
        U[0][col] = 1.0
        U[1][col] = 1.0
        U[2][col] = 0.0
        U[3][col] = 0.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[1 * p + j2][col] = 1.0  # b1,j2
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j1+0][col] = 1.0  # C[0, j1]
        
        col += 1
        
        # M6
        U[0][col] = -1.0
        U[1][col] = 0.0
        U[2][col] = 1.0
        U[3][col] = 0.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[0 * p + j1][col] = 1.0  # b0,j1
        V[1 * p + j1][col] = 1.0  # b1,j1
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j2+1][col] = 1.0  # C[1, j2]
        
        col += 1
        
        # M7
        U[0][col] = 0.0
        U[1][col] = 1.0
        U[2][col] = 0.0
        U[3][col] = -1.0
        
        for t in range(p):
            V[0 * p + t][col] = 0.0
            V[1 * p + t][col] = 0.0
        V[0 * p + j2][col] = 1.0  # b0,j2
        V[1 * p + j2][col] = 1.0  # b1,j2
        
        for t in range(p):
            for r in range(2):
                W[2*t+r][col] = 0.0
        W[2*j1+0][col] = -1.0  # -C[0, j1]
        W[2*j1+1][col] = 1.0   # C[1, j1]
        W[2*j2+1][col] = -1.0  # -C[1, j2]
        
        col += 1
    
    # For column 10: 4 standard terms
    for jj in range(4):
        if jj == 0:
            # a00 * b0,10 -> C[0,10]
            U[0][col] = 1.0
            U[1][col] = 0.0
            U[2][col] = 0.0
            U[3][col] = 0.0
            for t in range(p):
                V[0*p+t][col] = 0.0
                V[1*p+t][col] = 0.0
            V[0*p+10][col] = 1.0
            for t in range(p):
                for r in range(2):
                    W[2*t+r][col] = 0.0
            W[2*10+0][col] = 1.0
        elif jj == 1:
            # a01 * b1,10 -> C[0,10]
            U[0][col] = 0.0
            U[1][col] = 1.0
            U[2][col] = 0.0
            U[3][col] = 0.0
            for t in range(p):
                V[0*p+t][col] = 0.0
                V[1*p+t][col] = 0.0
            V[1*p+10][col] = 1.0
            for t in range(p):
                for r in range(2):
                    W[2*t+r][col] = 0.0
            W[2*10+0][col] = 1.0
        elif jj == 2:
            # a10 * b0,10 -> C[1,10]
            U[0][col] = 0.0
            U[1][col] = 0.0
            U[2][col] = 1.0
            U[3][col] = 0.0
            for t in range(p):
                V[0*p+t][col] = 0.0
                V[1*p+t][col] = 0.0
            V[0*p+10][col] = 1.0
            for t in range(p):
                for r in range(2):
                    W[2*t+r][col] = 0.0
            W[2*10+1][col] = 1.0
        else:
            # a11 * b1,10 -> C[1,10]
            U[0][col] = 0.0
            U[1][col] = 0.0
            U[2][col] = 0.0
            U[3][col] = 1.0
            for t in range(p):
                V[0*p+t][col] = 0.0
                V[1*p+t][col] = 0.0
            V[1*p+10][col] = 1.0
            for t in range(p):
                for r in range(2):
                    W[2*t+r][col] = 0.0
            W[2*10+1][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}


def _try_search_rank(r_target, n=2, m=2, p=11, max_time=10):
    """Try to find a decomposition of rank r_target using numerical optimization."""
    import numpy as np
    from scipy.optimize import minimize
    
    start = time.time()
    deadline = start + max_time
    
    # Use a random initialization and try to minimize the error
    # This is a non-convex problem, so we try multiple restarts
    
    best_result = None
    best_err = float('inf')
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 1:
            break
        
        np.random.seed(42 + restart)
        
        U = np.random.randn(n*m, r_target) * 0.5
        V = np.random.randn(m*p, r_target) * 0.5
        W = np.random.randn(p*n, r_target) * 0.5
        
        def compute_error(params):
            U_flat = params[:n*m*r_target].reshape(n*m, r_target)
            V_flat = params[n*m*r_target:n*m*r_target + m*p*r_target].reshape(m*p, r_target)
            W_flat = params[n*m*r_target + m*p*r_target:].reshape(p*n, r_target)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r_target):
                            val += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        err += (val - 1.0)**2
            return err**0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def obj(x):
            return compute_error(x)
        
        try:
            res = minimize(obj, params, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
            if time.time() > deadline - 1:
                break
            err = res.fun
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    U_opt = res.x[:n*m*r_target].reshape(n*m, r_target)
                    V_opt = res.x[n*m*r_target:n*m*r_target + m*p*r_target].reshape(m*p, r_target)
                    W_opt = res.x[n*m*r_target + m*p*r_target:].reshape(p*n, r_target)
                    # Round to nearest 0.5
                    U_r = np.round(U_opt * 2) / 2
                    V_r = np.round(V_opt * 2) / 2
                    W_r = np.round(W_opt * 2) / 2
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    # Try integer rounding too
                    U_ri = np.round(U_opt).astype(int).astype(float)
                    V_ri = np.round(V_opt).astype(int).astype(float)
                    W_ri = np.round(W_opt).astype(int).astype(float)
                    if _verify(U_ri.tolist(), V_ri.tolist(), W_ri.tolist()):
                        return U_ri.tolist(), V_ri.tolist(), W_ri.tolist()
        except Exception:
            pass
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 by 2x11 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # First, try the known rank-39 construction
    result_39 = _construct_rank39()
    
    if _verify(result_39['U'], result_39['V'], result_39['W']):
        # Try to find a better rank
        best = result_39
        best_rank = 39
        
        start = time.time()
        deadline = start + 130  # Leave 20s margin
        
        # Try ranks from 38 down to 30
        for r in range(38, 29, -1):
            if time.time() > deadline - 15:
                break
            
            result = _try_search_rank(r, n, m, p, max_time=min(15, deadline - time.time()))
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = r
                    break
        
        return best
    
    # Fallback: schoolbook rank 44
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
    return {"U": U, "V": V, "W": W}
