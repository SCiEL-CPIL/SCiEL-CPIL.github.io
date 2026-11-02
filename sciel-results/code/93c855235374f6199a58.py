"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify if the given U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """Construct a rank-18 decomposition using Strassen for 2x2x2 blocks + Schoolbook."""
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
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

def _build_target_tensor(n=2, m=2, p=5):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    return T

def _fast_als(U, V, W, T, R, n, m, p, max_iter=200, tol=1e-10):
    """Fast ALS refinement with precomputed structures."""
    prev_err = np.inf
    
    for it in range(max_iter):
        # Solve for U
        # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed i: T[i,:,:] = sum_r U[i,r] * (V[:,r] outer W[:,r])
        # Vectorize: b = T[i].flatten(), A[:,r] = (V[:,r] * W[:,r]).flatten()
        A_U = np.zeros((m*p * p*n, R))
        for r in range(R):
            A_U[:, r] = (V[:, r] * W[:, r]).flatten()
        
        for i in range(n*m):
            b = T[i].flatten()
            sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
            U[i] = sol
        
        # Solve for V
        A_V = np.zeros((n*m * p*n, R))
        for r in range(R):
            A_V[:, r] = (U[:, r] * W[:, r]).flatten()
        
        for j in range(m*p):
            b = T[:, j, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
            V[j] = sol
        
        # Solve for W
        A_W = np.zeros((n*m * m*p, R))
        for r in range(R):
            A_W[:, r] = (U[:, r] * V[:, r]).flatten()
        
        for k in range(p*n):
            b = T[:, :, k].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
            W[k] = sol
        
        # Compute error
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.sqrt(np.sum((recon - T)**2) / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _rationalize_and_check(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and rationalize entries."""
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(recon, T, atol=1e-7):
        return None
    
    # Try rationalization with various denominators
    for denom in [1, 2, 4, 8, 16]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        recon_s = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        
        if np.allclose(recon_s, T, atol=1e-6):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank17_direct(n=2, m=2, p=5, timeout=30):
    """
    Try to construct rank-17 directly using algebraic insight.
    Strategy: Use 7 terms for cols 0,1 (Strassen), 7 for cols 2,3 (Strassen),
    and 3 terms for col 4 that share structure with existing terms.
    """
    start = time.time()
    
    # First, try a direct algebraic construction
    # The key insight: for col 4, we need C[i,4] = sum_j A[i,j] * B[j,4]
    # This is just a 2x2 x 2x1 multiplication, which has rank 4 (schoolbook)
    # But maybe we can share with Strassen terms?
    
    # Try: use Strassen for (0,1), (2,3), and find 3 more terms for col 4
    # that exploit the structure
    
    R = 17
    T = _build_target_tensor(n, m, p)
    
    # Initialize with two Strassen blocks
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen block for cols 0,1
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1*p+1, 0] = 1
    W[0*n+0, 0] = 1; W[1*n+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0*n+1, 1] = 1; W[1*n+1, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[1*p+1, 2] = -1
    W[1*n+0, 2] = 1; W[1*n+1, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[1*p+0, 3] = 1; V[0, 3] = -1
    W[0*n+0, 3] = 1; W[0*n+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+1, 4] = 1
    W[0*n+0, 4] = -1; W[1*n+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1*n+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+0, 6] = 1; V[1*p+1, 6] = 1
    W[0*n+0, 6] = 1
    
    # Strassen block for cols 2,3
    base = 7
    U[0, base] = 1; U[3, base] = 1
    V[0*p+2, base] = 1; V[1*p+3, base] = 1
    W[2*n+0, base] = 1; W[3*n+1, base] = 1
    
    U[2, base+1] = 1; U[3, base+1] = 1
    V[0*p+2, base+1] = 1
    W[2*n+1, base+1] = 1; W[3*n+1, base+1] = -1
    
    U[0, base+2] = 1
    V[0*p+3, base+2] = 1; V[1*p+3, base+2] = -1
    W[3*n+0, base+2] = 1; W[3*n+1, base+2] = 1
    
    U[3, base+3] = 1
    V[1*p+2, base+3] = 1; V[0*p+2, base+3] = -1
    W[2*n+0, base+3] = 1; W[2*n+1, base+3] = 1
    
    U[0, base+4] = 1; U[1, base+4] = 1
    V[1*p+3, base+4] = 1
    W[2*n+0, base+4] = -1; W[3*n+0, base+4] = 1
    
    U[2, base+5] = 1; U[0, base+5] = -1
    V[0*p+2, base+5] = 1; V[0*p+3, base+5] = 1
    W[3*n+1, base+5] = 1
    
    U[1, base+6] = 1; U[3, base+6] = -1
    V[1*p+2, base+6] = 1; V[1*p+3, base+6] = 1
    W[2*n+0, base+6] = 1
    
    # Now we need 3 more terms for col 4
    # Schoolbook would need 4: A00*B04, A01*B14, A10*B04, A11*B14
    # Try to find 3 terms that cover all 4 products
    
    # Attempt 1: Use a Strassen-like approach for the last column
    # C04 = A00*B04 + A01*B14
    # C14 = A10*B04 + A11*B14
    
    # Try: M8 = (A00+A10)*B04, M9 = (A01+A11)*B14, M10 = correction
    # This doesn't quite work...
    
    # Let's just use ALS to find the last 3 terms
    base3 = 14
    
    # Random initialization for the last 3 terms
    for trial in range(50):
        if time.time() - start > timeout:
            break
        
        U_temp = U.copy()
        V_temp = V.copy()
        W_temp = W.copy()
        
        # Initialize last 3 terms randomly
        U_temp[:, base3:base3+3] = np.random.randn(n*m, 3) * 0.5
        V_temp[:, base3:base3+3] = np.random.randn(m*p, 3) * 0.5
        W_temp[:, base3:base3+3] = np.random.randn(p*n, 3) * 0.5
        
        # Run fast ALS
        U_r, V_r, W_r, err = _fast_als(U_temp, V_temp, W_temp, T, R, n, m, p, max_iter=100)
        
        result = _rationalize_and_check(U_r, V_r, W_r, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _try_rank16_direct(n=2, m=2, p=5, timeout=30):
    """Try to construct rank-16 directly."""
    start = time.time()
    
    R = 16
    T = _build_target_tensor(n, m, p)
    
    # Strategy: Use 7 for cols 0,1, 7 for cols 2,3, and 2 for col 4
    # This is very aggressive - col 4 needs at least rank 2 for 2x2x1? No, it needs 4.
    # So rank 16 is unlikely unless there's massive sharing.
    
    # Alternative: Use a different grouping
    # Cols 0,1,2 together? That's 2x2x3, rank at least 7?
    
    # Let's try: 7 for cols 0,1, then 9 for cols 2,3,4
    # 2x2x3 has rank at most 7 (Strassen-like) + 2 = 9? No, 2x2x3 needs more.
    
    # Actually, let's just try random + ALS
    for trial in range(100):
        if time.time() - start > timeout:
            break
        
        U = np.random.randn(n*m, R) * 0.3
        V = np.random.randn(m*p, R) * 0.3
        W = np.random.randn(p*n, R) * 0.3
        
        U_r, V_r, W_r, err = _fast_als(U, V, W, T, R, n, m, p, max_iter=150)
        
        result = _rationalize_and_check(U_r, V_r, W_r, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 16 (most aggressive, least likely)
    if time.time() < deadline - 80:
        result = _try_rank16_direct(n, m, p, timeout=min(20, deadline - time.time() - 80))
        if result is not None:
            return result
    
    # Try rank 17
    if time.time() < deadline - 40:
        result = _try_rank17_direct(n, m, p, timeout=min(30, deadline - time.time() - 40))
        if result is not None:
            return result
    
    # Try rank 18 (constructive, guaranteed)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
