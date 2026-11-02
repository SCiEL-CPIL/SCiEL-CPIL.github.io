"""
Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication.
Strategy:
1. Try to find a rank-16 decomposition (7 for 2x2x2 Strassen + 9 for 2x2x3) using a global
   optimization approach on the 2x2x3 sub-problem.
2. If rank-16 fails, fall back to rank-18 (7 for 2x2x2 + 7 for 2x2x2 + 4 for 2x2x1) which
   is guaranteed to work and fast to construct.
"""
import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
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

def _verify_np(U, V, W, T, n=2, m=2, p=5, atol=1e-8):
    """Vectorized verification using numpy."""
    R = U.shape[1]
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=atol)

def _rational_approx(x, max_denom=100):
    """Approximate x as a fraction with small denominator using continued fractions."""
    if abs(x) < 1e-12:
        return 0.0
    # Try simple denominators
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 100]:
        n_val = round(x * d)
        if abs(x - n_val / d) < 1e-10:
            return n_val / d
    # Continued fraction approach
    try:
        frac = Fraction(x).limit_denominator(max_denom)
        if abs(x - float(frac)) < 1e-10:
            return float(frac)
    except:
        pass
    return x

def _snap_to_half_int(x, tol=1e-6):
    """Snap x to nearest half-integer if within tolerance."""
    h = round(x * 2) / 2
    if abs(x - h) < tol:
        return h
    return x

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-8):
        return None
    
    # Try to rationalize each entry individually with half-integer snapping
    U_s = np.array([[ _snap_to_half_int(_rational_approx(x)) for x in row] for row in U])
    V_s = np.array([[ _snap_to_half_int(_rational_approx(x)) for x in row] for row in V])
    W_s = np.array([[ _snap_to_half_int(_rational_approx(x)) for x in row] for row in W])
    
    result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
    
    if np.allclose(result_s, T, atol=1e-7):
        U_list = [[x if isinstance(x, int) else float(x) for x in row] for row in U_s.tolist()]
        V_list = [[x if isinstance(x, int) else float(x) for x in row] for row in V_s.tolist()]
        W_list = [[x if isinstance(x, int) else float(x) for x in row] for row in W_s.tolist()]
        
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                        mat[i][j] = int(round(mat[i][j]))
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try with small denominator grid
    for denom in [2, 4, 8, 16, 32]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
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

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
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
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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

def _search_rank16(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-16 decomposition.
    Uses Strassen for first 2 cols (rank 7) and random/structured for the rest (rank 9).
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 16
    R_strassen = 7
    
    # Get Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(30):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize first 7 columns with Strassen for cols 0,1
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Initialize remaining 9 columns randomly
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        # Use ALS refinement
        prev_err = np.inf
        for it in range(300):
            if time.time() - start_time > timeout - 5:
                break
            # Solve for U
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            
            # Solve for V
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            
            # Solve for W
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            
            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
            err = np.sqrt(err / (n*m * m*p * p*n))
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank15(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-15 decomposition.
    Uses Strassen for first 2 cols (rank 7) and random/structured for the rest (rank 8).
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 15
    R_strassen = 7
    
    # Get Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(40):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize first 7 columns with Strassen for cols 0,1
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Initialize remaining 8 columns randomly
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        # Use ALS refinement
        prev_err = np.inf
        for it in range(300):
            if time.time() - start_time > timeout - 5:
                break
            # Solve for U
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            
            # Solve for V
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            
            # Solve for W
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            
            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
            err = np.sqrt(err / (n*m * m*p * p*n))
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-17 decomposition.
    Uses Strassen for first 2 cols (rank 7) and random/structured for the rest (rank 10).
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 17
    R_strassen = 7
    
    # Get Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize first 7 columns with Strassen for cols 0,1
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Initialize remaining 10 columns randomly
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        # Use ALS refinement
        prev_err = np.inf
        for it in range(300):
            if time.time() - start_time > timeout - 5:
                break
            # Solve for U
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            
            # Solve for V
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            
            # Solve for W
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            
            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
            err = np.sqrt(err / (n*m * m*p * p*n))
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # First, get a valid baseline (rank 18) quickly
    baseline = _construct_rank18(n, m, p)
    
    # Try rank 15 search first (best case)
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 20
        result = _search_rank15(n, m, p, timeout=min(30, remaining))
        if result is not None:
            return result
    
    # Try rank 16 search
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 15
        result = _search_rank16(n, m, p, timeout=min(40, remaining))
        if result is not None:
            return result
    
    # Try rank 17 search
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        result = _search_rank17(n, m, p, timeout=min(30, remaining))
        if result is not None:
            return result
    
    # Return baseline rank 18
    if baseline is not None:
        return baseline
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)
