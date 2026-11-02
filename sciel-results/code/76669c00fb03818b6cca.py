"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
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

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    return U, V, W

def _get_fourier_basis_U(n=2, m=2, p=5, R=16):
    """
    Build a structured U matrix using Fourier modes of Z_p combined with
    Strassen modes for the 2x2 A-block.
    
    U has shape (n*m, R) = (4, R).
    We use the cyclic structure in k to build U columns.
    
    Strategy: For each of the 4 A-entries (A00, A01, A10, A11),
    assign Fourier-like coefficients over the p dimension.
    """
    U = np.zeros((n*m, R))
    # A entries: A00=0, A01=1, A10=2, A11=3
    
    # Use a structured basis: for each rank r, U has a pattern.
    # We'll use the idea that U should be "diagonal" in some sense
    # to make the linear system for V,W well-conditioned.
    
    # Strategy: Use a Kronecker-like structure
    # For rank r, assign U[i*m+j, r] based on a structured pattern
    
    # Let's use: for r in 0..R-1, 
    #   U[0, r] = cos(2*pi*r/R) or similar
    # But we need real entries. Let's use a simpler structured basis.
    
    # Use the first 4 rows as a structured basis:
    # Row 0 (A00): [1, 1, 1, 1, ...] with variations
    # Row 1 (A01): [0, 1, 0, 1, ...]
    # Row 2 (A10): [1, 0, 1, 0, ...]
    # Row 3 (A11): [0, 0, 1, 1, ...]
    
    # Actually, let's use a more principled approach:
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed (j,k), the sum over i gives: sum_i U[i*m+j, r] * W[k*n+i, r] = 1/V[j*p+k, r]
    # This is a 2x2 system for each (j,k,r).
    
    # Let's use a basis that makes U have full column rank and good conditioning.
    # Use a Hadamard-like or Fourier-like basis.
    
    # For R columns, use the first R columns of a 4xR structured matrix.
    # Pattern: U[i, r] = omega^(i*r) where omega is a 4th root of unity... 
    # but we need real. Use cos/sin pairs.
    
    # Simpler: use a structured basis where each column r has a distinct pattern
    # across the 4 A-entries.
    
    # Column r: 
    #   A00 = 1
    #   A01 = (r % 2)  # 0 or 1
    #   A10 = ((r+1) % 2)  # 1 or 0
    #   A11 = (r // 2) % 2  # 0 or 1
    
    # This gives a 4xR matrix with rank min(4, R) and good structure.
    
    for r in range(R):
        U[0, r] = 1.0
        U[1, r] = float(r % 2)
        U[2, r] = float((r + 1) % 2)
        U[3, r] = float((r // 2) % 2)
    
    return U

def _solve_VW_from_U(U, T, n=2, m=2, p=5, R=None):
    """
    Given U, solve for V and W such that sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w].
    
    This is a bilinear system. We can solve it by:
    1. For each (u,v), we need: sum_r W[w,r] * (U[u,r]*V[v,r]) = T[u,v,w] for all w.
    2. This is a linear system in W if V is known, or in V if W is known.
    3. Use ALS: alternate between solving for V (given W) and W (given V).
    
    Actually, a better approach:
    - Fix U. The tensor equation is: T = sum_r u_r ⊗ v_r ⊗ w_r
    - For each pair (u,v): sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
    - This is: (U[:,r]*V[:,r]) contracted with W[w,r] = T[u,v,w]
    
    Let's use a different formulation:
    - Reshape: for each u, T[u] is a (m*p, p*n) matrix.
    - T[u] = sum_r U[u,r] * (V[:,r] ⊗ W[:,r])  [outer product]
    - So T[u] = sum_r U[u,r] * vec(V[:,r] * W[:,r]^T)  [vectorized outer product]
    
    This is a linear system in the entries of V and W given U.
    But it's bilinear (V*W), so not linear.
    
    Better: use the fact that for each r, we have a rank-1 matrix M_r = V[:,r] ⊗ W[:,r].
    Then T[u] = sum_r U[u,r] * M_r (flattened).
    
    We can solve this with ALS:
    - Initialize V, W randomly
    - Iterate:
      - For each v: V[v,:] = argmin sum_u sum_w (T[u,v,w] - sum_r U[u,r]*V[v,r]*W[w,r])^2
        This is linear in V[v,:] given W: 
        sum_r V[v,r] * (sum_w U[u,r]*W[w,r]*... ) 
        Actually: for fixed v, the equation is:
        sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,w
        This is a linear system in V[v,:] with (n*m)*(p*n) equations and R unknowns.
        
    Let me reformulate more carefully.
    
    For fixed v, define: for each (u,w), we need sum_r U[u,r]*W[w,r]*V[v,r] = T[u,v,w]
    Let A_{(u,w), r} = U[u,r]*W[w,r], b_{(u,w)} = T[u,v,w]
    Then A * V[v,:] = b. This is a linear system (n*m*p*n) x R.
    """
    if R is None:
        R = U.shape[1]
    
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    for iteration in range(200):
        # Solve for V
        for v in range(m*p):
            # Build linear system: for each (u,w), sum_r U[u,r]*W[w,r]*V[v,r] = T[u,v,w]
            A = np.zeros((n*m * p*n, R))
            b = np.zeros(n*m * p*n)
            idx = 0
            for u in range(n*m):
                for w in range(p*n):
                    for r in range(R):
                        A[idx, r] = U[u, r] * W[w, r]
                    b[idx] = T[u, v, w]
                    idx += 1
            V[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Solve for W
        for w in range(p*n):
            A = np.zeros((n*m * m*p, R))
            b = np.zeros(n*m * m*p)
            idx = 0
            for u in range(n*m):
                for v in range(m*p):
                    for r in range(R):
                        A[idx, r] = U[u, r] * V[v, r]
                    b[idx] = T[u, v, w]
                    idx += 1
            W[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return V, W

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-5):
        return None
    
    # Try to rationalize with various denominators
    # Expand denominators to find more precise solutions
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        # Check if the rounded solution is valid
        # We can do a quick check on a few entries or full check
        # Full check is safer
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            # Convert to int if close to integer
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

def _search_with_structured_U(n=2, m=2, p=5, R=16, timeout=120):
    """
    Search for rank-R decomposition using structured U basis.
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
    
    # Try different structured U bases
    for trial in range(20):
        if time.time() - start_time > timeout - 5:
            break
        
        # Build structured U
        if trial == 0:
            # Original structured basis
            U = np.zeros((n*m, R))
            for r in range(R):
                U[0, r] = 1.0
                U[1, r] = float(r % 2)
                U[2, r] = float((r + 1) % 2)
                U[3, r] = float((r // 2) % 2)
        elif trial == 1:
            # Fourier-like basis (real part)
            U = np.zeros((n*m, R))
            for r in range(R):
                for i in range(4):
                    U[i, r] = np.cos(2 * np.pi * i * r / R)
        elif trial == 2:
            # Mixed: first 4 are identity-like, rest random
            U = np.zeros((n*m, R))
            U[0, 0] = 1; U[0, 1] = 1
            U[1, 1] = 1; U[1, 2] = 1
            U[2, 2] = 1; U[2, 3] = 1
            U[3, 3] = 1; U[3, 4] = 1
            for r in range(4, R):
                U[:, r] = np.random.randn(n*m) * 0.5
        elif trial == 3:
            # Strassen-like for A block, extended
            U = np.zeros((n*m, R))
            # First 7 columns: Strassen modes
            U[0, 0] = 1; U[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            U[0, 2] = 1
            U[3, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            U[1, 6] = 1; U[3, 6] = -1
            # Remaining columns: random
            for r in range(7, R):
                U[:, r] = np.random.randn(n*m) * 0.5
        elif trial == 4:
            # Full random
            U = np.random.randn(n*m, R) * 0.5
        elif trial == 5:
            # Cyclic pattern
            U = np.zeros((n*m, R))
            for r in range(R):
                U[0, r] = 1.0
                U[1, r] = np.cos(2 * np.pi * r / R)
                U[2, r] = np.sin(2 * np.pi * r / R)
                U[3, r] = np.cos(4 * np.pi * r / R)
        else:
            # Random with structure
            U = np.zeros((n*m, R))
            for r in range(R):
                U[0, r] = 1.0
                U[1, r] = np.random.choice([0, 1, 0.5])
                U[2, r] = np.random.choice([0, 1, 0.5])
                U[3, r] = np.random.choice([0, 1, 0.5])
        
        # Solve for V, W
        V, W = _solve_VW_from_U(U, T, n, m, p, R)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

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

def _als_refine(U, V, W, T, R, n, m, p, max_iter=300, tol=1e-14):
    """Run ALS iterations to refine the decomposition."""
    prev_err = np.inf
    for it in range(max_iter):
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
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _search_rank_R(n=2, m=2, p=5, R=16, timeout=120):
    """
    Search for rank-R decomposition using random initialization + ALS.
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
    
    # Increase trials for better coverage, especially for R=17
    max_trials = 50 if R == 17 else 30
    
    for trial in range(max_trials):
        if time.time() - start_time > timeout - 5:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Initialize with Strassen for first 2 cols
        if R >= 7:
            U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
            U[:, :7] = U_str
            V[:, :7] = V_str
            W[:, :7] = W_str
            start_idx = 7
        else:
            start_idx = 0
        
        # Initialize remaining randomly
        # Use different noise scales for diversity
        noise_scale = 0.5 + 0.1 * (trial % 5)
        if start_idx < R:
            U[:, start_idx:] = np.random.randn(n*m, R - start_idx) * noise_scale
            V[:, start_idx:] = np.random.randn(m*p, R - start_idx) * noise_scale
            W[:, start_idx:] = np.random.randn(p*n, R - start_idx) * noise_scale
        
        # Increase iterations for better convergence
        max_iter = 500 if R == 17 else 300
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=max_iter)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Prioritize R=17 as it is more likely to exist and be found than R=16
    # Give it a significant portion of the time
    if time.time() < deadline - 30:
        result = _search_rank_R(n, m, p, R=17, timeout=min(45, deadline - time.time() - 30))
        if result is not None:
            return result
    
    # Try rank 16 with random + ALS
    if time.time() < deadline - 15:
        result = _search_rank_R(n, m, p, R=16, timeout=min(20, deadline - time.time() - 15))
        if result is not None:
            return result
        
    # Try rank 16 with structured U approach (if time permits)
    if time.time() < deadline - 5:
        result = _search_with_structured_U(n, m, p, R=16, timeout=min(5, deadline - time.time() - 5))
        if result is not None:
            return result

    # Try rank 18 (constructive)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
