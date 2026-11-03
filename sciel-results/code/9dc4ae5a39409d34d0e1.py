import time
import numpy as np
import scipy.optimize as optimize

START_TIME = time.time()
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape to 3D for efficient einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if S is all 1s
    return np.allclose(S, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _snap_to_integers(matrix):
    """Snap matrix entries to nearest integer."""
    return np.round(matrix)

def _polish_solution(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """
    Try to polish a near-exact solution to exact half-integers.
    Uses a local search on the snapped values.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    
    # First, snap to half-integers
    U = _snap_to_half_integers(U)
    V = _snap_to_half_integers(V)
    W = _snap_to_half_integers(W)
    
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        return U, V, W
    
    # Try integer snapping
    U_int = _snap_to_integers(U)
    V_int = _snap_to_integers(V)
    W_int = _snap_to_integers(W)
    
    if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return U_int, V_int, W_int
    
    # Local search: try small perturbations
    best_U, best_V, best_W = U, V, W
    best_loss = float('inf')
    
    for _ in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Compute current loss
        U3 = U.reshape(n, m, U.shape[1])
        V3 = V.reshape(m, p, V.shape[1])
        W3 = W.reshape(p, n, W.shape[1])
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        loss = np.sum((S - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Try perturbing a random entry
        which = np.random.randint(0, 3)
        if which == 0:
            idx = np.random.randint(0, U.size)
            row, col = idx // U.shape[1], idx % U.shape[1]
            U[row, col] += np.random.choice([-0.5, 0.5])
        elif which == 1:
            idx = np.random.randint(0, V.size)
            row, col = idx // V.shape[1], idx % V.shape[1]
            V[row, col] += np.random.choice([-0.5, 0.5])
        else:
            idx = np.random.randint(0, W.size)
            row, col = idx // W.shape[1], idx % W.shape[1]
            W[row, col] += np.random.choice([-0.5, 0.5])
        
        U = _snap_to_half_integers(U)
        V = _snap_to_half_integers(V)
        W = _snap_to_half_integers(W)
    
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        return best_U, best_V, best_W
    
    return None

def _solve_rank_r(n, m, p, r, max_restarts=200):
    """
    Attempt to find a rank-r decomposition for n x m x p matmul.
    Uses least squares with multiple restarts and polishing.
    Returns (U, V, W) or None.
    """
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def objective(params):
        U = params[:n_u*r].reshape(n, m, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(m, p, r)
        W = params[n_u*r + n_v*r :].reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return (S - 1.0).flatten()
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Random initialization with varying scales
        scale = np.random.choice([0.5, 1.0, 2.0])
        U0 = np.random.randn(n, m, r) * scale
        V0 = np.random.randn(m, p, r) * scale
        W0 = np.random.randn(p, n, r) * scale
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=150)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n_u*r].reshape(n, m, r)
            V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(m, p, r)
            W_sol = res.x[n_u*r + n_v*r :].reshape(p, n, r)
            
            # Try polishing
            polished = _polish_solution(U_sol, V_sol, W_sol, n, m, p)
            if polished is not None:
                U_p, V_p, W_p = polished
                return U_p.reshape(n_u, r), V_p.reshape(n_v, r), W_p.reshape(n_w, r)
    
    return None

def _try_structural_decomposition(n, m, p, target_rank):
    """
    Try to construct a decomposition using structural properties.
    For 2x3x6, exploit that 6 = 2*3.
    """
    # For n=2, m=3, p=6, we can try to use the fact that p = n*m
    # This might allow a more efficient decomposition
    
    # Approach: use the identity that for p = n*m, we can potentially
    # reduce the rank by exploiting the tensor structure
    
    # Try a direct construction with rank = n*m*p - something
    # For now, try to find a decomposition with rank < n*m*p
    
    # One idea: use the fact that the tensor has symmetries
    # For 2x3x6, the tensor T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
    
    # Let's try a specific construction:
    # Split k into k1 (0..1) and k2 (0..2), so k = k1*3 + k2
    # Then C[i, k1*3+k2] = sum_j A[i,j] B[j, k1*3+k2]
    
    # This doesn't immediately give a rank reduction, but let's try
    # a different approach: use bilinear decomposition
    
    # For now, return None and rely on the numerical search
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # 2. Try structural decomposition first
    if time.time() - START_TIME < TIME_LIMIT - 30:
        result = _try_structural_decomposition(n, m, p, best_rank - 1)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = U_try.shape[1]
    
    # 3. Try direct numerical search for lower ranks
    if best_rank == n * m * p:
        # Try ranks from 35 down to 25
        for r_target in range(best_rank - 1, 24, -1):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            result = _solve_rank_r(n, m, p, r_target, max_restarts=100)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = r_target
                    break
    
    # 4. Final verification and cleanup
    if best_U is None:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Ensure we return the best found
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if verification fails
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
