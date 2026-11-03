import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Reshape to 3D for efficient einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by k*n+i, so W3[k,i,r]
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

def _als_step(U, V, W, n, m, p, fix='U'):
    """
    Perform one step of Alternating Least Squares.
    Fix two matrices, solve for the third to minimize residual.
    """
    r = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    if fix == 'U':
        # Solve for U
        # For each i, j: sum_c U[i*m+j, c] * (V[j*p+?, c] * W[?, i, c]) ... 
        # A_c = sum_k V[j*p+k, c] * W[k*n+i, c]
        # System: A_{ij} u_{ij} = 1_vector (size p)
        # A_{ij} is p x r
        for i in range(n):
            for j in range(m):
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # V[j*p+k, :] has shape (r,)
                # W[k*n+i, :] has shape (r,)
                # We need to construct A of shape (p, r)
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A u = b
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = u_sol
                except np.linalg.LinAlgError:
                    pass
                    
    elif fix == 'V':
        # Solve for V
        # For each j, k: sum_c V[j*p+k, c] * (U[?, j, c] * W[k, ?, c])
        # A_c = sum_i U[i*m+j, c] * W[k*n+i, c]
        # System: B_{jk} v_{jk} = 1_vector (size n)
        # B_{jk} is n x r
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, r))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    V_new[j*p+k, :] = v_sol
                except np.linalg.LinAlgError:
                    pass

    elif fix == 'W':
        # Solve for W
        # For each k, i: sum_c W[k*n+i, c] * (U[?, ?, c] * V[?, k, c])
        # A_c = sum_j U[i*m+j, c] * V[j*p+k, c]
        # System: C_{ki} w_{ki} = 1_vector (size m)
        # C_{ki} is m x r
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, r))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                    W_new[k*n+i, :] = w_sol
                except np.linalg.LinAlgError:
                    pass

    if fix == 'U':
        return U_new, V, W
    elif fix == 'V':
        return U, V_new, W
    else:
        return U, V, W_new

def _als_optimize(n, m, p, r, U0, V0, W0, max_iters=200, tol=1e-12):
    """
    Run Alternating Least Squares to minimize the residual.
    """
    U, V, W = U0.copy(), V0.copy(), W0.copy()
    
    prev_err = float('inf')
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Cycle through fixing U, V, W
        U, V, W = _als_step(U, V, W, n, m, p, fix='U')
        U, V, W = _als_step(U, V, W, n, m, p, fix='V')
        U, V, W = _als_step(U, V, W, n, m, p, fix='W')
        
        # Calculate error
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.sum((S - 1.0)**2)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _try_rank_r(n, m, p, r, max_restarts=1000):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Initialize with random small integers or floats
        # Using random normal distribution often works well for ALS
        U0 = np.random.randn(n*m, r)
        V0 = np.random.randn(m*p, r)
        W0 = np.random.randn(p*n, r)
        
        # Occasionally try sparse/integer-like initialization
        if restart % 5 == 0:
             U0 = np.random.choice([-1, 0, 1], size=(n*m, r))
             V0 = np.random.choice([-1, 0, 1], size=(m*p, r))
             W0 = np.random.choice([-1, 0, 1], size=(p*n, r))
             # Add small noise to avoid exact zeros leading to singular matrices in ALS
             U0 += 0.01 * np.random.randn(n*m, r)
             V0 += 0.01 * np.random.randn(m*p, r)
             W0 += 0.01 * np.random.randn(p*n, r)

        U_sol, V_sol, W_sol = _als_optimize(n, m, p, r, U0, V0, W0)
        
        # Calculate final error
        U3 = U_sol.reshape(n, m, r)
        V3 = V_sol.reshape(m, p, r)
        W3 = W_sol.reshape(p, n, r)
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.sum((S - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
            
            # Check for exact solution after snapping
            # Try half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
                
            # Try integers
            U_snap2 = _snap_to_integers(U_sol)
            V_snap2 = _snap_to_integers(V_sol)
            W_snap2 = _snap_to_integers(W_sol)
            if _verify_decomposition(U_snap2, V_snap2, W_snap2, n, m, p):
                return U_snap2, V_snap2, W_snap2

    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # 2. Try to find lower ranks
    # We try ranks from 35 down to 30.
    for r_target in [35, 34, 33, 32, 31, 30]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Allocate more restarts for higher ranks (easier), fewer for lower
        # ALS is fast, so we can afford many restarts
        if r_target >= 34:
            max_restarts = 500
        else:
            max_restarts = 200
            
        result = _try_rank_r(n, m, p, r_target, max_restarts=max_restarts)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r_target
                break # Found a valid lower rank, stop searching lower

    # 3. Select best
    if best_U is None:
        best_U, best_V, best_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
