"""
Program for 4x5 * 5x6 matrix multiplication tensor decomposition.
Uses Simulated Annealing over the discrete space of half-integer entries
to find a rank-minimal exact tensor decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS


def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W


def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    def snap(arr):
        arr2 = 2.0 * arr
        rounded = np.round(arr2)
        if np.allclose(arr2, rounded, atol=tol):
            return rounded / 2.0
        return arr
    U = snap(U)
    V = snap(V)
    W = snap(W)
    return U, V, W


def _sa_discrete(target_rank, max_iter=10000, deadline=None, seed=0):
    """
    Simulated Annealing to find a rank-target_rank decomposition
    with half-integer entries.
    """
    rng = np.random.RandomState(seed)
    
    # Define the set of allowed values (half-integers)
    # Range [-3, 3] seems reasonable for small ranks, might need adjustment
    vals = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Initialize
    U = rng.choice(vals, size=(n * m, target_rank))
    V = rng.choice(vals, size=(m * p, target_rank))
    W = rng.choice(vals, size=(p * n, target_rank))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Annealing parameters
    T_current = 1.0
    T_min = 1e-5
    alpha = 0.9995
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate current error
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = np.sum((T_calc - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-10:
            break
            
        # Propose a move
        mat_idx = rng.randint(0, 3)
        if mat_idx == 0:
            mat = U
            shape = (n * m, target_rank)
        elif mat_idx == 1:
            mat = V
            shape = (m * p, target_rank)
        else:
            mat = W
            shape = (p * n, target_rank)
            
        i = rng.randint(0, shape[0])
        j = rng.randint(0, shape[1])
        
        old_val = mat[i, j]
        
        # Choose a new value from the set of half-integers
        new_val = rng.choice(vals)
        
        if new_val == old_val:
            continue
            
        mat[i, j] = new_val
        
        # Calculate new error
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T_calc_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        new_err = np.sum((T_calc_new - 1.0)**2)
        
        # Accept or reject
        if new_err < err:
            accept = True
        else:
            delta = new_err - err
            if T_current > T_min:
                accept = rng.rand() < np.exp(-delta / T_current)
            else:
                accept = False
                
        if not accept:
            mat[i, j] = old_val
            
        T_current *= alpha
        
    return best_U, best_V, best_W, best_err


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 30
    # We try ranks from high to low to ensure we have a valid candidate early
    # and improve it if possible.
    
    # We will try a range of ranks.
    # If we find a solution at rank R, we continue trying R-5, R-10, etc.
    # If we fail, we keep the previous best.
    
    ranks_to_try = list(range(110, 30, -5))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        found = False
        # Try multiple seeds for robustness
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
            U_try, V_try, W_try, err = _sa_discrete(r, max_iter=5000, deadline=deadline, seed=seed)
            if err < 1e-6:
                # Verify and snap
                U_snap, V_snap, W_snap = _to_integer_or_half(U_try, V_try, W_try)
                if _verify(U_snap, V_snap, W_snap, tol=1e-6):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = r
                    found = True
                    break
                    
        if found:
            # We found a solution at this rank. 
            # The loop continues to try lower ranks.
            pass
            
    # If no better found, return schoolbook
    # Snap to integer/half-integer just in case
    U_out, V_out, W_out = _to_integer_or_half(best_U, best_V, best_W)
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
