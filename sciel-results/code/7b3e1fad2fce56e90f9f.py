import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta(i,j,k)"""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    # T[i,j,k] should be 1 if i==j==k... wait, no.
    # For matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} = 1 if the "multiplication" connects (i,j) in U, (j,k) in V, (k,i) in W
    # Actually the convention is: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1 for ALL i,j,k
    # Wait no - it should equal 1 only when the indices form a valid multiplication chain.
    # Actually for the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k
    # because every (i,j,k) triple contributes to the multiplication.
    # Let me re-read: the tensor is the structure tensor of matrix multiplication.
    # T_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    # This means the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    expected = np.ones((n, m, p))
    error = np.max(np.abs(T - expected))
    return error < tol, error

def _compute_max_error(U, V, W, n, m, p):
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    return np.max(np.abs(T - 1.0))

def _objective(params, n, m, p, rank, beta=50.0):
    """Smooth objective using log-sum-exp for the max error."""
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.abs(T - 1.0)
    
    # Log-sum-exp for smooth max
    max_err = np.max(err)
    if max_err < 1e-15:
        return 0.0
    
    shifted = beta * err
    shifted_max = np.max(shifted)
    log_sum_exp = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
    soft_max = log_sum_exp / beta
    
    # Small regularization to avoid degenerate solutions
    reg = 1e-8 * (np.sum(U_flat**2) + np.sum(V_flat**2) + np.sum(W_flat**2))
    return soft_max + reg

def _snap_and_verify(U, V, W, n, m, p):
    """Snap to half-integers and verify."""
    U_snap = np.round(U * 2) / 2.0
    V_snap = np.round(V * 2) / 2.0
    W_snap = np.round(W * 2) / 2.0
    valid, err = _verify(U_snap, V_snap, W_snap, n, m, p)
    if valid:
        return U_snap, V_snap, W_snap
    return None, None, None

def _try_optimize_block(U_init, V_init, W_init, n, m, p, rank, deadline, max_time=8.0):
    """Try to optimize a fixed-rank decomposition using L-BFGS-B."""
    start = time.time()
    timeout = min(deadline - start, max_time)
    if timeout < 1.0:
        return None, None, None, 1.0
    
    U_flat = U_init.flatten()
    V_flat = V_init.flatten()
    W_flat = W_init.flatten()
    params = np.concatenate([U_flat, V_flat, W_flat])
    
    try:
        from scipy.optimize import minimize
        result = minimize(
            _objective,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if time.time() - start > timeout:
            return None, None, None, 1.0
        
        final_params = result.x
        U_opt = final_params[:n*m*rank].reshape(n * m, rank)
        V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
        
        err = _compute_max_error(U_opt, V_opt, W_opt, n, m, p)
        return U_opt, V_opt, W_opt, err
        
    except Exception:
        return None, None, None, 1.0

def _polish(U, V, W, n, m, p, deadline):
    """High-precision polish pass."""
    if time.time() > deadline - 5:
        return None, None, None
    
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    params = np.concatenate([U_flat, V_flat, W_flat])
    rank = U.shape[1]
    
    try:
        from scipy.optimize import minimize
        result = minimize(
            _objective,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if time.time() > deadline - 3:
            return None, None, None
        
        final_params = result.x
        U_opt = final_params[:n*m*rank].reshape(n * m, rank)
        V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
        
        err = _compute_max_error(U_opt, V_opt, W_opt, n, m, p)
        if err < 1e-10:
            U_snap, V_snap, W_snap = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
            if U_snap is not None:
                return U_snap, V_snap, W_snap
        return None, None, None
    except Exception:
        return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook fallback for full 3x3x4
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Strategy: Decompose M_{3,3,4} as 2 copies of M_{3,3,2}
    # M_{3,3,2}: n2=3, m2=3, p2=2
    n2, m2, p2 = 3, 3, 2
    
    # Lower bound for rank of M_{3,3,2}: max(n2*m2, m2*p2, p2*n2) = max(9, 6, 6) = 9
    # Try R2 from 9 upward
    found_decomp = None
    for R2 in range(9, 19):
        if time.time() > deadline - 10:
            break
        
        # Multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with various scales
            if trial % 3 == 0:
                scale = 1.0
            elif trial % 3 == 1:
                scale = 0.5
            else:
                scale = 2.0
            
            U2 = np.random.randn(n2 * m2, R2) * scale
            V2 = np.random.randn(m2 * p2, R2) * scale
            W2 = np.random.randn(p2 * n2, R2) * scale
            
            # Optimize
            U_opt, V_opt, W_opt, err = _try_optimize_block(U2, V2, W2, n2, m2, p2, R2, deadline, max_time=6.0)
            
            if U_opt is None:
                continue
            
            # Check if we got a good solution
            if err < 1e-8:
                # Try snapping
                U_snap, V_snap, W_snap = _snap_and_verify(U_opt, V_opt, W_opt, n2, m2, p2)
                if U_snap is not None:
                    found_decomp = (U_snap, V_snap, W_snap, R2)
                    break
                
                # If snapping fails, try polish
                if time.time() < deadline - 10:
                    U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, n2, m2, p2, deadline)
                    if U_pol is not None:
                        found_decomp = (U_pol, V_pol, W_pol, R2)
                        break
            
            elif err < 0.1:
                # Close but not exact - try polish
                if time.time() < deadline - 10:
                    U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, n2, m2, p2, deadline)
                    if U_pol is not None:
                        found_decomp = (U_pol, V_pol, W_pol, R2)
                        break
        
        if found_decomp:
            break
    
    if found_decomp:
        U2, V2, W2, R2 = found_decomp
        R_total = 2 * R2
        
        if R_total < best_rank:
            # Construct U, V, W for 3x3x4
            U = np.zeros((n * m, R_total))
            V = np.zeros((m * p, R_total))
            W = np.zeros((p * n, R_total))
            
            # First half: corresponds to k=0,1 (first two columns of B)
            U[:, :R2] = U2
            for j in range(m):
                for k in range(2):
                    V[j * p + k, :R2] = V2[j * m2 + k]
            for k in range(2):
                for i in range(n):
                    W[k * n + i, :R2] = W2[k * n2 + i]
            
            # Second half: corresponds to k=2,3 (last two columns of B)
            U[:, R2:] = U2
            for j in range(m):
                for k in range(2, 4):
                    V[j * p + k, R2:] = V2[j * m2 + (k - 2)]
            for k in range(2, 4):
                for i in range(n):
                    W[k * n + i, R2:] = W2[(k - 2) * n2 + i]
            
            # Final verification
            valid, err = _verify(U, V, W, n, m, p)
            if valid:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R_total
    
    return best_result
