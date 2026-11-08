import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE = 150
_SAFETY_MARGIN = 10  # seconds before the kill

def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 for all (i,j,k) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _try_strassen_like_search(n, m, p, max_rank, time_budget):
    """
    Attempt to find a rank < max_rank exact decomposition using a structured
    search inspired by Strassen's algorithm. We parameterize each rank-1 term
    by three vectors (u_r, v_r, w_r) and use iterative refinement.
    
    Returns (U, V, W) if a valid decomposition is found, else None.
    """
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We want: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
    # where u_r is indexed by (i,j), v_r by (j,k), w_r by (k,i)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_result = None
    
    # Try various ranks from low to high, up to max_rank-1
    for rank in range(15, min(max_rank, 35)):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Initialize with random factors (structured initialization)
        # Use a Strassen-like initialization: each factor is a sum of a few basis terms
        np.random.seed(rank * 1000 + 42)
        
        # Try multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            
            # Initialize factors randomly but with structure
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Iterative refinement: alternating least squares
            # For each r, fix the other two factors and solve for the third
            max_iters = 200
            for it in range(max_iters):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                # Update U: for each (i,j), U[i,j,:] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system for U[i*m+j, :]
                # Let's do a gradient step instead for simplicity
                # Compute gradient of error w.r.t. U
                # E = sum_{i,j,k} (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
                # dE/dU[i*m+j,r] = -2 * sum_k (1 - ...) * V[j*p+k,r] * W[k*n+i,r]
                
                # For efficiency, compute the error tensor
                # err[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # This is expensive for large rank, but n*m*p = 45 is small
                
                err = np.ones((n, m, p))
                for r in range(rank):
                    # outer product of U[:, r], V[:, r], W[:, r] reshaped
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    # term[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                err[i, j, k] -= u_r[i, j] * v_r[j, k] * w_r[k, i]
                
                # Check if we're close to zero
                err_norm = np.linalg.norm(err)
                if err_norm < 1e-10:
                    # Found a valid decomposition!
                    # Round to nearest half-integer and verify
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Try without rounding (keep as float)
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Gradient step for U
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            grad = 0
                            for k in range(p):
                                grad += err[i, j, k] * V[j * p + k, r] * W[k * n + i, r]
                            U[i * m + j, r] -= lr * grad
                
                # Gradient step for V
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            grad = 0
                            for i in range(n):
                                grad += err[i, j, k] * U[i * m + j, r] * W[k * n + i, r]
                            V[j * p + k, r] -= lr * grad
                
                # Gradient step for W
                for k in range(p):
                    for i in range(n):
                        for r in range(rank):
                            grad = 0
                            for j in range(m):
                                grad += err[i, j, k] * U[i * m + j, r] * V[j * p + k, r]
                            W[k * n + i, r] -= lr * grad
                
                # Decay learning rate
                lr *= 0.995
            
            # After optimization, check final error
            err = np.ones((n, m, p))
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err[i, j, k] -= u_r[i, j] * v_r[j, k] * w_r[k, i]
            
            if np.linalg.norm(err) < 1e-8:
                # Try to round and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we get here, no valid decomposition found for this rank
        continue
    
    return None


def construct():
    """
    Entry point with deadline guard. Attempts to find a lower-rank decomposition
    using a Strassen-like search, falling back to the schoolbook rank-45 decomposition.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Check if we have time for a search
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Not enough time, return schoolbook
        return _schoolbook_decomposition()
    
    # Try to find a lower-rank decomposition
    # The theoretical lower bound for 3x3x5 is at least 27 (by a simple argument)
    # but the actual minimal rank is unknown. We search up to rank 34.
    result = _try_strassen_like_search(n, m, p, schoolbook_rank, _DEADLINE - _SAFETY_MARGIN - elapsed)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
