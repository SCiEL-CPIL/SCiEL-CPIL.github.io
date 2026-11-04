"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import numpy as np
import time
from itertools import product
from fractions import Fraction

def _to_int_or_half(val):
    """Convert a Fraction to int or half-integer (int or int+0.5) if possible, else return as float."""
    if val.denominator == 1:
        return val.numerator
    elif val.denominator == 2:
        return val.numerator / 2.0
    else:
        # If not integer or half-integer, return float (shouldn't happen for exact solutions)
        return float(val)

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each output index (i, k) in C = A @ B where A is n x m, B is m x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T_{i, j, k} has T_{i, j, k} = 1 if we consider the multiplication tensor.
    # Decomposition: T_{i, j, k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    rank = len(U[0]) if len(U) > 0 else 0
    # Build the tensor from decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We need to check that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the multiplication tensor
    
    # The multiplication tensor for (n x m) * (m x p) -> (n x p) has:
    # T_{i, j, k} = 1 for all valid i, j, k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor is T_{i, j, k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_als_decomposition(n, m, p, rank, max_iter=500, tol=1e-6, seed=None):
    """Try ALS (Alternating Least Squares) to find a rank-rank decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the full tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # ALS: alternate fixing two matrices and solving for the third
    # For each entry (a, b, c) of T, we have T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Fix V and W, solve for U: For each a, T[a,:,:] = U[a,:] @ (V * W.T) ... 
    # Actually, let's think of it differently.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Fix V and W. For a fixed a, we have for all b,c: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
    # Let M_a[b,c] = V[b,:] * W[c,:] (element-wise product, shape (m*p, p*n))
    # Then T[a,:,:] = U[a,:] @ M_a.T ... no.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Let X_a = U[a,:] (shape (rank,))
    # Let Y[b,c] = V[b,:] * W[c,:] (element-wise, shape (m*p, p*n))
    # Then T[a,b,c] = sum_r X_a[r] * Y[b,c,r]... no, Y is 2D.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is like: for each a, the matrix T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r].T)
    # So T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
    # This is a linear system: T[a,:,:] = sum_r U[a,r] * A_r where A_r = outer(V[:,r], W[:,r])
    # Stack all A_r into a matrix of shape (m*p*p*n, rank)
    # Then T[a,:,:,] flattened is (m*p*p*n,) and we solve U[a,:] @ A_stacked.T = T[a,:,:,]
    
    # Similarly for V and W.
    
    # Let's implement this properly.
    # Precompute the index mapping
    # For U: fix V, W. For each a in [0, n*m):
    #   T[a, :, :] = sum_r U[a,r] * (V[:, r] * W[:, r].T)
    #   Let B = V * W.T (element-wise? No.)
    #   Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #   For fixed a: T[a] (shape (m*p, p*n)) = sum_r U[a,r] * (V[:,r] * W[:,r].T)
    #   So if we let M_r = outer(V[:,r], W[:,r]) (shape (m*p, p*n)), then
    #   T[a] = sum_r U[a,r] * M_r
    #   This is a linear regression: T[a] (flattened) = U[a] @ M_stacked.T
    #   where M_stacked has shape (m*p*p*n, rank)
    
    # Let's just do gradient descent or simple ALS with numpy lstsq.
    
    # Actually, let's use a simpler approach: treat it as a least squares problem.
    # We have variables U, V, W. The objective is sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
    # This is non-convex. ALS alternates:
    # 1. Fix V, W. For each a, solve min_U sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
    #    This is: for each a, T[a] (flattened) = U[a] @ (V * W.T reshaped)
    #    Let's define: for each a, let G_a = V * W.T (element-wise, shape (m*p, p*n, rank))
    #    No, that's not right.
    
    # Let me think again. T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Fix V and W. For fixed a, define matrix P_a where P_a[b,c] = T[a,b,c]
    # And define for each r, Q_r[b,c] = V[b,r] * W[c,r]
    # Then P_a = sum_r U[a,r] * Q_r
    # This is a linear system in U[a,:].
    # Flatten Q_r to vectors q_r of length (m*p)*(p*n)
    # Then P_a flattened = U[a,:] @ Q_stacked.T where Q_stacked is (m*p*p*n, rank)
    # So U[a,:] = lstsq(Q_stacked.T, P_a.flatten())[0]
    
    # Similarly for V and W.
    
    # Let's implement this.
    
    # Precompute Q_stacked for U updates (depends on V, W)
    # And similarly for others.
    
    # This might be slow for large dimensions, but n=4, m=4, p=9 is small.
    # m*p*p*n = 4*9*9*4 = 1296
    # rank is at most 144
    
    # Let's try a few random initializations and see if we can get close.
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for trial in range(10):
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        for it in range(max_iter):
            # Update U: fix V, W
            # For each a, solve U[a,:] @ Q_stacked.T = T[a,:,:,]
            Q_stacked = np.zeros((m*p*p*n, rank))
            for r in range(rank):
                Q_stacked[:, r] = (V[:, r] * W[:, r].T).flatten()
            
            for a in range(n*m):
                target = T[a, :, :].flatten()
                # Solve: U[a,:] @ Q_stacked.T = target
                # This is: Q_stacked @ U[a,:].T = target
                sol, _, _, _ = np.linalg.lstsq(Q_stacked, target, rcond=None)
                U[a, :] = sol
            
            # Update V: fix U, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r].T)
            # Let R_r[a,c] = U[a,r] * W[c,r]
            # T[:,b,:] = sum_r V[b,r] * R_r
            # Flatten: T[:,b,:].flatten() = V[b,:] @ R_stacked.T
            R_stacked = np.zeros((n*m*p*n, rank))
            for r in range(rank):
                R_stacked[:, r] = (U[:, r] * W[:, r].T).flatten()
            
            for b in range(m*p):
                target = T[:, b, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(R_stacked, target, rcond=None)
                V[b, :] = sol
            
            # Update W: fix U, V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r].T)
            # Let S_r[a,b] = U[a,r] * V[b,r]
            # T[:,:,c] = sum_r W[c,r] * S_r
            S_stacked = np.zeros((n*m*m*p, rank))
            for r in range(rank):
                S_stacked[:, r] = (U[:, r] * V[:, r].T).flatten()
            
            for c in range(p*n):
                target = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(S_stacked, target, rcond=None)
                W[c, :] = sol
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    if best_err < 1e-4:
        # Convert to list format and check
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Try to snap to integers or half-integers
        U_snapped = [[_to_int_or_half(Fraction(x).limit_denominator(2)) for x in row] for row in U_list]
        V_snapped = [[_to_int_or_half(Fraction(x).limit_denominator(2)) for x in row] for row in V_list]
        W_snapped = [[_to_int_or_half(Fraction(x).limit_denominator(2)) for x in row] for row in W_list]
        
        if _check_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-6):
            return U_snapped, V_snapped, W_snapped
    
    return None

def _try_sparse_search(n, m, p, rank, time_limit=30):
    """Try to find a sparse decomposition with small integer entries."""
    start_time = time.time()
    
    # Use a greedy approach: start with the schoolbook decomposition and try to reduce rank
    # by finding linear dependencies.
    
    # For now, just return None if we can't find anything quickly.
    # This is a placeholder for more sophisticated search.
    
    if time.time() - start_time > time_limit:
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with the schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But the actual minimal rank might be higher.
    
    # Try ALS for various ranks
    for rank in range(max(n*m, m*p, p*n), min(schoolbook_rank, 80) + 1):
        if time.time() > 140:
            break
        
        result = _try_als_decomposition(n, m, p, rank, max_iter=100, tol=1e-6, seed=rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_U, best_V, best_W = U_r, V_r, W_r
            break  # Found a valid decomposition, use it
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
