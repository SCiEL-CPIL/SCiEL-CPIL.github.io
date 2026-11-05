"""Matrix multiplication 3x3x6 rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T should be 1 when indices are consistent
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Compute the decomposition: sum_r U[:,r] V[:,r] W[:,r]
    # This is a tensor: sum_r U[i,r] * V[j,r] * W[k,r]
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        recon += np.einsum('ir,jr,kr->ijk', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(recon, target, atol=tol)

def _to_rational_approx(vals, max_denom=100):
    """Convert float values to nearest fractions with small denominators."""
    from fractions import Fraction
    result = []
    for v in vals:
        f = Fraction(v).limit_denominator(max_denom)
        result.append(f)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for the multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Strategy: Try ranks from a lower bound up, using numerical optimization
    # Lower bound: The tensor has "fiber" structure. For fixed i, the sub-tensor over (j,k) 
    # is a 3x6 all-ones, which has rank 1. So the total rank is at least... 
    # Actually, let's just try various ranks.
    
    # Known: 3x3x3 matmul has rank 23. 3x3x6 might have rank around 23*2=46 or less.
    # Let's try to find a decomposition using random initialization and gradient descent.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Build target tensor
        target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        # Use alternating least squares / optimization
        # For fixed V, W: U = argmin ||target - sum_r U[:,r] V[:,r] W[:,r]||
        # This is a linear system for U
        
        lr = 0.01
        for it in range(500):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for i_idx in range(n * m):
                # target[i_idx, :, :] = sum_r U[i_idx, r] * outer(V[:,r], W[:,r])
                # So for each r, contribution is U[i_idx,r] * V[:,r] * W[:,r]
                # We want: sum_r U[i_idx,r] * (V[:,r] * W[:,r]) = target[i_idx, :, :]
                # This is a linear system: M * u = t where M[j,k] = V[j,r]*W[k,r] for each r
                # Actually: sum_r U[i_idx,r] * V[j,r] * W[k,r] = target[i_idx,j,k]
                # Let's build the matrix: for each (j,k), the coefficient of U[i_idx,r] is V[j,r]*W[k,r]
                # So we have a system: A * u = b where A is (m*p*p*n) x r
                # Flatten target[i_idx] to a vector
                t_vec = target[i_idx].flatten()
                # Build A: A[j*p_n + k, r] = V[j, r] * W[k, r] where j in [0,m*p), k in [0,p*n)
                # Wait, j indexes V which has m*p rows, k indexes W which has p*n rows
                A = np.zeros((m * p * p * n, r))
                for j in range(m * p):
                    for k in range(p * n):
                        for r_idx in range(r):
                            A[j * (p * n) + k, r_idx] = V[j, r_idx] * W[k, r_idx]
                # Solve A * u = t_vec
                try:
                    u_sol = np.linalg.lstsq(A, t_vec, rcond=None)[0]
                    U[i_idx, :] = u_sol
                except:
                    pass
            
            # Update V
            for j_idx in range(m * p):
                t_vec = target[:, j_idx, :].flatten()
                A = np.zeros((n * m * p * n, r))
                for i in range(n * m):
                    for k in range(p * n):
                        for r_idx in range(r):
                            A[i * (p * n) + k, r_idx] = U[i, r_idx] * W[k, r_idx]
                try:
                    v_sol = np.linalg.lstsq(A, t_vec, rcond=None)[0]
                    V[j_idx, :] = v_sol
                except:
                    pass
            
            # Update W
            for k_idx in range(p * n):
                t_vec = target[:, :, k_idx].flatten()
                A = np.zeros((n * m * m * p, r))
                for i in range(n * m):
                    for j in range(m * p):
                        for r_idx in range(r):
                            A[i * (m * p) + j, r_idx] = U[i, r_idx] * V[j, r_idx]
                try:
                    w_sol = np.linalg.lstsq(A, t_vec, rcond=None)[0]
                    W[k_idx, :] = w_sol
                except:
                    pass
            
            # Check convergence
            recon = np.zeros((n * m, m * p, p * n))
            for r_idx in range(r):
                recon += np.einsum('ir,jr,kr->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            err = np.linalg.norm(recon - target)
            if err < 1e-10:
                # Found a solution! Convert to rational
                U_r = np.array([_to_rational_approx(U[i], 100) for i in range(n*m)])
                V_r = np.array([_to_rational_approx(V[i], 100) for i in range(m*p)])
                W_r = np.array([_to_rational_approx(W[i], 100) for i in range(p*n)])
                
                # Verify with exact arithmetic
                U_f = [[float(x) for x in U_r[i]] for i in range(n*m)]
                V_f = [[float(x) for x in V_r[i]] for i in range(m*p)]
                W_f = [[float(x) for x in W_r[i]] for i in range(p*n)]
                
                if _verify_decomposition(U_f, V_f, W_f, n, m, p, tol=1e-8):
                    return (U_f, V_f, W_f)
                else:
                    # Try with the floating point version directly
                    U_f = U.tolist()
                    V_f = V.tolist()
                    W_f = W.tolist()
                    if _verify_decomposition(U_f, V_f, W_f, n, m, p, tol=1e-8):
                        return (U_f, V_f, W_f)
        
        return None
    
    # Try ranks from lower to higher
    # The minimum possible rank: The tensor has 54 nonzero entries in schoolbook.
    # A lower bound: consider the "slice" rank. For each i, the slice T[i,*,*] is a 3x6 matrix of ones, rank 1.
    # The total tensor rank is at least the max slice rank, which is 1. That's not helpful.
    # Better lower bound: The tensor is a 3x3x6 multiplication tensor. 
    # For 3x3x3, min rank is 23. For 3x3x6, it should be at least 23 (by restriction).
    # Let's try from rank 20 upward.
    
    for r in range(20, 55):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U_f, V_f, W_f = result
            best_result = {"U": U_f, "V": V_f, "W": W_f}
            best_rank = r
            break  # Found the lowest rank, stop
    
    return best_result
