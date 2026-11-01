"""
Tensor decomposition for 2x3 x 3x7 matrix multiplication with rank minimization.

Combines structured algebraic decomposition ideas with iterative refinement:
1. Start with schoolbook rank-42 decomposition.
2. Try structured low-rank decompositions exploiting small m=3.
3. Use alternating least squares with structured initialization.
4. Exact verification with tight tolerance.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - elapsed()


def build_target_tensor():
    """Build the standard matrix multiplication tensor for 2x3 * 3x7."""
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i_A in range(n):
        for j in range(m):
            for k in range(p):
                i_input = i_A * m + j
                j_input = j * p + k
                k_output = k * n + i_A
                T[i_input, j_input, k_output] = 1.0
    return T


def verify_decomposition(U, V, W, T=None, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    Returns (is_valid, max_error).
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.ndim == 1:
        U_arr = U_arr.reshape(nm, -1)
    if V_arr.ndim == 1:
        V_arr = V_arr.reshape(mp, -1)
    if W_arr.ndim == 1:
        W_arr = W_arr.reshape(pn, -1)
    
    rank = U_arr.shape[1]
    
    if T is None:
        T = build_target_tensor()
    
    # Compute the tensor: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    T_computed = np.zeros((nm, mp, pn), dtype=float)
    for t in range(rank):
        T_computed += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
    
    max_err = np.max(np.abs(T_computed - T))
    return max_err < tol, max_err


def schoolbook_decomposition():
    """Generate the trivial schoolbook decomposition of rank 42."""
    n, m, p = 2, 3, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def try_alternating_least_squares(T, target_rank, max_iters=200, seed=0):
    """
    Try to find a rank-target_rank decomposition using alternating least squares.
    Returns (U, V, W, converged) or (None, None, None, False).
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(seed)
    
    U = np.random.randn(nm, target_rank) * 0.5
    V = np.random.randn(mp, target_rank) * 0.5
    W = np.random.randn(pn, target_rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iters):
        if remaining() < 5:
            break
        
        # Update U: T[a,:,:] = sum_t U[a,t] * V[:,t] * W[:,t]
        # M[b,c,t] = V[b,t] * W[c,t], then T[a] = U[a] @ M_flat.T
        M = np.einsum('bt,ct->bct', V, W)  # (mp, pn, target_rank)
        M_flat = M.transpose(2, 0, 1).reshape(target_rank, mp * pn)
        T_flat = T.reshape(nm, mp * pn)
        
        try:
            pinv_M = np.linalg.pinv(M_flat)
            U = (pinv_M @ T_flat.T).T
        except:
            break
        
        # Update V
        M2 = np.einsum('at,ct->act', U, W)  # (nm, pn, target_rank)
        M2_flat = M2.transpose(2, 0, 1).reshape(target_rank, nm * pn)
        T2_flat = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        
        try:
            pinv_M2 = np.linalg.pinv(M2_flat)
            V = (pinv_M2 @ T2_flat.T).T
        except:
            break
        
        # Update W
        M3 = np.einsum('at,bt->abt', U, V)  # (nm, mp, target_rank)
        M3_flat = M3.transpose(2, 0, 1).reshape(target_rank, nm * mp)
        T3_flat = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        try:
            pinv_M3 = np.linalg.pinv(M3_flat)
            W = (pinv_M3 @ T3_flat.T).T
        except:
            break
        
        # Check convergence
        T_computed = np.zeros((nm, mp, pn), dtype=float)
        for t in range(target_rank):
            T_computed += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
        
        err = np.max(np.abs(T_computed - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            return U, V, W, True
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, True
    
    return None, None, None, False


def try_structured_decomposition(T):
    """
    Try structured decompositions exploiting the small m=3 dimension.
    
    For m=3, we can think of this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can try to use linear combinations of A entries and B entries
    to reduce the number of multiplications.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Try various structured patterns
    # For m=3, we can potentially use 5 or 6 multiplications instead of 3
    # by using Strassen-like ideas on the 3-element sum
    
    # Pattern 1: Use (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) style
    # This gives us terms like:
    # (A0+A1)*(B0+B1) = A0*B0 + A0*B1 + A1*B0 + A1*B1
    # Then we need to subtract A0*B1 + A1*B0 and add A2*B2
    
    # Let's try to find a decomposition with rank < 42
    # by exploiting the structure
    
    results = []
    
    # Try ranks from 35 down to 25
    for target_rank in range(35, 24, -1):
        if remaining() < 10:
            break
        
        # Try multiple random initializations with structured bias
        for attempt in range(20):
            if remaining() < 5:
                break
            
            # Initialize with some structure
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Fill with random values but with some structure
            for t in range(target_rank):
                # Random pattern
                U[:, t] = np.random.choice([-1, 0, 1], size=nm) * 0.5
                V[:, t] = np.random.choice([-1, 0, 1], size=mp) * 0.5
                W[:, t] = np.random.choice([-1, 0, 1], size=pn) * 0.5
                
                # Ensure non-trivial
                if np.all(U[:, t] == 0) or np.all(V[:, t] == 0) or np.all(W[:, t] == 0):
                    U[:, t] = np.random.randn(nm) * 0.5
                    V[:, t] = np.random.randn(mp) * 0.5
                    W[:, t] = np.random.randn(pn) * 0.5
            
            # Use ALS to refine
            U_new, V_new, W_new, converged = try_alternating_least_squares(
                T, target_rank, max_iters=100, seed=attempt
            )
            
            if converged:
                results.append((target_rank, U_new, V_new, W_new))
                return target_rank, U_new, V_new, W_new
    
    return None, None, None, False


def try_exact_rational_search(T):
    """
    Try to find an exact decomposition using rational numbers.
    This is more reliable than floating point for exact verification.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Try small integer/half-integer values
    # This is a heuristic approach
    
    for target_rank in range(30, 20, -1):
        if remaining() < 10:
            break
        
        # Try a few random rational initializations
        for attempt in range(5):
            if remaining() < 5:
                break
            
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
            
            # Refine with ALS
            U_new, V_new, W_new, converged = try_alternating_least_squares(
                T, target_rank, max_iters=50, seed=1000 + attempt
            )
            
            if converged:
                return target_rank, U_new, V_new, W_new
    
    return None, None, None, False


def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    best_result = schoolbook_decomposition()
    best_rank = 42
    T = build_target_tensor()
    
    # First, try structured decomposition
    if remaining() > 30:
        rank, U, V, W = try_structured_decomposition(T)
        if rank is not None:
            valid, err = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), T)
            if valid and rank < best_rank:
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Try exact rational search
    if remaining() > 20:
        rank, U, V, W = try_exact_rational_search(T)
        if rank is not None:
            valid, err = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), T)
            if valid and rank < best_rank:
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Try direct ALS from various ranks
    if remaining() > 15:
        for target_rank in range(35, 25, -1):
            if remaining() < 10:
                break
            
            for seed in range(10):
                if remaining() < 5:
                    break
                
                U, V, W, converged = try_alternating_least_squares(
                    T, target_rank, max_iters=150, seed=seed + 2000
                )
                
                if converged:
                    valid, err = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), T)
                    if valid and target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break
    
    # Final verification
    valid, max_err = verify_decomposition(
        best_result["U"], best_result["V"], best_result["W"], T
    )
    if not valid:
        # Fallback to schoolbook
        best_result = schoolbook_decomposition()
    
    return best_result
