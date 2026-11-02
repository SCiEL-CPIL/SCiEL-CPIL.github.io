import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from itertools import product
import random

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    All other entries are 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2 where U,V,W are 2D arrays (rows x rank).
    The decomposition is T = sum_r U[:,r] V[:,r] W[:,r]^T
    which in matrix form for the first two indices is:
    T[:,:,c] = U @ (V * W[:,c])^T for each c? No.
    
    Actually T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    So for fixed k, the matrix M_k of size (n*m x m*p) is U @ diag(V[:,k]??) no.
    
    Let's reshape: T is (A, B, C).
    The decomposition means T = sum_r U[:,r] outer V[:,r] outer W[:,r].
    We can compute the error by iterating over r, or by using einsum.
    np.einsum('ar,br,cr->abc', U, V, W)
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, T, seed=0, time_limit=10.0):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Normalize to prevent scaling issues
    norms_U = np.linalg.norm(U, axis=0, keepdims=True)
    norms_V = np.linalg.norm(V, axis=0, keepdims=True)
    norms_W = np.linalg.norm(W, axis=0, keepdims=True)
    norms_U[norms_U == 0] = 1.0
    norms_V[norms_V == 0] = 1.0
    norms_W[norms_W == 0] = 1.0
    U /= norms_U
    V /= norms_V
    W /= norms_W

    best_err = _tensor_error(U, V, W, T)
    
    # Alternating Least Squares (ALS)
    start_time = time.time()
    iterations = 0
    max_iterations = 5000
    
    # We need to solve for one factor at a time.
    # For U: T_reshaped (flattened B*C) = U @ (V * W)^T ? No.
    # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
    # For fixed i: sum_r U[i, r] (V[:,r] * W[:,r])^T? No.
    # Let's reshape T to (n*m, m*p * p*n)
    # T_flat = T.reshape(n*m, m*p*p*n)
    # T_flat[i, (b, c)] = sum_r U[i, r] V[b, r] W[c, r]
    # This is not a simple matrix product for U because V and W are coupled in the second index.
    # However, if we fix V and W, for each i, we have a linear system:
    # T[i, :, :] = U[i, :] @ ? No.
    # T[i, b, c] = sum_r U[i, r] * (V[b, r] * W[c, r])
    # Let M_i be a matrix of size (m*p * p*n) x rank.
    # M_i[(b, c), r] = V[b, r] * W[c, r]
    # Then T[i, :, :] flattened is U[i, :] @ M_i.
    # So U[i, :] = T[i, :, :].flatten() @ M_i^+
    
    # Similarly for V and W.
    
    # To make this efficient, we can precompute the "basis" matrices.
    # But since V and W change, we have to recompute.
    
    # Alternative: Use gradient descent with Adam or similar.
    # Given the small dimensions (10x25x10), direct construction of gradients might be feasible.
    
    # Let's stick to ALS but implement it carefully.
    # Reshape T to (n*m, m*p, p*n)
    
    while iterations < max_iterations:
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        # For each i, solve U[i, :] M_i = T[i, :, :].flatten()
        # M_i is (m*p*p*n) x rank.
        # M_i[(b, c), r] = V[b, r] * W[c, r]
        
        # We can build the full matrix M of shape (n*m, m*p*p*n, rank) ? Too big?
        # n*m=10, m*p*p*n=250, rank <= 50. 10*250*50 = 125,000 floats. This is fine.
        
        M_U = np.zeros((n*m, m*p*p*n, rank), dtype=np.float64)
        # V is (m*p, rank), W is (p*n, rank)
        # We need V[b, r] * W[c, r]
        # Let's create V_rep and W_rep
        # V_rep shape (m*p, 1, rank) -> (m*p, 1, rank)
        # W_rep shape (1, p*n, rank) -> (1, p*n, rank)
        # Product shape (m*p, p*n, rank)
        # Reshape to (m*p*p*n, rank)
        
        V_rep = V[:, np.newaxis, :]  # (m*p, 1, rank)
        W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
        MW = V_rep * W_rep  # (m*p, p*n, rank)
        MW_flat = MW.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
        
        # M_U[i, :, :] = MW_flat for all i?
        # Wait, M_i depends on V and W, which are global.
        # So M_U[i, :, :] is the same for all i?
        # Yes! The basis for the linear combination is the same for every output index i.
        # The only thing that changes is the target vector T[i, :, :].
        # So U = T.reshape(n*m, -1) @ pinv(MW_flat)
        
        T_flat = T.reshape(n*m, m*p*p*n)
        # Solve U T_flat = U @ MW_flat
        # U = T_flat @ pinv(MW_flat)
        # Use lstsq for stability
        U_new, _, _, _ = lstsq(MW_flat, T_flat.T, cond=None)
        U = U_new.T  # (n*m, rank)
        
        # Update V
        # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
        # For fixed b: T[:, b, :] flattened (n*m * p*n) = V[b, :] @ M_V
        # M_V[(i, c), r] = U[i, r] * W[c, r]
        
        U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
        W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
        MW = U_rep * W_rep  # (n*m, p*n, rank)
        MW_flat = MW.reshape(n*m*p*n, rank)
        
        T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n) # (m*p, n*m*p*n)
        V_new, _, _, _ = lstsq(MW_flat, T_flat_V.T, cond=None)
        V = V_new.T  # (m*p, rank)
        
        # Update W
        # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
        # For fixed c: T[:, :, c] flattened (n*m * m*p) = W[c, :] @ M_W
        # M_W[(i, b), r] = U[i, r] * V[b, r]
        
        U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
        V_rep = V[np.newaxis, :, :]  # (1, m*p, rank)
        MW = U_rep * V_rep  # (n*m, m*p, rank)
        MW_flat = MW.reshape(n*m*m*p, rank)
        
        T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p) # (p*n, n*m*m*p)
        W_new, _, _, _ = lstsq(MW_flat, T_flat_W.T, cond=None)
        W = W_new.T  # (p*n, rank)
        
        iterations += 1
        err = _tensor_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Convergence check
        if best_err < 1e-10:
            break
            
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    best_err = 0.0
    T = _build_tensor(n, m, p)
    
    # Try to reduce rank from n*m*p - 1 downwards
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    # But known results might be lower.
    # Let's try ranks: 49, 48, ... down to 25.
    
    # To save time, we won't try all ranks. We'll try a few promising ones.
    # Ranks to try: 49, 45, 40, 35, 30, 25
    
    candidate_ranks = [49, 45, 40, 35, 30, 28, 26, 25]
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        success = False
        # Try multiple seeds for each rank
        for seed in range(5):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
                
            time_limit = 10.0
            try:
                U_new, V_new, W_new, err = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if err < 1e-8:
                    # Check for exactness with tighter tolerance
                    # Re-verify
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-10:
                        U, V, W = U_new, V_new, W_new
                        success = True
                        break
            except Exception:
                continue
                
        if success:
            # Found a solution for this rank. Can we go lower?
            # Yes, continue loop to next lower rank
            continue
        else:
            # Failed to find solution for this rank.
            # Should we try lower? Probably not if we assume monotonicity of difficulty,
            # but sometimes lower rank is easier? Unlikely.
            # However, if we failed at 49, maybe 48 is also hard.
            # Let's stop searching for lower ranks if we failed at a higher one?
            # No, it's possible that a specific rank is "hard" due to local minima,
            # while a lower one is "easy". But generally, lower rank is harder.
            # To be safe, let's try the next lower rank only if we have time.
            # Actually, if 49 fails, 48 is likely to fail too.
            # Let's just break to save time and return the schoolbook.
            # Or try one more lower rank?
            # Let's break.
            break
            
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
