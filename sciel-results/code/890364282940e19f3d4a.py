"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Vectorized verification for speed
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a contraction.
    # Let's do it efficiently.
    # For each r, U[:, r] is a vector of length n*m.
    # V[:, r] is a vector of length m*p.
    # W[:, r] is a vector of length p*n.
    
    # We can compute the tensor T = sum_r outer(U[:, r], V[:, r], W[:, r])
    # But that's huge.
    # Instead, check the identity:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # We can loop over r and accumulate.
    # Or use einsum.
    # np.einsum('ir, jr, kr -> ijk', U, V, W) ? No, indices are mixed.
    
    # Let's stick to a somewhat vectorized loop over i, j, k or r.
    # Given small dimensions (4,5,5), loops are fine.
    
    for i in range(n):
        for j in range(m):
            # U[i*m+j, :] is a vector u of length rank
            u = U[i*m+j, :]
            for k in range(p):
                # V[j*p+k, :] is v
                # W[k*n+i, :] is w
                val = np.dot(u, V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest 0.5."""
    return np.round(x * 2) / 2

def _try_rank(r, n, m, p, max_iterations=2000, seed=42, time_limit=10):
    """
    Try to find a valid decomposition of rank r using ALS.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b_vec = np.ones(p)
                try:
                    u_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    U_new[i*m+j, :] = u_sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b_vec = np.ones(n)
                try:
                    v_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    V_new[j*p+k, :] = v_sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b_vec = np.ones(m)
                try:
                    w_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    W_new[k*n+i, :] = w_sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                u = U[i*m+j, :]
                for k in range(p):
                    val = np.dot(u, V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            return U, V, W
            
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try ranks in increasing order
    # We only need to try a few ranks below schoolbook if we are lucky.
    # Start from 100 down to 90? Or 2 up to 100?
    # The prompt says "rank-minimal".
    # If we find rank 99, we should stop? No, we want minimal.
    # But finding minimal is hard.
    # Let's try to find *any* rank < 100.
    
    # Given the time limit, we can't search all ranks.
    # Let's try rank 99, 98, ... with a few restarts.
    
    for target_rank in range(rank_schoolbook - 1, 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
                
            U_r, V_r, W_r = _try_rank(target_rank, n, m, p, max_iterations=1000, seed=seed, time_limit=5)
            
            # Round to half-integers
            U_r = _round_to_half(U_r)
            V_r = _round_to_half(V_r)
            W_r = _round_to_half(W_r)
            
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break
        
        if found:
            # We found a lower rank. Can we go lower?
            # Given time constraints, we might stop here or continue.
            # Let's continue to see if we can go lower, but with less time per rank.
            pass
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
