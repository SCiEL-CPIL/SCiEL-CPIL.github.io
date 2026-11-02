"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Use einsum for speed
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # W3: (p, n, rank)
    # Result: (n, m, p)
    # Sum over rank
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Note: W3 is indexed by k, i. We need W3[k, i, c].
    # So W3_trans = W3.transpose(1, 0, 2) -> (n, p, rank)? No.
    # W3 is (p, n, rank). We need (k, i, c) -> W3[k, i, c].
    # So W3 is already (k, i, c) if we view first axis as k, second as i.
    # Let's check indices:
    # W is (p*n, rank). W[k*n+i, c].
    # W3 = W.reshape(p, n, rank). So W3[k, i, c].
    
    # We want sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Let's use einsum: 'ijc,jkc,kic->ijk'
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    except:
        return False
        
    return np.allclose(result, 1.0, atol=1e-9)

def _try_round(U, V, W, n, m, p):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_decompose_fast(n, m, p, rank, max_iters=200, tol=1e-8, seed=42):
    """Fast vectorized ALS for tensor decomposition."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Update U
        # A[i,j,k,c] = V3[j,k,c] * W3[k,i,c]
        # We need to solve for each (i,j): sum_c U3[i,j,c] * A[i,j,k,c] = T[i,j,k]
        # A is (n, m, p, rank)
        # A[i,j] is (p, rank)
        # T[i,j] is (p,)
        
        # Compute A efficiently
        # V3: (m, p, rank)
        # W3: (p, n, rank)
        # We want A[i,j,k,c] = V3[j,k,c] * W3[k,i,c]
        # Note: W3[k,i,c] requires accessing W3 with k, i.
        # W3 is (p, n, rank). So W3[k, i, c] is just W3[k, i, c].
        
        # Broadcast V3: (1, m, p, rank)
        # Broadcast W3: (p, n, 1, rank) -> need to align k with p axis of V3?
        # V3 axis 1 is m (j), axis 2 is p (k).
        # W3 axis 0 is p (k), axis 1 is n (i).
        
        # Let's reshape for einsum
        # A[i,j,k,c] = V3[j,k,c] * W3[k,i,c]
        # V3: (m, p, rank) -> (1, m, p, rank)
        # W3: (p, n, rank) -> (p, n, 1, rank)
        # We need to sum? No, element-wise product.
        # But dimensions:
        # V3: j, k, c
        # W3: k, i, c
        # Result: i, j, k, c
        
        V3_b = V3[np.newaxis, :, :, :] # (1, m, p, rank)
        W3_b = W3[:, :, np.newaxis, :] # (p, n, 1, rank) -> (k, i, 1, c)
        
        # We need to align k.
        # V3_b has k at axis 2.
        # W3_b has k at axis 0.
        # We want A[i, j, k, c].
        # This is not a simple broadcast multiplication because k is in different positions.
        # We can use einsum to compute A directly if needed, but it's large.
        # A size: n*m*p*rank = 2*4*7*50 = 2800. Small.
        
        # Let's just compute A explicitly for simplicity and speed (it's small)
        # A[i,j,k,c]
        A = np.zeros((n, m, p, rank), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                # V3[j, :, :] is (p, rank)
                # W3[:, i, :] is (p, rank)
                A[i, j, :, :] = V3[j, :, :] * W3[:, i, :]
        
        # Solve for U3[i,j,:]
        for i in range(n):
            for j in range(m):
                try:
                    U3[i, j, :] = np.linalg.lstsq(A[i, j, :, :].T, T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V
        # B[j,k,i,c] = U3[i,j,c] * W3[k,i,c]
        # Solve for V3[j,k,:]: sum_c V3[j,k,c] * B[j,k,i,c] = T[i,j,k]
        # B is (m, p, n, rank)
        B = np.zeros((m, p, n, rank), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                # U3[:, j, :] is (n, rank)
                # W3[k, :, :] is (n, rank)
                B[j, k, :, :] = U3[:, j, :] * W3[k, :, :]
        
        for j in range(m):
            for k in range(p):
                try:
                    V3[j, k, :] = np.linalg.lstsq(B[j, k, :, :].T, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W
        # C[k,i,j,c] = U3[i,j,c] * V3[j,k,c]
        # Solve for W3[k,i,:]: sum_c W3[k,i,c] * C[k,i,j,c] = T[i,j,k]
        # C is (p, n, m, rank)
        C = np.zeros((p, n, m, rank), dtype=np.float64)
        for k in range(p):
            for i in range(n):
                # U3[i, :, :] is (m, rank)
                # V3[:, k, :] is (m, rank)
                C[k, i, :, :] = U3[i, :, :] * V3[:, k, :]
        
        for k in range(p):
            for i in range(n):
                try:
                    W3[k, i, :] = np.linalg.lstsq(C[k, i, :, :].T, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        # Result = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        res = np.linalg.norm(recon - T)
        if res < tol:
            break
    
    return U3.reshape(n*m, rank), V3.reshape(m*p, rank), W3.reshape(p*n, rank)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 28
    # Try from 55 down to 40
    for rank in range(rank_school - 1, 35, -1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            seed = rank * 100 + trial
            U, V, W = _als_decompose_fast(n, m, p, rank, max_iters=100, tol=1e-8, seed=seed)
            
            # Try to round
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            rounded = _try_round(U_list, V_list, W_list, n, m, p)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
