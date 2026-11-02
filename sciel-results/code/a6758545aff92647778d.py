import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n, m, p) where T[i,j,k]=1.
    """
    return np.ones((n, m, p), dtype=np.float64)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the tensor T[i,j,k]=1.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    R = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape to 3D for easier indexing in loops if needed, or just use 2D
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    # We can do this efficiently with einsum if we reshape, but for small n,m,p loops are fine and transparent.
    # Let's use einsum for speed:
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    
    # Result[i, j, k] = sum_r U3[i, j, r] V3[j, k, r] W3[k, i, r]
    try:
        # einsum: 'ijr,jkr,kir->ijk'
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        if not np.allclose(result, 1.0, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _als_decompose_explicit(n, m, p, rank, max_iter=500, tol=1e-8, seed=42, time_limit=None, start_time=None):
    """
    Explicit ALS for tensor decomposition.
    T[i,j,k] = 1 for all i,j,k.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Normalize columns to prevent divergence
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    for it in range(max_iter):
        if time_limit and start_time and (time.time() - start_time > time_limit):
            break
            
        # Update U
        # For each i, j:
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Let A[c, k] = V[j*p+k, c] * W[k*n+i, c] for c in range(rank)
        # Then T[i, j, :] = U[i*m+j, :] @ A.T
        # Solve A @ U[i*m+j, :].T = T[i, j, :].T
        for i in range(n):
            for j in range(m):
                A = np.zeros((rank, p), dtype=np.float64)
                for k in range(p):
                    A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    # Solve A x = b where x is U[i*m+j, :].T
                    # A is rank x p, b is p x 1
                    x, _, _, _ = np.linalg.lstsq(A, T[i, j, :].T, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        # For each j, k:
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Let B[c, i] = U[i*m+j, c] * W[k*n+i, c]
        # T[:, j, k] = V[j*p+k, :] @ B.T
        for j in range(m):
            for k in range(p):
                B = np.zeros((rank, n), dtype=np.float64)
                for i in range(n):
                    B[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(B, T[:, j, k].T, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        # For each k, i:
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Let C[c, j] = U[i*m+j, c] * V[j*p+k, c]
        # T[i, :, k] = W[k*n+i, :] @ C.T
        for k in range(p):
            for i in range(n):
                C = np.zeros((rank, m), dtype=np.float64)
                for j in range(m):
                    C[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(C, T[i, :, k].T, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 5 == 0:
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
            err = np.linalg.norm(recon - T)
            if err < tol:
                break
                
    # Final check
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    err = np.linalg.norm(recon - T)
    
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit
    
    # 1. Schoolbook Fallback (Rank 56)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # 2. Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 28
    # We try ranks from 55 down to 28
    # To save time, we don't try all 28 restarts for every rank if we are running out of time.
    
    # We will prioritize higher ranks (closer to 56) as they are more likely to be solvable.
    # If we find a solution, we break and try lower? No, we want minimal rank.
    # So we should try from low to high? Or high to low?
    # If we try low to high, we might waste time on unsolvable low ranks.
    # If we try high to low, we might find 55 quickly, then 54, etc.
    # Given the difficulty, let's try a few restarts for each rank from 55 down to 28.
    
    # Strategy:
    # For R in range(55, 27, -1):
    #   For restart in range(5): # Fewer restarts to allow more ranks to be tried
    #     Run ALS
    #     If success, round, verify.
    #     If verified, update best and break out of restart loop.
    #   If found, continue to R-1.
    #   If time runs out, break.
    
    # Actually, finding rank 55 is harder than 56? No, 56 is trivial.
    # Finding rank 55 is a specific constraint satisfaction problem.
    
    # Let's try ranks from 55 down to 40 first, with 3 restarts each.
    # Then if time permits, go lower.
    
    for rank in range(schoolbook_rank - 1, 27, -1):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        found = False
        # Number of restarts decreases as we go lower? Or fixed small number.
        # Let's use 3 restarts to be safe on time.
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            seed = rank * 100 + restart
            res = _als_decompose_explicit(n, m, p, rank, max_iter=200, tol=1e-8, seed=seed, time_limit=time_limit - (time.time() - start_time), start_time=start_time)
            
            if res is not None:
                U, V, W = res
                # Verify real solution
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    # Try to round to half-integers
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    
                    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        # Found a valid half-integer solution
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = rank
                        found = True
                        break # Move to next lower rank
        
        # If we found a solution for this rank, we continue to lower rank.
        # If not, we also continue, as a lower rank might be easier (unlikely) or we just run out of time.
        # Actually, if we fail at 55, 54 is likely also hard. 
        # But we must search.
        
    return best_result
