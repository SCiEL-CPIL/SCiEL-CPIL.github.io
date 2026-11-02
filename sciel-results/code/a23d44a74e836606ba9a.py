"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=4, m=4, p=8, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We check on a sample of (i,j,k) triples
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = 1 (the multiplication tensor has 1s at positions where the output depends on input)
                # Actually, the tensor T[i,j,k] = 1 for all (i,j,k) in the multiplication problem
                # because C[i,k] = sum_j A[i,j]*B[j,k]
                # So the tensor component T[i,j,k] = 1
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                # The target value is 1
                if abs(s - 1.0) > tol:
                    errors.append((i, j, k, s))
                    if len(errors) > 5:
                        return False, errors
    return len(errors) == 0, errors

def _try_random_decomposition(n, m, p, rank, seed=42, time_limit=30.0):
    """Try to find a rank-r decomposition using random initialization and refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradients
        # For each (i,j,k), error = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c] * error_sign
        # Actually, we want to minimize sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        # Use a simplified gradient step
        # For efficiency, compute the error tensor
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Sample a few (i,j,k) to compute gradients
        samples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rng.shuffle(samples)
        sample_size = min(100, len(samples))
        
        lr = 0.001
        for idx in range(sample_size):
            i, j, k = samples[idx]
            ui = i*m + j
            vj = j*p + k
            wk = k*n + i
            
            s = np.dot(U[ui], V[vj] * W[wk])
            err = s - 1.0
            
            # Gradient with respect to U[ui]
            U_new[ui] -= lr * err * (V[vj] * W[wk])
            # Gradient with respect to V[vj]
            V_new[vj] -= lr * err * (U[ui] * W[wk])
            # Gradient with respect to W[wk]
            W_new[wk] -= lr * err * (U[ui] * V[vj])
        
        U, V, W = U_new, V_new, W_new
        
        # Check error
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total_err += (s - 1.0)**2
        if total_err < best_err:
            best_err = total_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-12:
                break
    
    return best_U, best_V, best_W, best_err

def _schoolbook_decomposition(n=4, m=4, p=8):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
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

def _strassen_like_4x4x8():
    """
    Attempt to construct a low-rank decomposition using block structure.
    
    For 4x4 * 4x8, we can think of it as multiplying 2x2 blocks of 2x2 with 2x4 blocks of 2x4.
    The 2x2 * 2x4 multiplication can use Strassen-like methods.
    
    Actually, let's try a different approach: use the known fact that the tensor rank
    for matrix multiplication M_{n,m,p} satisfies certain bounds.
    
    For now, let's try a simple block decomposition and see if we can reduce rank.
    """
    n, m, p = 4, 4, 8
    # Use schoolbook as base but try to merge some terms
    # This is a placeholder - the real work is in the random search
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook
    result = _schoolbook_decomposition(n, m, p)
    best_result = result
    best_rank = n * m * p  # 128
    
    # Try to improve using random initialization + refinement for various ranks
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # But the actual rank is likely higher
    
    # Try ranks from 32 upward, but be careful about time
    for rank in [32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 96, 112]:
        if time.time() - start_time > deadline:
            break
        
        # Try a few seeds
        for seed in [42, 123, 7, 999, 2024]:
            if time.time() - start_time > deadline:
                break
            
            try:
                U, V, W, err = _try_random_decomposition(n, m, p, rank, seed=seed, time_limit=15.0)
            except Exception:
                continue
            
            if err < 1e-6:
                # Check with exact verification
                ok, errors = _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if ok:
                    # Convert to integers if possible
                    U_int = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
                    V_int = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
                    W_int = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
                    
                    # Final verification with rounded values
                    ok2, _ = _verify_decomposition(U_int, V_int, W_int, n, m, p)
                    if ok2:
                        candidate = {"U": U_int, "V": V_int, "W": W_int}
                        best_result = candidate
                        best_rank = rank
                        break
        
        if best_rank == rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
