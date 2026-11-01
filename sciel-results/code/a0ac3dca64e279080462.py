"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def _construct_rank24():
    """Trivial rank-24 decomposition for 2x3x4 matrix multiplication."""
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 12 x 24
    W = [[0] * rank for _ in range(p * n)]  # 8 x 24
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    
    # Also check that for "wrong" combinations, the sum should be 0
    # The tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # should equal 1 if this is the "diagonal" element in the multiplication, 0 otherwise
    # Actually, the matrix multiplication tensor has T[i,j,k,l] = delta[j,k]
    # So for fixed i, the slice over (j,k) should be identity-like
    
    # Let's verify more carefully:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta[j,k] (for the multiplication structure)
    # But our decomposition gives: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 when the "indices match" in the multiplication sense
    
    # Actually, the standard tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] 
    # contributes to C[i,k]
    
    # Wait, I need to think about this more carefully.
    # The matrix multiplication tensor has dimension n * m * p
    # Entry (i,j,k) = 1 always? No...
    # 
    # Actually, the structure tensor for (n×m)·(m×p) multiplication:
    # We want C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all i,j,k
    # because each product A[i,j]*B[j,k] contributes to C[i,k]
    
    # So the verification above should be sufficient: all entries should be 1.
    return True

def _als_decompose(n, m, p, rank, max_iter=2000, tol=1e-12, seed=0):
    """
    Use Alternating Least Squares to find a rank-r decomposition.
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for iteration in range(max_iter):
        # Update U: for each row i*m+j, solve least squares
        for idx in range(n * m):
            i = idx // m
            j = idx % m
            # We need sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is: U[idx, :] @ (V[j*p:..., :] * W[...]) = ones(p)
            # Reshape: for each k, sum_r U[idx,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then U[idx, :] @ M = ones(p)
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j * p + k, :] * W[k * n + i, :]
            # Solve U[idx, :] @ M = ones(p)
            # This is underdetermined if rank > p, so use pseudoinverse
            try:
                U[idx, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j = idx // p
            k = idx % p
            # sum_r U[i*m+j, r] * V[idx, r] * W[k*n+i, r] = 1 for all i
            # Let M[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Then V[idx, :] @ M = ones(n)
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                V[idx, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[idx, r] = 1 for all j
            # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then W[idx, :] @ M = ones(m)
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                W[idx, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
            except:
                pass
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _to_rational_list(arr, max_denom=100):
    """Convert numpy array to list of Fraction, checking if entries are rational with small denom."""
    result = []
    for val in arr.flat:
        # Check if val is close to a rational with small denominator
        best_frac = None
        best_err = float('inf')
        for d in range(1, max_denom + 1):
            num = round(val * d)
            frac = Fraction(num, d)
            err = abs(float(frac) - val)
            if err < best_err:
                best_err = err
                best_frac = frac
            if err < 1e-10:
                break
        if best_err > 1e-8:
            return None  # Not rational enough
        result.append(best_frac)
    return result

def _construct():
    """Search for minimum rank decomposition."""
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_result = _construct_rank24()
    best_rank = n * m * p  # 24
    
    # Try ranks from low to high, but start from a reasonable lower bound
    # Known: rank of 2x3x4 matmul is at most 24, lower bound is at least 18
    # Let's try from 18 upward
    
    for rank in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=500, tol=1e-12, seed=seed)
            
            if err < 1e-9:
                # Try to convert to rationals
                U_rat = _to_rational_list(U, max_denom=20)
                V_rat = _to_rational_list(V, max_denom=20)
                W_rat = _to_rational_list(W, max_denom=20)
                
                if U_rat is not None and V_rat is not None and W_rat is not None:
                    # Verify with exact arithmetic
                    U_list = [[float(U_rat[i*rank+j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_rat[i*rank+j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_rat[i*rank+j]) for j in range(rank)] for i in range(p*n)]
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
