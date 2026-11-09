import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 115.0

def _elapsed():
    return time.time() - START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _build_schoolbook():
    """Return (U, V, W) as numpy arrays for the schoolbook decomposition, rank 45."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _target_tensor():
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n) as a numpy array."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U, V, W are numpy arrays of half-integers (floats).
    """
    n, m, p = DIMENSIONS
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_target = _target_tensor()
    
    if not np.allclose(T_recon, T_target, atol=tol, rtol=0):
        return False
    
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for val in mat.flat:
            if abs(2 * val - round(2 * val)) > tol:
                return False
    return True

def _round_to_half_integer(arr, tol=1e-7):
    """Round each entry to the nearest half-integer if within tolerance."""
    arr_rounded = np.round(arr * 2) / 2.0
    if not np.allclose(arr, arr_rounded, atol=tol, rtol=0):
        return None
    return arr_rounded

def _solve_for_factors(T, indices, r):
    """
    Given the target tensor T and a set of 'indices' (which define the structure of the decomposition),
    solve for the optimal factors U, V, W.
    Here, we assume the decomposition is of the form:
    T = sum_{s=1}^r u_s \otimes v_s \otimes w_s
    
    We use a simplified approach:
    We fix the structure to be similar to schoolbook but with some columns removed/modified.
    Actually, for a general rank-r tensor, the factors are not unique.
    We use the "CP decomposition" via alternating least squares (ALS) initialized with schoolbook subsets.
    """
    I, J, K = T.shape
    # Initialize with random or schoolbook-like
    # We'll use a random initialization seeded by the indices to ensure determinism
    seed = hash(tuple(indices)) % 10000
    np.random.seed(seed)
    
    U = np.random.randn(I, r)
    V = np.random.randn(J, r)
    W = np.random.randn(K, r)
    
    # ALS Iterations
    for _ in range(50):
        # Update W
        UV = np.einsum('ir,jr->ijr', U, V)
        UV_flat = UV.reshape(I * J, r)
        T2 = T.reshape(I * J, K)
        try:
            W, _, _, _ = np.linalg.lstsq(UV_flat, T2, rcond=None)
            W = W.T
        except:
            return None
            
        # Update V
        UW = np.einsum('ir,kr->ikr', U, W)
        UW_flat = UW.reshape(I * K, r)
        T3 = T.reshape(I * K, J)
        try:
            V, _, _, _ = np.linalg.lstsq(UW_flat, T3, rcond=None)
            V = V.T
        except:
            return None
            
        # Update U
        VW = np.einsum('jr,kr->jkr', V, W)
        VW_flat = VW.reshape(J * K, r)
        T4 = T.reshape(J * K, I)
        try:
            U, _, _, _ = np.linalg.lstsq(VW_flat, T4, rcond=None)
            U = U.T
        except:
            return None
            
    return U, V, W

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    T = _target_tensor()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = len(U_sb[0])
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try to reduce rank from 44 down to 30
    for r in range(44, 29, -1):
        if _time_remaining() < 2.0:
            break
            
        # Strategy: Try multiple random subsets of the schoolbook structure
        # Or just run ALS with different seeds
        found = False
        for attempt in range(5):
            if _time_remaining() < 1.0:
                break
                
            # Initialize factors using schoolbook columns
            # We take the first r columns of schoolbook as a starting point
            # But since schoolbook columns are orthogonal, taking a subset might not be a good start for a general rank-r tensor.
            # Better: Random initialization
            
            U, V, W = _solve_for_factors(T, range(r), r)
            
            if U is None:
                continue
                
            # Snap to half-integers
            U_r = _round_to_half_integer(U)
            V_r = _round_to_half_integer(V)
            W_r = _round_to_half_integer(W)
            
            if U_r is not None and V_r is not None and W_r is not None:
                if _verify_exact(U_r, V_r, W_r):
                    best_rank = r
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    found = True
                    break # Found a valid decomposition for this rank, try lower?
                    # Actually, we want the MINIMAL rank. So we should break the inner loop and continue to r-1.
                    
        # If we found a valid rank r, we can break the outer loop? 
        # No, we want the lowest rank. So we continue to r-1.
        # But if we found rank 44, we should check 43.
        
    return best_result
