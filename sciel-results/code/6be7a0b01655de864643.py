import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, all other entries 0."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all non-zero entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
                
                # Also check some zero entries to ensure no spurious terms
                # We don't need to check all zeros, but a few random ones helps
                # Actually, if the non-zeros are 1 and the tensor is sparse, 
                # it's likely correct if rank is minimal and structure is tight.
                # But to be safe, we can check that the reconstruction matches T exactly.
                
    # Full check is expensive, so we rely on the non-zero check + low rank.
    # However, for exactness, we should check that the reconstructed tensor is 0 where T is 0.
    # Let's do a full reconstruction check for small dimensions.
    T_rec = np.einsum('ik,jk,lk->ijl', U_arr, V_arr, W_arr)
    T_true = _build_tensor(n, m, p)
    if not np.allclose(T_rec, T_true, atol=tol):
        return False
    return True

def _solve_rank_system(n, m, p, rank, deadline, seed=0):
    """
    Solve the polynomial system for a given rank using least_squares.
    Variables are flattened U, V, W.
    """
    rng = np.random.RandomState(seed)
    
    dim1 = n * m
    dim2 = m * p
    dim3 = p * n
    
    # Total number of variables
    num_vars = rank * (dim1 + dim2 + dim3)
    
    # Flatten U, V, W
    # U: dim1 x rank
    # V: dim2 x rank
    # W: dim3 x rank
    
    def residuals(x):
        if time.time() > deadline - 5.0:
            return None # Signal timeout
        
        U_flat = x[:rank * dim1].reshape(dim1, rank)
        V_flat = x[rank * dim1:rank * (dim1 + dim2)].reshape(dim2, rank)
        W_flat = x[rank * (dim1 + dim2):].reshape(dim3, rank)
        
        # Compute reconstructed tensor
        # T_rec[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        T_rec = np.einsum('ic,jc,kc->ijk', U_flat, V_flat, W_flat)
        
        # Target tensor
        T_true = _build_tensor(n, m, p)
        
        return (T_rec - T_true).flatten()
    
    # Initial guess: Random
    x0 = rng.randn(num_vars) * 0.1
    
    # Try to solve
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=5000)
    except Exception as e:
        # LM might fail, try trf
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=5000)
        except Exception:
            return None
    
    if result.cost < 1e-10:
        # Extract U, V, W
        U_flat = result.x[:rank * dim1].reshape(dim1, rank)
        V_flat = result.x[rank * dim1:rank * (dim1 + dim2)].reshape(dim2, rank)
        W_flat = result.x[rank * (dim1 + dim2):].reshape(dim3, rank)
        
        # Round to nearest rational/integer if close
        # The entries might be small integers or half-integers
        U_rounded = np.round(U_flat, 4)
        V_rounded = np.round(V_flat, 4)
        W_rounded = np.round(W_flat, 4)
        
        # Verify
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            
        # If rounding didn't work, try the unrounded if it's very precise
        if _verify_decomposition(U_flat.tolist(), V_flat.tolist(), W_flat.tolist(), n, m, p, tol=1e-9):
            return {
                "U": U_flat.tolist(),
                "V": V_flat.tolist(),
                "W": W_flat.tolist()
            }
            
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    # For 2x3x6: max(6, 18, 12) = 18.
    # However, Strassen-like methods can go lower.
    # Let's try ranks from 14 to 18 first, then up to 36.
    
    # Theoretical lower bound is actually max(n*m, m*p, p*n) / something? 
    # No, the rank must be at least the maximum of the dimensions of the slices? 
    # Actually, rank >= max(n*m, m*p, p*n) is NOT true. 
    # Rank >= n*m*p is schoolbook.
    # Rank >= max(n*m, m*p, p*n) is a common lower bound for some tensors, 
    # but for matrix multiplication, the rank is at least max(n*m, m*p, p*n) is FALSE.
    # The rank of M_{2,2,2} is 7, while max(4,4,4)=4. So 7 > 4.
    # The rank of M_{n,n,n} is at least n^2 + n.
    # For 2x3x6, let's just search from a reasonable lower bound.
    # Known bounds: R(2,3,3) = 11. 
    # If we split 2x3x6 into two 2x3x3, rank <= 22.
    # Can we do better? 
    # Let's try ranks 14, 15, 16, 17, 18, 19, 20, 21, 22.
    
    for rank in range(14, 23):
        if time.time() > deadline - 10.0:
            break
            
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
                
            result = _solve_rank_system(n, m, p, rank, deadline, seed=seed)
            if result is not None:
                return result
                
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
