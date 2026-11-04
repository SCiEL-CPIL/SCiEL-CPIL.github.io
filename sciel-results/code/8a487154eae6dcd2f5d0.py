"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Build tensors
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For all i,j,k, T[i,j,k] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_valid_entry(x):
    """Check if x is an integer or half-integer."""
    x2 = 2 * x
    return abs(x2 - round(x2)) < 1e-9

def _snap_to_half(x):
    """Snap x to nearest half-integer."""
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.all([_is_valid_entry(x) for x in arr.flatten()]):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Sum over r: U3[i,j,:] * V3[j,k,:] * W3[k,i,:]
    # This is a 4D contraction
    # T[i,j,k] = einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random decomposition and try to optimize."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with small random values
    U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
    V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
    W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
    
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=2000, seed=None):
    """Try to find a valid decomposition of given rank using iterative method."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
    V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
    W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    lr = 0.01
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        
        # Gradients
        # dT/dU3[i,j,r] = V3[j,k,r] * W3[k,i,r] summed over k
        # grad_U[i,j,r] = sum_k err[i,j,k] * V3[j,k,r] * W3[k,i,r]
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        # Update
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        # Check convergence
        if np.max(np.abs(err)) < 1e-10:
            break
        
        # Decay learning rate
        if it % 200 == 0:
            lr *= 0.5
    
    Ua = U3.reshape(n*m, rank)
    Va = V3.reshape(m*p, rank)
    Wa = W3.reshape(p*n, rank)
    
    # Snap to half-integers
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    # Also try without snapping (in case the optimization already produced valid entries)
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Try various ranks from low to high
    # Known: rank must be at least max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # But actually the lower bound for matrix mult tensor rank is more subtle
    # For 3x5x6, let's try ranks from 30 upward
    
    min_rank = max(n*m, m*p, n*p)
    
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 10.0:
            break
        
        found = False
        for attempt in range(10):
            if _remaining() < 5.0:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=500, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition, can stop
            break
    
    # Final check: ensure the best result is valid
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if something went wrong
        best = schoolbook
    
    return best
