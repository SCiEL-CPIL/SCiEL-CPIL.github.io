"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x4x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k].
    The tensor is T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k], else 0.
    Here T[i,j,k] = 1 for all valid i,j,k since every product contributes."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W with the trace convention W[k*n+i, r] exactly decompose the tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    r = U.shape[1]
    T = _build_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - T[i,j,k]) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = _build_tensor(n, m, p)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Compute via einsum: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # W3[k,i,c] needs to be indexed as [k, i, c], which is W3 reshaped
    # W is (p*n, r) with W[k*n+i, c], so W3[k,i,c] = W[k*n+i, c]
    # Let's compute: for each i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Use broadcasting
    U_flat = U  # (n*m, r)
    V_flat = V  # (m*p, r)
    W_flat = W  # (p*n, r)
    # For each (i,j,k): U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    # Build index arrays
    idx_U = np.array([i*m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_V = np.array([j*p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_W = np.array([k*n + i for i in range(n) for j in range(m) for k in range(p)])
    U_sel = U_flat[idx_U]  # (n*m*p, r)
    V_sel = V_flat[idx_V]  # (n*m*p, r)
    W_sel = W_flat[idx_W]  # (n*m*p, r)
    # Sum over r
    T_flat = np.sum(U_sel * V_sel * W_sel, axis=1)
    T_expected = np.ones(n*m*p, dtype=np.int64)
    return np.allclose(T_flat, T_expected, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _to_list_of_lists(mat):
    return [[float(x) for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find better rank via structured search
    # Strategy: try ranks from lower bound up to schoolbook
    # Known lower bound for 2x4x4: at least ceil(log2(2*4*4)) = ceil(6) = 6, but realistically ~16-20
    
    # Try a few candidate ranks with randomized search
    candidate_ranks = list(range(16, best_rank + 1))
    # Sort by rank ascending
    candidate_ranks.sort()
    
    rng = np.random.default_rng(42)
    
    for r in candidate_ranks:
        if _remaining() < 5.0:
            break
        # Try multiple random initializations
        max_attempts = 200
        found = False
        for attempt in range(max_attempts):
            if _remaining() < 3.0:
                break
            # Random small integer/half-integer entries
            # Scale by 2 to use integers, then divide by 2
            U_rand = rng.integers(-2, 3, size=(n*m, r)).astype(np.float64) / 2.0
            V_rand = rng.integers(-2, 3, size=(m*p, r)).astype(np.float64) / 2.0
            W_rand = rng.integers(-2, 3, size=(p*n, r)).astype(np.float64) / 2.0
            if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = r
                found = True
                break
        
        if found:
            # Try to reduce further
            continue
        else:
            # If we couldn't find at this rank, skip to next
            # But we might have found at a higher rank already
            if r < best_rank:
                continue
            else:
                break
    
    # Convert to lists
    result = {
        "U": _to_list_of_lists(best_U),
        "V": _to_list_of_lists(best_V),
        "W": _to_list_of_lists(best_W),
    }
    return result

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
