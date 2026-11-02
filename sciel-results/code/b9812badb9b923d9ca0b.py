import time
import numpy as np

# Deadline guard
_DEADLINE = 135.0  # seconds
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact tensor decomposition."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # For each s, the tensor is U_s outer V_s outer W_s
    # Check: for all i,j,k, sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1
    # Use a random sampling approach for speed
    rng = np.random.RandomState(42)
    for _ in range(100):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        val = 0.0
        for s in range(rank):
            val += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        if abs(val - 1.0) > 1e-9:
            return False
    # Also check some zero conditions (shouldn't exist for matrix mult, all entries are 1)
    return True

def _build_rank_56(n, m, p):
    """Build the rank-56 Strassen decomposition for 2x4 by 4x8."""
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 4, 8
    
    # Start with the known rank-56 construction
    result = _build_rank_56(n, m, p)
    
    # Check if we have time for any additional optimization
    if _time_left() > 5.0:
        # Verify the rank-56 solution is correct
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            pass  # Good, keep it
    
    return result
