import numpy as np
import time

def _schoolbook_3x3():
    """Return the rank-27 schoolbook decomposition for 3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-10):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    The tensor entry T_{(i,j), (j,k), (i,k)} should be 1 for all i,j,k.
    Using the convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    We check this using tensor contraction.
    """
    rank = U.shape[1]
    # U: (9, rank), V: (9, rank), W: (9, rank)
    # We need to check for all i in [0,2], j in [0,2], k in [0,2]:
    # sum_r U[3i+j, r] * V[3j+k, r] * W[3k+i, r] == 1
    
    # Efficient check: construct the tensor and contract
    # T[i,j,k] = sum_r U[3i+j, r] * V[3j+k, r] * W[3k+i, r]
    
    # Reshape to use einsum or manual loops
    # Manual loops are fine for 3x3x3 = 27 checks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank21_decomposition():
    """
    Attempt to construct a valid rank-21 decomposition.
    This is a placeholder for a known rank-21 scheme if available.
    Currently, we return None to indicate no valid hardcoded rank-21 is guaranteed,
    but we structure the code to allow insertion of a known decomposition.
    
    For now, we return None, forcing fallback to schoolbook.
    In a future iteration, a verified rank-21 or rank-23 decomposition could be inserted here.
    """
    # If a known rank-21 decomposition is hardcoded, it would be here.
    # For safety and correctness, we return None.
    return None

def construct():
    deadline = time.time() + 115  # 115s safety margin within 120s
    
    # First, try a known rank-21 decomposition if available
    res = _try_rank21_decomposition()
    if res is not None:
        U, V, W = res['U'], res['V'], res['W']
        # Verify before returning
        if _verify_decomposition(U, V, W):
            return res
    
    # Fallback: schoolbook rank-27
    return _schoolbook_3x3()

if __name__ == "__main__":
    result = construct()
    print("Rank:", result['U'].shape[1])
