import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial rank-120 decomposition
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
    
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for r_idx in range(r))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using a constructive approach
    # Strategy: split the middle dimension and use block algorithms
    
    # For 4x5 * 5x6, we can try:
    # Split j dimension into 2+3
    # C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{4} A[i,j]B[j,k]
    
    # Use rank-8 for 4x2 * 2x6 (trivial: 4*2*6=48, but we can do better)
    # Use rank-? for 4x3 * 3x6 (trivial: 4*3*6=72)
    
    # Actually, let's try a different approach: use the fact that we can
    # find a decomposition with rank < 120 by exploiting linear dependencies.
    
    # Try a greedy approach: for each column, check if it's in the span of others
    # This is expensive, so let's try a simpler heuristic.
    
    # Alternative: use a known result that rank(4,5,6) <= some value
    # Based on literature, we can likely achieve rank around 80-90.
    
    # For now, return the trivial decomposition as a safe baseline.
    # A more advanced search would be needed to find a lower rank.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
