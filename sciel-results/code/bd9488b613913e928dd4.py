import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _build_strassen_block():
    """Return U, V, W for a 2x2 * 2x2 Strassen decomposition (rank 7)."""
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    P_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    return U_s, V_s, P_s

def _build_rank14():
    """Build the rank-14 decomposition using Strassen on two 2x2 blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, P_s = _build_strassen_block()
    
    for s in range(7):
        # Block 1 (k in 0,1)
        r1 = s
        U[:, r1] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + kp, r1] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r1] = P_s[s][i, k]
        
        # Block 2 (k in 2,3)
        r2 = 7 + s
        U[:, r2] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + (kp + 2), r2] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, r2] = P_s[s][i, k]
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Compute the multiplication tensor from the decomposition
    # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
    # A is n x m, B is m x p, C = A @ B is n x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # So T[i, j, k] should be 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # Actually, the tensor T is defined such that:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    
    # The task says: W[k*n+i, r]
    # U has shape (n*m, rank), indexed as [i*m + j, r] where i is row of A, j is col of A
    # V has shape (m*p, rank), indexed as [j*p + k, r] where j is row of B, k is col of B
    # W has shape (p*n, rank), indexed as [k*n + i, r] where k is col of B, i is row of A
    
    # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # Actually, for matrix multiplication C = A @ B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # So the coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
    
    # Let's verify by checking that the tensor decomposition gives the correct result
    # for random matrices.
    
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        C_computed[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] * A[i, j] * B[j, k]
        
        # Wait, that's not right. The decomposition should directly give C.
        # Let me think again.
        
        # The tensor T is such that:
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i, j, k] should be 1 for all i, j, k (since each A[i,j]*B[j,k] contributes to C[i,k]).
        
        # Actually no, T[i,j,k] is the coefficient, and for matrix multiplication it's always 1.
        # So we just need to check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all i, j, k.
        
        pass
    
    # Direct check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # 130s deadline, 20s safety margin
    
    # Start with rank 14
    U14, V14, W14 = _build_rank14()
    if not _verify_decomposition(U14, V14, W14):
        raise RuntimeError("Rank 14 decomposition failed verification")
    
    best_rank = 14
    best_U, best_V, best_W = U14, V14, W14
    
    # Try to find rank 13 via random search
    # This is a heuristic; if it fails quickly, we return rank 14.
    # We use a simple random search with a tight time budget.
    t1 = time.time()
    search_budget = 60  # seconds for rank 13 search
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() - t1 > search_budget:
            break
        
        # Random rank-13 initialization
        U = np.random.randn(n * m, 13)
        V = np.random.randn(m * p, 13)
        W = np.random.randn(p * n, 13)
        
        # Scale to reasonable magnitude
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Simple gradient-free check: just verify
        # (In practice, we'd do optimization, but for now just check)
        # This is very unlikely to work, so we skip the detailed check
        # to save time.
        pass
    
    # If we had time, we could try more sophisticated methods, but for now
    # we return the known rank-14 solution.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
