"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition
    # Start with schoolbook rank 40, try to reduce
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = 1
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Use a greedy/iterative approach to find low rank
    # First, let's try rank 20 using the direct sum structure
    
    # The tensor T[i,j,k] for i in 0..1, j in 0..3, k in 0..4
    # Split j into j1 in 0..1 and j2 in 2..3
    # T = T1 + T2 where T1[i,j1,k] = 1 for j1 in 0..1, T2[i,j2,k] = 1 for j2 in 2..3
    
    # For each sub-tensor (2x2x5), try rank 10
    # 2x2x5 tensor: U1 shape (4,10), V1 shape (10,10), W1 shape (10,10)
    
    # Let me try a specific construction for 2x2x5 with rank 10
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Term c for (i=0,k=0): U[0,0]=1,V[0,0]=1,W[0,0]=1 gives A[0,0]*B[0,0]*1
    # This is getting complex. Let me use a numerical search.
    
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Build target tensor
    T = np.ones((n, m, p))
    
    # Try ranks from 20 upward
    for target_rank in range(20, 41):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, T, target_rank, deadline)
        if result is not None:
            return result
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _try_rank(n, m, p, T, rank, deadline):
    """Try to find a rank-`rank` decomposition using iterative methods."""
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Iterative minimization
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        # Update U: for each (i,j), U[i*m+j, c] = T[i,j,k] * V[j*p+k, c] * W[k*n+i, c] summed over k
        # Actually, the update rule for alternating least squares:
        # Fix V,W, update U: U[i*m+j, :] = sum_k T[i,j,k] * V[j*p+k, :] * W[k*n+i, :]
        # But T[i,j,k]=1 for all, so:
        for i in range(n):
            for j in range(m):
                new_row = np.zeros(rank)
                for k in range(p):
                    new_row += V[j * p + k, :] * W[k * n + i, :]
                U[i * m + j, :] = new_row
        
        # Fix U,W, update V:
        for j in range(m):
            for k in range(p):
                new_row = np.zeros(rank)
                for i in range(n):
                    new_row += U[i * m + j, :] * W[k * n + i, :]
                V[j * p + k, :] = new_row
        
        # Fix U,V, update W:
        for k in range(p):
            for i in range(n):
                new_row = np.zeros(rank)
                for j in range(m):
                    new_row += U[i * m + j, :] * V[j * p + k, :]
                W[k * n + i, :] = new_row
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-8:
        # Try to snap to rational/half-integer values
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        # Verify
        err_snap = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                    err_snap += (val - 1.0) ** 2
        if err_snap < 1e-6:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            # Use the floating point values (they might be close enough)
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None
# EVOLVE-BLOCK-END
