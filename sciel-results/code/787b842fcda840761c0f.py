"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    R = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    diff = np.max(np.abs(R - T))
    return diff < tol

def _build_strassen_decomp(n, m, p):
    """
    Build a Strassen-based decomposition for n x m * m x p matrix multiplication.
    For 2x5x6: partition into 2x2 blocks of (5x5, 5x6, 6x2).
    Use Strassen for 2x2 block multiplication where possible.
    """
    # For 2x5x6, we have:
    # A is 2x5, B is 5x6, C is 2x6
    # We can partition A as [A11 A12; A21 A22] where A11, A21 are 1x5, A12, A22 are 1x5
    # But Strassen needs square blocks. Let's use a different approach.
    
    # Actually, for 2x5x6, the most straightforward Strassen-like approach:
    # Treat the 2x5 matrix as having 2 rows, and use the 5x5 and 5x6 structure.
    
    # Let's use the known result: rank(2x5x6) can be achieved by combining
    # Strassen for the 2x2 part with direct multiplication for the rest.
    
    # A simpler approach: use rank 54 from applying Strassen to 2x2x2 sub-blocks
    # For 2x5x6, we can think of it as 2x(5x6) = 2x(5x5 + 5x1)
    # Or partition differently.
    
    # Let's construct a rank 54 solution using the following strategy:
    # Split the 5x5 and 5x6 into 2x2, 2x2, 2x1, 1x2, 1x2, 1x1 blocks
    # This is complex. Let's use a direct construction instead.
    
    # For now, let's return a schoolbook solution with rank 60
    # and later optimize if needed.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try to find a low-rank decomposition using numerical optimization.
    # We'll use a simple gradient descent on the rank-30 hypothesis.
    # If it fails, we fall back to a known higher rank.
    
    # Target rank: 30 (lower bound for 2x5x6)
    target_rank = 30
    
    # Initialize U, V, W with random values
    np.random.seed(42)
    
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U_shape = (n * m, target_rank)
    V_shape = (m * p, target_rank)
    W_shape = (p * n, target_rank)
    
    # We'll try a few different initializations and steps.
    best_solution = None
    best_error = float('inf')
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Gradient descent loop
    num_restarts = 50
    max_iters = 200
    lr = 0.01
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(*U_shape) * 0.1
        V = np.random.randn(*V_shape) * 0.1
        W = np.random.randn(*W_shape) * 0.1
        
        error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute R
            R = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for r in range(target_rank):
                R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.sum((R - T) ** 2)
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-6:
                break
            
            # Compute gradients using stochastic sampling
            num_samples = 50
            indices_a = np.random.randint(0, n * m, num_samples)
            indices_b = np.random.randint(0, m * p, num_samples)
            indices_c = np.random.randint(0, p * n, num_samples)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for s in range(num_samples):
                a, b, c = indices_a[s], indices_b[s], indices_c[s]
                err = R[a, b, c] - T[a, b, c]
                for r in range(target_rank):
                    grad_U[a, r] += err * V[b, r] * W[c, r]
                    grad_V[b, r] += err * U[a, r] * W[c, r]
                    grad_W[c, r] += err * U[a, r] * V[b, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent divergence
            norm_U = np.linalg.norm(U)
            norm_V = np.linalg.norm(V)
            norm_W = np.linalg.norm(W)
            if norm_U > 0 and norm_V > 0 and norm_W > 0:
                scale = 1.0 / (norm_U * norm_V * norm_W) ** (1.0 / 3.0)
                U *= scale
                V *= scale
                W *= scale
        
        # After local search, check if we have a solution
        if error < 1e-4:
            if _verify(U, V, W, n, m, p, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no low-rank solution found, return schoolbook rank 60
    U, V, W = _build_strassen_decomp(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
