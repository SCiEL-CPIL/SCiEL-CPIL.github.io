"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomp(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition using random search with structured initialization
    # The key insight: for 4x4x7, we can try to exploit the structure where
    # each output column k corresponds to a 4x4 matrix-vector multiplication
    
    rng = np.random.default_rng(42)
    
    # Try ranks below schoolbook
    # For 4x4x7, the theoretical lower bound is max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But realistically, we're looking for something between 28 and 112
    
    # Use a smarter random search: initialize with sparse patterns that respect the tensor structure
    for rank in range(best_rank - 1, max(28, best_rank - 40), -1):
        if time.time() > deadline:
            break
        
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Create structured random matrices with sparsity
            # Density decreases as we try lower ranks
            density = max(0.05, 0.2 * (rank / best_rank))
            
            U_rand = np.zeros((n * m, rank))
            V_rand = np.zeros((m * p, rank))
            W_rand = np.zeros((p * n, rank))
            
            # Randomly select entries to be non-zero
            for r in range(rank):
                # For each rank component, select a few entries
                num_nonzero_U = max(1, int(n * m * density))
                num_nonzero_V = max(1, int(m * p * density))
                num_nonzero_W = max(1, int(p * n * density))
                
                U_idx = rng.choice(n * m, size=num_nonzero_U, replace=False)
                V_idx = rng.choice(m * p, size=num_nonzero_V, replace=False)
                W_idx = rng.choice(p * n, size=num_nonzero_W, replace=False)
                
                U_rand[U_idx, r] = rng.choice([-1, 1])
                V_rand[V_idx, r] = rng.choice([-1, 1])
                W_rand[W_idx, r] = rng.choice([-1, 1])
            
            if _verify_decomp(U_rand.tolist(), V_rand.tolist(), n, m, p):
                best_rank = rank
                U, V, W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
