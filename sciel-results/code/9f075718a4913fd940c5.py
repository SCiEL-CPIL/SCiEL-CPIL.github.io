"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    M = np.einsum('ijc,jkc->ikc', U_reshaped, V_reshaped)
    C_prime = np.einsum('ikc,kic->ik', M, W_reshaped)
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    return np.allclose(C_prime, C_expected, atol=1e-8)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Schoolbook rank 54
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower-rank decompositions
    rng = np.random.RandomState(123)
    
    # Try various target ranks
    for target_rank in range(50, 20, -2):
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U_try = rng.randn(n * m, target_rank)
            V_try = rng.randn(m * p, target_rank)
            W_try = rng.randn(p * n, target_rank)
            
            if _verify_decomposition_fast(U_try, V_try, W_try, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
