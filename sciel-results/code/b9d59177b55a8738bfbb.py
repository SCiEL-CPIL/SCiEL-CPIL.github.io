"""Rewritten program: tensor decomposition for 3x5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
TOTAL = n * m * p  # 105

# Build the full tensor T of shape (n, m, p) where T[i, j, k] = 1
# In the standard matrix multiplication tensor, the decomposition is:
# T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

def build_tensor():
    """Build the 3D tensor of shape (n, m, p) with all ones."""
    return np.ones((n, m, p), dtype=np.float64)

def verify(U, V, W):
    """Verify that U, V, W correctly decompose the tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check that for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        # Reshape U[:, r] to (n, m), V[:, r] to (m, p), W[:, r] to (p, n)
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # The contribution to T[i,j,k] is Ur[i,j] * Vr[j,k] * Wr[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-9)

def schoolbook():
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = TOTAL
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

def try_low_rank_random(max_rank, num_trials=5000, seed=42):
    """Try to find a decomposition with rank <= max_rank using random search."""
    rng = np.random.RandomState(seed)
    for trial in range(num_trials):
        rank = max_rank
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        # Try to verify
        if verify(U, V, W):
            return U, V, W, rank
    return None

def try_structured_low_rank():
    """Try structured approaches to reduce rank below schoolbook."""
    # For 3x5x7, known lower bounds suggest rank is around 100-105.
    # Let's try rank values from 90 up to 104.
    for target_rank in range(90, 105):
        # Try a few random initializations
        for seed in range(20):
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            if verify(U, V, W):
                return U, V, W, target_rank
    return None

def construct():
    """Main entry point. Returns dict with U, V, W matrices."""
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline, 10s safety margin
    
    # Try to find a lower-rank decomposition
    result = try_structured_low_rank()
    if result is not None and time.time() < deadline:
        U, V, W, rank = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no low-rank found, try a few more random attempts
    if time.time() < deadline:
        for target_rank in range(100, 105):
            if time.time() > deadline:
                break
            res = try_low_rank_random(target_rank, num_trials=1000, seed=12345)
            if res is not None:
                U, V, W, rank = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
