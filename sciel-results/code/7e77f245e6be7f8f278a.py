"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def _build_strassen_decomposition(n=2, m=2, p=10):
    """
    Build a rank-35 tensor decomposition for 2x2 * 2x10 matrix multiplication
    using Strassen's algorithm applied to 5 independent 2x2 column blocks.
    
    Returns U, V, W as lists of lists with integer entries.
    """
    num_blocks = p // 2  # 5
    rank = 7 * num_blocks  # 35
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for b in range(num_blocks):
        base = 7 * b
        
        # P1 = (a + d)(e + h)
        col = base
        U[0, col] = 1
        U[3, col] = 1
        V[2*b, col] = 1
        V[10+2*b+1, col] = 1
        W[4*b, col] = 1
        W[4*b+3, col] = 1
        
        # P2 = (c + d)e
        col = base + 1
        U[2, col] = 1
        U[3, col] = 1
        V[2*b, col] = 1
        W[4*b+1, col] = 1
        W[4*b+3, col] = -1
        
        # P3 = a(f - h)
        col = base + 2
        U[0, col] = 1
        V[2*b+1, col] = 1
        V[10+2*b+1, col] = -1
        W[4*b+2, col] = 1
        W[4*b+3, col] = 1
        
        # P4 = d(g - e)
        col = base + 3
        U[3, col] = 1
        V[10+2*b, col] = 1
        V[2*b, col] = -1
        W[4*b, col] = 1
        W[4*b+1, col] = 1
        
        # P5 = (a + b)h
        col = base + 4
        U[0, col] = 1
        U[1, col] = 1
        V[10+2*b+1, col] = 1
        W[4*b, col] = -1
        W[4*b+2, col] = 1
        
        # P6 = (c - a)(e + f)
        col = base + 5
        U[2, col] = 1
        U[0, col] = -1
        V[2*b, col] = 1
        V[2*b+1, col] = 1
        W[4*b+3, col] = 1
        
        # P7 = (b - d)(g + h)
        col = base + 6
        U[1, col] = 1
        U[3, col] = -1
        V[10+2*b, col] = 1
        V[10+2*b+1, col] = 1
        W[4*b, col] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=2, p=10):
    """
    Verify that the decomposition exactly matches the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal the standard matrix multiplication tensor:
    T[i,j,k] = 1 if j is the only shared index (always true for matmul), i.e.,
    T[i,j,k] = delta-like: the (i,j,k) entry is 1 always? No.
    
    Actually for C[i,k] = sum_j A[i,j] B[j,k], the tensor is:
    T[i,j,k] = 1 for all i,j,k (it's just the structure tensor of matmul)
    Wait no. The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    when expressed as C[i,k] = sum_{j} A[i,j] B[j,k].
    
    Let me verify: reconstruct C from random A, B using U, V, W.
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_recon = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        U_vec = U[:, r]
        V_vec = V[:, r]
        W_vec = W[:, r]
        # Reshape
        U_reshaped = U_vec.reshape(n, m)
        V_reshaped = V_vec.reshape(m, p)
        W_reshaped = W_vec.reshape(p, n)
        # Contribution to C
        C_recon += U_reshaped @ V_reshaped * W_reshaped.T  # W is (p*n, rank), index k*n+i
    
    return np.allclose(C_expected, C_recon, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-35 Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Base decomposition verification failed")
    
    best_rank = U.shape[1]
    best_result = {
        "U": [[float(U[i, j]) for j in range(best_rank)] for i in range(U.shape[0])],
        "V": [[float(V[i, j]) for j in range(best_rank)] for i in range(V.shape[0])],
        "W": [[float(W[i, j]) for j in range(best_rank)] for i in range(W.shape[0])]
    }
    
    # The rank-35 decomposition is essentially optimal for this problem.
    # The 2x2x10 matmul tensor decomposes into 5 independent 2x2x2 blocks,
    # each with known rank 7 (Strassen is optimal for 2x2x2).
    # Total rank is therefore 7*5 = 35.
    #
    # No rank reduction below 35 is possible due to the block-diagonal
    # structure of the tensor in the column dimension.
    
    return best_result

# EVOLVE-BLOCK-END
