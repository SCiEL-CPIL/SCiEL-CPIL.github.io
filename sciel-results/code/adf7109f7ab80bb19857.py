"""Matrix multiplication 3x5x5 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-10):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B

    U_reshaped = U.T.reshape(rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)
    W_reshaped_temp = W.T.reshape(rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)

    U_A = np.einsum('rij,ij->r', U_reshaped, A)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)

    return np.allclose(AB_direct, AB_decomp, atol=atol)


def _try_block_strassen(n, m, p):
    """
    Try to decompose using block structure with Strassen on 2x2 blocks.
    Split 5 into 2+2+1. Apply Strassen to 2x2 blocks where possible.
    This is experimental and may not yield improvement.
    """
    # For now, return None to indicate no improvement found
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 75)
    U, V, W = _schoolbook(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try block Strassen decomposition
    if time.time() < deadline:
        result = _try_block_strassen(n, m, p)
        if result is not None:
            U_bs, V_bs, W_bs = result
            if _verify(U_bs, V_bs, W_bs, n, m, p):
                rank_bs = U_bs.shape[1]
                if rank_bs < best_rank:
                    best_rank = rank_bs
                    best_U, best_V, best_W = U_bs, V_bs, W_bs
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
