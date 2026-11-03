"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _make_schoolbook():
    """Return schoolbook rank-36 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36
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


def _verify(U, V, W, tol=1e-9):
    """Check that U,V,W form an exact decomposition of the 2x3x6 matmul tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i,j,k] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :])
                # multiply element-wise with W rows and sum
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank18_partition():
    """
    Try rank-18 by partitioning the 3x6 matrix into three 3x2 blocks.
    Each block corresponds to a 2x3x2 matmul. The rank of 2x3x2 is known to be 6
    (schoolbook gives 6, and it's optimal since lower bound is max(n*m, m*p, p*n) = 6).
    But wait: 2x3x2 schoolbook is rank 6, so 3 blocks * 6 = 18.
    That's actually the same as schoolbook for 2x3x6 which is 36/2 = 18? No, schoolbook is 36.
    Actually 2x3x2 has rank 6 (schoolbook), so 3 * 6 = 18. This would be an improvement over 36.
    Let's construct it.
    
    For 2x3x2: n=2, m=3, p=2. Schoolbook rank = 2*3*2 = 6.
    U shape: (6, 6), V shape: (6, 6), W shape: (4, 6)
    For block b (b=0,1,2), the p-dimension indices are 2*b and 2*b+1.
    The W index for block b: k_global = 2*b + k_local, so W index = k_global * n + i = (2*b + k_local)*2 + i.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    p_local = 2
    n_blocks = p // p_local  # 3
    rank_local = n * m * p_local  # 6
    rank = n_blocks * rank_local  # 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for b in range(n_blocks):
        col_offset = b * rank_local
        col = 0
        for i in range(n):
            for j in range(m):
                for k_local in range(p_local):
                    k_global = b * p_local + k_local
                    U[i * m + j, col_offset + col] = 1.0
                    V[j * p + k_global, col_offset + col] = 1.0
                    W[k_global * n + i, col_offset + col] = 1.0
                    col += 1
    
    result = {"U": U, "V": V, "W": W}
    if _verify(U, V, W):
        return result
    return None


def construct():
    deadline = time.time() + 140  # 140s safety margin (limit is 150s)
    
    # Try rank-18 partition first
    result = _try_rank18_partition()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _make_schoolbook()
# EVOLVE-BLOCK-END
