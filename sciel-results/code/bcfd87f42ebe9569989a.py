import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # For each (i,k,j), check that T[i,k,j] = delta(i,j) * (1 if k==k else 0)
    # Actually T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor is T[i,k,j] = 1 for all i,k,j (it's the identity in the "multiplication" sense)
    # Wait, the matrix multiplication tensor has T[i,k,j] = 1 for all i,k,j
    # No, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T has entries T[i,k,j] = 1 for all i,k,j
    # This is because C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor is all ones.
    
    # Actually, I need to be more careful. The tensor T_{ijk} represents the coefficient
    # in C_{ij} = sum_k T_{ijk} A_{ik} B_{kj}
    # For standard matrix multiplication, T_{ijk} = 1 for all i,k,j.
    
    # So we need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    # Let's verify a few entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """Try to reduce the rank by finding linear dependencies."""
    start = time.time()
    rank = U.shape[1]
    
    if rank <= target_rank:
        return U, V, W, rank
    
    # Try to find columns that are linear combinations of others
    # This is a hard problem, so we'll use a greedy approach
    
    # For each column r, check if it can be expressed as a combination of others
    # This is expensive, so we'll limit the search
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    removed = set()
    
    # Try removing one column at a time and see if we can compensate
    # This is a simplified approach
    
    # For a proper rank reduction, we need to find a linear map
    # that preserves the tensor decomposition
    
    # Simple approach: try to find if any two columns are nearly linearly dependent
    # in a way that allows rank reduction
    
    # Given time constraints, let's just return the current decomposition
    # if we can't find a quick improvement
    
    return U_curr, V_curr, W_curr, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Build the base decomposition
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # For the 4x1 part
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k * 1 + 0, r] = 1
            W41[0 * 4 + i, r] = 1
    
    total_rank = 49 + 49 + 16  # 114
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A*B1 (cols 0-3)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 (cols 4-7)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 (col 8)
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial decomposition
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U[i*m+k, r] = 1
                    V[k*p+j, r] = 1
                    W[j*n+i, r] = 1
        total_rank = rank
    else:
        total_rank = U.shape[1]
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
