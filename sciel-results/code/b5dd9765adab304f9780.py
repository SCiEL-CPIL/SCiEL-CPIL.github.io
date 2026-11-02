"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _build_rank_56():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Partition into 8 independent 2x2 * 2x2 block multiplications,
    each decomposed via Strassen (rank 7). Total rank = 8 * 7 = 56.
    
    Index conventions:
    A[i,j] -> U index: i*m + j = i*4 + j
    B[j,k] -> V index: j*p + k = j*8 + k
    C[i,k] -> W index: k*n + i = k*2 + i
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11 = 0 * 4 + 2 * u
            a12 = 0 * 4 + 2 * u + 1
            a21 = 1 * 4 + 2 * u
            a22 = 1 * 4 + 2 * u + 1
            
            b11 = 2 * u * 8 + 2 * v
            b12 = 2 * u * 8 + 2 * v + 1
            b21 = (2 * u + 1) * 8 + 2 * v
            b22 = (2 * u + 1) * 8 + 2 * v + 1
            
            c11 = 4 * v
            c12 = 4 * v + 2
            c21 = 4 * v + 1
            c22 = 4 * v + 3
            
            U[a11][base_col] += 1
            U[a22][base_col] += 1
            V[b11][base_col] += 1
            V[b22][base_col] += 1
            W[c11][base_col] += 1
            W[c22][base_col] += 1
            
            U[a21][base_col + 1] += 1
            U[a22][base_col + 1] += 1
            V[b11][base_col + 1] += 1
            W[c21][base_col + 1] += 1
            W[c22][base_col + 1] -= 1
            
            U[a11][base_col + 2] += 1
            V[b12][base_col + 2] += 1
            V[b22][base_col + 2] -= 1
            W[c12][base_col + 2] += 1
            W[c22][base_col + 2] += 1
            
            U[a22][base_col + 3] += 1
            V[b21][base_col + 3] += 1
            V[b11][base_col + 3] -= 1
            W[c11][base_col + 3] += 1
            W[c21][base_col + 3] += 1
            
            U[a11][base_col + 4] += 1
            U[a12][base_col + 4] += 1
            V[b22][base_col + 4] += 1
            W[c11][base_col + 4] -= 1
            W[c12][base_col + 4] += 1
            
            U[a21][base_col + 5] += 1
            U[a11][base_col + 5] -= 1
            V[b11][base_col + 5] += 1
            V[b12][base_col + 5] += 1
            W[c22][base_col + 5] += 1
            
            U[a12][base_col + 6] += 1
            U[a22][base_col + 6] -= 1
            V[b21][base_col + 6] += 1
            V[b22][base_col + 6] += 1
            W[c11][base_col + 6] += 1
    
    return {"U": U, "V": V, "W": W}


def _build_rank_48():
    """
    Attempt to build a rank-48 decomposition by sharing some Strassen forms
    across adjacent 2x2 blocks. If verification fails, return None.
    
    Strategy: Use 6 independent Strassen blocks (rank 42) plus 6 shared forms
    that handle cross-block interactions. This is a heuristic attempt.
    """
    n, m, p = DIMENSIONS
    total_rank = 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Use 6 full Strassen blocks for the first 6 (u,v) pairs
    for idx in range(6):
        u = idx // 4
        v = idx % 4
        base_col = 7 * idx
        
        a11 = 0 * 4 + 2 * u
        a12 = 0 * 4 + 2 * u + 1
        a21 = 1 * 4 + 2 * u
        a22 = 1 * 4 + 2 * u + 1
        
        b11 = 2 * u * 8 + 2 * v
        b12 = 2 * u * 8 + 2 * v + 1
        b21 = (2 * u + 1) * 8 + 2 * v
        b22 = (2 * u + 1) * 8 + 2 * v + 1
        
        c11 = 4 * v
        c12 = 4 * v + 2
        c21 = 4 * v + 1
        c22 = 4 * v + 3
        
        U[a11][base_col] += 1
        U[a22][base_col] += 1
        V[b11][base_col] += 1
        V[b22][base_col] += 1
        W[c11][base_col] += 1
        W[c22][base_col] += 1
        
        U[a21][base_col + 1] += 1
        U[a22][base_col + 1] += 1
        V[b11][base_col + 1] += 1
        W[c21][base_col + 1] += 1
        W[c22][base_col + 1] -= 1
        
        U[a11][base_col + 2] += 1
        V[b12][base_col + 2] += 1
        V[b22][base_col + 2] -= 1
        W[c12][base_col + 2] += 1
        W[c22][base_col + 2] += 1
        
        U[a22][base_col + 3] += 1
        V[b21][base_col + 3] += 1
        V[b11][base_col + 3] -= 1
        W[c11][base_col + 3] += 1
        W[c21][base_col + 3] += 1
        
        U[a11][base_col + 4] += 1
        U[a12][base_col + 4] += 1
        V[b22][base_col + 4] += 1
        W[c11][base_col + 4] -= 1
        W[c12][base_col + 4] += 1
        
        U[a21][base_col + 5] += 1
        U[a11][base_col + 5] -= 1
        V[b11][base_col + 5] += 1
        V[b12][base_col + 5] += 1
        W[c22][base_col + 5] += 1
        
        U[a12][base_col + 6] += 1
        U[a22][base_col + 6] -= 1
        V[b21][base_col + 6] += 1
        V[b22][base_col + 6] += 1
        W[c11][base_col + 6] += 1
    
    # For the remaining 2 blocks (u=1, v=0) and (u=1, v=1), use simplified forms
    # This is a placeholder - in practice this would need careful construction
    # to ensure correctness. For now, return None to indicate this approach
    # needs more work.
    return None


def _verify(U, V, W, n_tests=20):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(42)
    for _ in range(n_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(len(U[0])):
                    a_part = sum(U[i*m+j][r] * A[i,j] for j in range(m))
                    b_part = sum(V[j*p+k][r] * B[j,k] for j in range(m))
                    C_computed[i,k] += W[k*n+i][r] * a_part * b_part
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline (safety margin before 120s limit)
    
    # Try rank-48 first (if available)
    result = _build_rank_48()
    if result is not None:
        if time.time() < deadline:
            if _verify(result['U'], result['V'], result['W']):
                return result
    
    # Fall back to rank-56
    return _build_rank_56()

# EVOLVE-BLOCK-END
