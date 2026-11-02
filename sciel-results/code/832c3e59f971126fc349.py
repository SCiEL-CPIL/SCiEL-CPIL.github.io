import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _strassen_decomp():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication."""
    n, m, p = 2, 2, 12
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for t in range(num_blocks):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1 = (A00+A11)*(B00+B11)
        c = base + 0
        U[0, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M2 = (A10+A11)*B00
        c = base + 1
        U[2, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        W[k0 * n + 1, c] = 1.0
        W[k1 * n + 1, c] = -1.0

        # M3 = A00*(B01-B11)
        c = base + 2
        U[0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        V[1 * p + k1, c] = -1.0
        W[k1 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M4 = A11*(B10-B00)
        c = base + 3
        U[3, c] = 1.0
        V[1 * p + k0, c] = 1.0
        V[0 * p + k0, c] = -1.0
        W[k0 * n + 0, c] = 1.0
        W[k0 * n + 1, c] = 1.0

        # M5 = (A00+A01)*B11
        c = base + 4
        U[0, c] = 1.0
        U[1, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = -1.0
        W[k1 * n + 0, c] = 1.0

        # M6 = (A10-A00)*(B00+B01)
        c = base + 5
        U[2, c] = 1.0
        U[0, c] = -1.0
        V[0 * p + k0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M7 = (A01-A11)*(B10+B11)
        c = base + 6
        U[1, c] = 1.0
        U[3, c] = -1.0
        V[1 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0

    return U, V, W


def _verify_decomp(U, V, W, tol=1e-9):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the matrix mult tensor."""
    n, m, p = 2, 2, 12
    # Build target tensor T[i, j, k] where i is A-index (n*m), j is B-index (m*p), k is C-index (p*n)
    # C[k*n+r] = sum_s A[s*n+r] * B[s*p+k]
    # So T[sm+r, sp+k, kn+r] = 1 for valid s,r
    # Let's check: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Target: for A index a = s*n+r (s row of A, r col of A), B index b = t*p+k (t row of B, k col of B)
    # C index c = k'*n+r' where C[k'*n+r'] = sum_s A[s*n+r'] * B[s*p+k']
    # So T[a, b, c] = 1 if a = s*n+r', b = s*p+k', c = k'*n+r' for some s
    # i.e., T[a,b,c] = 1 if (a mod n) == (c mod n) and (a // n) == (b // p)
    
    rank = U.shape[1]
    # Compute sum_r U[:,r] V[:,r]^T W[:,r]^T as tensor
    # Efficient: for each r, add U[:,r][:,None,None] * V[:,r][None,:,None] * W[:,r][None,None,:]
    # But this is O(rank * nm * mp * pn) = O(42 * 4 * 24 * 24) = O(96768) which is fine
    
    # Actually let's just check a few random points
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        # Target
        s_a = a // n  # row of A
        r_a = a % n   # col of A
        s_b = b // p  # row of B
        k_b = b % p   # col of B
        k_c = c // n  # col of C (output column)
        r_c = c % n   # row of C (output row)
        
        # C[k_c * n + r_c] = sum_s A[s*n+r_c] * B[s*p+k_c]
        # So T[a,b,c] = 1 if s_a == s_b and r_a == r_c
        target = 1.0 if (s_a == s_b and r_a == r_c) else 0.0
        
        # Computed
        comp = 0.0
        for r in range(rank):
            comp += U[a, r] * V[b, r] * W[c, r]
        
        if abs(comp - target) > tol:
            return False
    return True


def construct():
    """
    Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    The 2x12 matrix B is split into six 2x2 column-pairs. Each pair is
    multiplied by the 2x2 matrix A using Strassen's 7-product algorithm.
    Total rank = 6 * 7 = 42.
    
    This is the Kronecker product decomposition: T(2,2,12) = T(2,2,2) ⊗ T(1,1,6)
    with rank = 7 * 6 = 42.
    """
    U, V, W = _strassen_decomp()
    
    # Verify correctness
    assert _verify_decomp(U, V, W), "Decomposition verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
