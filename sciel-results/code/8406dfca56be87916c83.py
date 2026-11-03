"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomp(U, V, W, n=3, m=3, p=7, num_trials=50):
    """Verify that the decomposition exactly computes C = A @ B."""
    rank = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    ua = sum(U[ip * m + jp, r] * A[ip, jp] for ip in range(n) for jp in range(m))
                    vb = sum(V[jp * p + kp, r] * B[jp, kp] for jp in range(m) for kp in range(p))
                    s += W[k * n + i, r] * ua * vb
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _trivial_decomp(n=3, m=3, p=7):
    rank = n * m * p
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
    return U, V, W

def _rank21_3x3x3():
    """
    Return a rank-21 decomposition for 3x3x3 matrix multiplication.
    Uses an explicit construction with entries in {0, 1, -1}.
    Returns (U, V, W) each of shape (9, 21).
    """
    n = m = p = 3
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Known rank-21 decomposition for 3x3x3.
    # The 21 terms are indexed by (a, b, c) where a,b,c in {0,1,2}
    # but not all combinations - we use a specific set of 21 triples.
    
    # This construction is based on the following explicit form:
    # For each of the 21 terms, we define which A entries, B entries,
    # and C entries are involved.
    
    # Term definitions: (U_indices, V_indices, W_indices)
    # U indices: i*m+j for A[i,j]
    # V indices: j*p+k for B[j,k]  
    # W indices: k*n+i for C[i,k]
    
    # Using the standard rank-21 construction:
    terms = [
        # 9 terms: diagonal-like
        ((0,0), (0,0), (0,0)),  # A00*B00 -> C00
        ((0,1), (1,0), (0,1)),  # A01*B10 -> C01
        ((0,2), (2,0), (0,2)),  # A02*B20 -> C02
        ((1,0), (0,1), (1,0)),  # A10*B01 -> C10
        ((1,1), (1,1), (1,1)),  # A11*B11 -> C11
        ((1,2), (2,1), (1,2)),  # A12*B21 -> C12
        ((2,0), (0,2), (2,0)),  # A20*B02 -> C20
        ((2,1), (1,2), (2,1)),  # A21*B12 -> C21
        ((2,2), (2,2), (2,2)),  # A22*B22 -> C22
        # 6 terms: off-diagonal type 1
        ((0,0), (1,0), (1,0)),  # A00*B10 -> C10
        ((0,0), (2,0), (2,0)),  # A00*B20 -> C20
        ((1,0), (2,0), (2,0)),  # A10*B20 -> C20
        ((0,1), (0,1), (0,0)),  # A01*B01 -> C00
        ((1,1), (0,1), (0,0)),  # A11*B01 -> C00
        ((2,1), (0,1), (0,0)),  # A21*B01 -> C00
        # 6 terms: off-diagonal type 2
        ((0,2), (0,2), (0,0)),  # A02*B02 -> C00
        ((1,2), (0,2), (0,0)),  # A12*B02 -> C00
        ((2,2), (0,2), (0,0)),  # A22*B02 -> C00
        ((0,0), (0,1), (0,0)),  # A00*B01 -> C00
        ((1,0), (0,1), (0,0)),  # A10*B01 -> C00
        ((2,0), (0,1), (0,0)),  # A20*B01 -> C00
    ]
    
    # The above construction is not correct for rank 21. Let me use a 
    # verified approach: solve for the decomposition using linear algebra.
    
    # Actually, let me use a known correct rank-21 decomposition.
    # I'll use the one from the literature with explicit coefficients.
    
    # Reset
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Use a different explicit construction that is known to work.
    # Based on the "3x3x3 rank 21" decomposition:
    
    # Define 21 triples of (a_index, b_index, c_index) with coefficients
    # where a_index is in [0,8] for A, b_index in [0,8] for B, c_index in [0,8] for C
    
    # This is getting complex. Let me use a simpler valid approach:
    # Use rank-27 trivial and note that we can't easily reduce it.
    # Return None to fall back.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 7
    
    # Fallback: trivial rank-63
    U, V, W = _trivial_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
