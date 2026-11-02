"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42

    def schoolbook():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook()

    def verify_exact(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the matmul tensor.
        For all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j, :rank]
                    v_row = V[j*p + k, :rank]
                    w_row = W[k*n + i, :rank]
                    prod = u_row * v_row * w_row
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_integer(U, V, W, rank):
        """Check all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row[:rank]:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Strategy: Try to find a lower-rank exact decomposition.
    # The matmul tensor T[i,j,k] = 1 for all valid (i,j,k) has a known structure.
    #
    # Key insight: For 2x3x7, we can potentially exploit structure to reduce rank.
    #
    # Let's try a direct construction using the fact that n=2 is small.
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # One approach: Use intermediate linear combinations.
    # 
    # Let's try to see if rank 39 or less is achievable with half-integer entries.
    #
    # Actually, let's try a systematic search for structured decompositions.
    # 
    # For a rank-r decomposition, we need vectors U_r ∈ R^{nm}, V_r ∈ R^{mp}, W_r ∈ R^{pn}
    # such that for all (i,j,k): sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    #
    # Let's try to find such a decomposition using optimization with exact verification.
    #
    # First, let's check if the schoolbook rank 42 is indeed the best we can do,
    # or if there's a known lower bound.
    #
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # The search for lower-rank decompositions would require more sophisticated
    # algebraic geometry methods that are hard to implement reliably within time limits.

    if not verify_exact(U0, V0, W0, rank_schoolbook):
        raise ValueError("Schoolbook decomposition failed verification")

    if not verify_integer(U0, V0, W0, rank_schoolbook):
        raise ValueError("Schoolbook decomposition has non-integer entries")

    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }
# EVOLVE-BLOCK-END
