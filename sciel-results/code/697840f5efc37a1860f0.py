"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U, V, W, best_rank = make_schoolbook()
    best_U, best_V, best_W = U, V, W

    deadline = time.time() + 140.0

    # Try to find a lower-rank decomposition using random initialization and iterative refinement
    # We'll use a simple approach: try random rank-r decompositions and see if they can be completed

    # For matrix multiplication tensor T[i,j,k] = delta (but it's structured)
    # The tensor is T[i,j,k] where the multiplication is (A B)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for all valid i,j,k in the standard basis
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is defined such that C[i,k] = sum_{j} sum_{r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So C[i,k] = sum_{j,r} A[i,j] B[j,k] * (something)
    # Actually the tensor decomposition says:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (using the trace convention W[k*n+i])
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.

    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Let's try to find a lower rank decomposition using a grid search or random methods
    # For small dimensions, we can try to verify candidate decompositions

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try ranks below schoolbook
    # Known results: for 3x4 * 4x8, the rank might be less than 96
    # Let's try a few random initializations with rank reduction

    # Strategy: Use the schoolbook as a starting point and try to merge/eliminate terms
    # Or use random low-rank factorizations and check

    # For efficiency, let's try a structured approach:
    # The tensor is separable in a certain way. Let's try rank = n*m*p - something

    # Actually, let's try a different approach: use the fact that the tensor is
    # a sum of rank-1 tensors, and try to find a more efficient representation.

    # For now, let's just try a few random low-rank attempts with verification
    # This is unlikely to find a better solution quickly, but let's try

    rng = np.random.RandomState(42)

    # Try a few ranks
    for rank_try in range(max(1, best_rank - 20), best_rank):
        if time.time() > deadline:
            break
        # Random initialization
        U_r = rng.randn(n * m, rank_try)
        V_r = rng.randn(m * p, rank_try)
        W_r = rng.randn(p * n, rank_try)

        # This random approach won't work well for exact decomposition
        # Let's instead try to use the structure

        # Skip random approach for now, it's too slow to verify

    # Let's try a smarter approach: exploit the structure of the matrix multiplication tensor
    # The tensor T[i,j,k] = 1 can be written as a product of three matrices in a structured way

    # One known technique: if one dimension is small, we can sometimes reduce the rank
    # For n=3, m=4, p=8, let's see if there's a known result

    # For now, return the schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
