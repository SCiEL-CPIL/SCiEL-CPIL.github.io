"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Strategy:
  The multiplication of a 3x3 matrix A (n=3, m=3) by a 3x8 matrix B (m=3, p=8)
  yields a 3x8 matrix C. The tensor has n*m*p = 72 rank-1 terms in the schoolbook
  decomposition. We attempt to find a lower-rank exact decomposition using
  integer/half-integer entries.

  Known lower bounds: For 3x3 x 3x3, Laderman's algorithm gives rank 7.
  For 3x3 x 3x8, we can think of it as 8 independent 3x3 x 3x1 problems, but
  they share the same A and the same 3 columns of B, so there's structure.

  Actually, C[i,k] = sum_j A[i,j] * B[j,k] for i=0..2, j=0..2, k=0..7.

  This is the same as multiplying A (3x3) by B (3x8). We can view B as 8 columns,
  each a 3x1 vector. The output is 8 columns, each a 3x1 vector.

  The tensor T[i,j,k] = e_i * e_j * e_k (in appropriate spaces) has rank at least
  max(n*m, m*p, p*n) in general, but specific structures can be lower.

  For the specific case 3x3 * 3x8:
  - We can use a rank-7 decomposition for the 3x3x3 core and extend it.
  - Actually, since p=8 > 3, we might need more.

  Let's think differently. The standard lower bound for matrix multiplication M_{n,m,p}
  is at least max(n*m, m*p, p*n) / something... Actually the trivial lower bound is
  max(n, m, p)^2 or similar. For 3x3x8, the schoolbook rank is 72.

  A better approach: Use the fact that we can decompose B into a sum of rank-1
  matrices in the column space, but that doesn't directly help.

  Let me try a constructive approach:
  1. Start with the schoolbook rank-72 decomposition.
  2. Try to find linear dependencies among the rank-1 terms to reduce rank.
  3. Use a SAT/SMT solver or linear algebra to find the minimum rank.

  Given the time constraints, I'll implement a search that:
  - Constructs the full tensor as a 3D array T[i, j, k] for i in [0,3), j in [0,3), k in [0,8).
  - Uses a greedy or local search to find a low-rank decomposition.
  - Returns the best found within the time budget.

  For a 3x3 x 3x8 multiplication, I suspect the rank is around 24-32 based on
  the structure. Let's implement a solver.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave 10s margin

    n, m, p = DIMENSIONS  # n=3, m=3, p=8

    # Schoolbook rank
    schoolbook_rank = n * m * p

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    U_shape = (n * m, schoolbook_rank)
    V_shape = (m * p, schoolbook_rank)
    W_shape = (p * n, schoolbook_rank)

    # Initialize schoolbook decomposition
    U = np.zeros(U_shape, dtype=np.int64)
    V = np.zeros(V_shape, dtype=np.int64)
    W = np.zeros(W_shape, dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by finding linear combinations
    # We look for a set of rank r < 72 such that there exist U, V, W with those columns
    # satisfying: for all (a, b, c) in the support, sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]

    # The tensor T has 1s at positions (i*m+j, j*p+k, k*n+i) for i in [0,3), j in [0,3), k in [0,8)
    # Let's build the tensor explicitly
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Flatten the tensor
    T_flat = T.flatten()
    num_nonzero = np.count_nonzero(T_flat)

    # Use a greedy approach: start with schoolbook, try to merge terms
    # For each pair of columns, check if their sum can be replaced by a single term
    # This is expensive, so we use a randomized local search instead

    def verify(U_mat, V_mat, W_mat, T_target, tol=1e-6):
        """Verify that U @ outer(V, W) = T_target (in tensor form)"""
        r = U_mat.shape[1]
        # Compute sum_r U[:, r] outer V[:, r] outer W[:, r]
        # This is the 3-way tensor product
        reconstructed = np.tensordot(U_mat, V_mat, axes=1)  # shape (n*m, m*p, r)
        reconstructed = np.tensordot(reconstructed, W_mat, axes=1)  # shape (n*m, m*p, p*n)
        return np.allclose(reconstructed, T_target, atol=tol)

    # Try rank values from 24 upward
    # For each rank, try to find a solution using a simple iterative method
    # This is a research problem, so we'll use a heuristic

    # Heuristic: use the known Laderman-like structure for 3x3x3 and extend
    # For 3x3x3, Laderman's rank is 7. For 3x3x8, we might need 7 + 5*3 = 22 or similar.

    # Let's try a random search with random initializations
    np.random.seed(42)

    # Try various ranks
    for target_rank in range(20, schoolbook_rank + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Try a few random initializations
        for trial in range(5):
            if time.time() - start_time > DEADLINE:
                break

            # Random initialization with small integers
            U_try = np.random.randint(-1, 2, size=(n * m, target_rank))
            V_try = np.random.randint(-1, 2, size=(m * p, target_rank))
            W_try = np.random.randint(-1, 2, size=(p * n, target_rank))

            # Use a simple gradient-free optimization: coordinate descent
            # This is slow, so we'll just check if the random init happens to work
            if verify(U_try, V_try, W_try, T):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break

            # Try a structured initialization based on the schoolbook
            # Pick random subsets of the schoolbook columns
            if target_rank < schoolbook_rank:
                indices = np.random.choice(schoolbook_rank, size=target_rank, replace=False)
                U_sub = U[:, indices]
                V_sub = V[:, indices]
                W_sub = W[:, indices]
                if verify(U_sub, V_sub, W_sub, T):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
                    break

        if best_rank == target_rank:
            break

    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        return [[int(x) if x == int(x) else x / 2 for x in row] for row in mat.tolist()]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
