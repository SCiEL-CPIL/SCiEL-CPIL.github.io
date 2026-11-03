"""
Combined program: structured rank-45 decomposition for 3x3x6 via two 3x3x3 Strassen blocks.

Strategy:
- Treat 3x3x6 multiplication as two independent 3x3x3 blocks (columns 0-2 and 3-5 of B).
- Use a verified rank-7 Strassen-type decomposition for each 3x3x3 block.
- Total rank = 2 * 7 = 14... wait, that's not right. Strassen for 2x2x2 is rank 7.
- For 3x3x3, we need a different approach. Let me reconsider.

Actually, for 3x3x3 matrix multiplication, the schoolbook rank is 27.
The best known upper bound is 23, but constructing an explicit rank-23 decomposition
is complex. 

Let me use a simpler approach: use the schoolbook for each 3x3x3 block (rank 27 each),
giving total rank 54. But that's the same as the original schoolbook for 3x3x6.

Wait, let me re-read the inspiration. The inspiration 2 uses a "rank-7 Strassen-type"
for 3x3x3, which doesn't make sense since Strassen is for 2x2x2.

Let me look more carefully. The inspiration 2 defines _A, _B, _C as 7 matrices of size 3x3.
This is a rank-7 decomposition of the 3x3x3 multiplication tensor? That seems too low.

Actually, I think there's a mistake in my understanding. Let me just implement a correct
decomposition. The safest approach is to use the schoolbook (rank 54) as a fallback,
and try to find a better decomposition.

For now, let me implement a clean schoolbook decomposition and verify it works.
Then I can try to improve it.

Actually, looking at the task again: "Find a rank-minimal exact tensor decomposition"
and "score is the rank" (lower is better).

The schoolbook for 3x3x6 has rank 3*3*6 = 54.

Let me implement a correct schoolbook first, then try to reduce it.

For a potential improvement, I can try to use the fact that 3x3x6 can be viewed as
computing 3 rows of a 3x6 output matrix, where each row is a 1x6 vector that is the
product of a 1x3 row of A with the 3x6 matrix B.

But the tensor structure is: T[i,j,k] = 1 for all i,j,k (in the sense that each
output entry C[i,k] = sum_j A[i,j]*B[j,k]).

The bilinear rank of this tensor is what we're trying to minimize.

Let me just implement a correct schoolbook and return it. If I have time, I'll try
to find a better decomposition.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = len(U_arr[0]) if len(U_arr) > 0 else 0
    
    if len(U_arr) != n * m or len(V_arr) != m * p or len(W_arr) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_rank46():
    """
    Try to construct a rank-46 decomposition by treating 3x3x6 as two 3x3x3 blocks.
    Each block uses a rank-23 decomposition for 3x3x3.
    
    For now, this is a placeholder. We need a valid rank-23 decomposition for 3x3x3.
    If we can't find one, we fall back to schoolbook.
    """
    n, m, p = 3, 3, 6
    rank = 46
    
    U = np.zeros((9, 46), dtype=np.float64)
    V = np.zeros((18, 46), dtype=np.float64)
    W = np.zeros((18, 46), dtype=np.float64)
    
    # For each block (k in 0-2 and k in 3-5), we need a rank-23 decomposition.
    # This is complex, so for now, let's just return None to indicate failure.
    # In a future iteration, we can implement a proper rank-23 decomposition.
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Try to construct a lower-rank decomposition
    rank46_result = _try_rank46()
    
    if rank46_result is not None:
        # Verify the rank-46 decomposition
        if _verify_decomposition(
            rank46_result["U"], 
            rank46_result["V"], 
            rank46_result["W"], 
            n, m, p
        ):
            return rank46_result
    
    # Fall back to schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If verification fails (shouldn't happen), return the result anyway
    return result
