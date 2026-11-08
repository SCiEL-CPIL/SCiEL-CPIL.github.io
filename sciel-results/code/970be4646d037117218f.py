"""
Rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication.

We keep the Kronecker / block-Strassen design (rank 42) from the seed program,
but restructure the code for clarity, add a robust deadline guard, and verify
the decomposition with a few random tests before returning.

The multiplication tensor T_{2,4,6} can be decomposed by splitting the 4
summation index into two pairs and the 6 output columns into three pairs.
For each (A-column-pair, B-row-pair, C-column-pair) we have a 2x2x2
subproblem, which Strassen solves in 7 rank-1 terms. There are
2 * 2 * 3 = 12 such subproblems, giving 12 * 7 = 84 terms.

Wait — that's 84, not 42. Let me reconsider.

Actually, the seed program's construction gives 42 terms. Let me trace through:
- a_block in range(2): 2 choices
- b_pair in range(2): 2 choices
- c_block in range(3): 3 choices
- 7 Strassen terms each
Total: 2 * 2 * 3 * 7 = 84.

But the seed says rank=42 and checks col==42. Let me re-examine...

Looking at the seed code more carefully:
  col = 0
  for a_block in range(2):      # 2
      for b_pair in range(2):    # 2
          for c_block in range(3): # 3
              ...
              for (u_sp, v_sp, w_sp) in strassen_terms:  # 7
                  ...
                  col += 1

That's 2*2*3*7 = 84 increments of col. But rank is set to 42. This would
cause the check `if col != rank` to raise ValueError!

So the seed program's _build_strassen_42 actually FAILS its own check and
falls back to schoolbook (rank 48). That's why the score is 1.0 (same as
schoolbook would give if the evaluator normalizes... wait, no, the score is
the rank value, lower is better. Schoolbook rank is 48. If the seed returns
schoolbook, score should be 48, not 1.0.

Hmm, the task says "score is the rank" and "the evaluator measures the raw
value (lower is better). The search score is seed/value, so 1.0 equals the
seed program." So seed score = 1.0 means the seed's rank is the baseline.
If seed returns schoolbook (rank 48), then seed value = 48, and score = 48/48 = 1.0.

So the seed program is actually returning the schoolbook decomposition (rank 48)
because its _build_strassen_42 function has a bug (col reaches 84, not 42).

This means we can easily beat it by providing a correct rank-42 or even
rank-48 decomposition. But can we do better than 48?

Let me think about the true minimal rank. The 2x4x6 multiplication tensor...

Actually, let me just fix the bug and provide a correct decomposition. The
simplest correct approach is the schoolbook (rank 48), but we can do better.

For 2x4x6:
- We can view this as computing C = A * B where A is 2x4, B is 4x6.
- The standard schoolbook uses 48 multiplications.
- Can we use Strassen-like tricks?

One approach: split A into two 2x2 blocks (columns 0-1 and 2-3), and B into
two 4x3 blocks (rows 0-1 and 2-3, each with 3 column pairs). Then:
C = A_left * B_top + A_right * B_bottom

where A_left is 2x2, B_top is 2x6, A_right is 2x2, B_bottom is 2x6.

Computing A_left * B_top (2x2 times 2x6): this is like computing 2x2 * 2x6,
which can be done with 2 * 7 = 14 multiplications (Strassen for the 2x2 part,
applied to each of the 3 column pairs of the 6 columns).

Similarly A_right * B_bottom: another 14 multiplications.

Total: 28 multiplications. So rank ≤ 28!

Let me verify: 2x2 * 2x6. Write B_top as [b0, b1, b2] where each b_i is a
2x2 matrix (columns 2i, 2i+1). Then A_left * B_top = [A_left*b0, A_left*b1,
A_left*b2]. Each A_left*b_i is a 2x2 matrix product, which Strassen does in
7 multiplications. So 3 * 7 = 21 multiplications for A_left * B_top.

Wait, that's 21, not 14. Let me recount.

2x2 * 2x6: The 2x6 matrix B_top has 6 columns. We can group them into 3
pairs. For each pair, we compute a 2x2 product using Strassen (7 mults).
So 3 * 7 = 21 multiplications.

Similarly for A_right * B_bottom: 21 multiplications.

Total: 21 + 21 = 42. That matches the seed's intended rank of 42!

So the correct construction is:
- For A_left (cols 0,1) * B_top (rows 0,1; all 6 cols): 21 terms
- For A_right (cols 2,3) * B_bottom (rows 2,3; all 6 cols): 21 terms
Total: 42 terms.

The seed's bug was that it was iterating over a_block, b_pair, AND c_block,
which double-counted. The correct approach is:
- For each of the 2 A-blocks (left, right):
  - For the corresponding B-row-pair (top for left, bottom for right):
    - For each of the 3 C-column-blocks:
      - Apply 7 Strassen terms

That's 2 * 1 * 3 * 7 = 42. The b_pair should NOT be a free loop variable;
it should be determined by a_block (a_block=0 → b_pair=0, a_block=1 → b_pair=1).

Let me implement this correctly.
"""

import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE - 10.0


def _schoolbook(n, m, p):
    """Fallback O(nmp) decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_42():
    """
    Build the 42-rank decomposition for 2x4 x 4x6.

    Layout:
      A (2x4): U[i*4 + j] for A[i,j]
      B (4x6): V[j*6 + k] for B[j,k]
      C (2x6): W[k*2 + i] for C[i,k]  (i.e., W[k*n + i])

    We split:
      A_left = A[:, 0:2]  (2x2)
      A_right = A[:, 2:4] (2x2)
      B_top = B[0:2, :]   (2x6)
      B_bottom = B[2:4, :] (2x6)

    C = A_left @ B_top + A_right @ B_bottom

    For A_left @ B_top: split B_top into 3 column-pairs, each a 2x2 block.
    Apply Strassen (7 terms) to each 2x2 x 2x2 product. Total: 3*7 = 21.
    Similarly for A_right @ B_bottom: 21 terms.
    Total: 42 terms.
    """
    n, m, p = 2, 4, 6
    rank = 42

    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows

    # Strassen 7-term formula for 2x2 x 2x2 -> 2x2.
    # Local A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),
    ]

    # For a_block in {0, 1}:
    #   a_block=0: A uses cols 0,1; B uses rows 0,1
    #   a_block=1: A uses cols 2,3; B uses rows 2,3
    # For c_block in {0, 1, 2}:
    #   C uses cols 2*c_block, 2*c_block+1

    col = 0
    for a_block in range(2):
        a0 = 2 * a_block  # starting column of A block
        b0 = 2 * a_block  # starting row of B block (linked!)

        for c_block in range(3):
            c0 = 2 * c_block  # starting column of C block

            # Global indices for the 2x2 A block
            a_global = [
                0 * 4 + a0,       # A[0, a0]
                0 * 4 + a0 + 1,   # A[0, a0+1]
                1 * 4 + a0,       # A[1, a0]
                1 * 4 + a0 + 1,   # A[1, a0+1]
            ]

            # Global indices for the 2x2 B block
            b_global = [
                b0 * 6 + c0,           # B[b0, c0]
                b0 * 6 + c0 + 1,       # B[b0, c0+1]
                (b0 + 1) * 6 + c0,     # B[b0+1, c0]
                (b0 + 1) * 6 + c0 + 1, # B[b0+1, c0+1]
            ]

            # Global indices for the 2x2 C block
            # C[i, k] is at W[k*2 + i]
            c_global = [
                2 * c0 + 0,     # C[0, c0]   = W[2*c0]
                2 * c0 + 2,     # C[0, c0+1] = W[2*(c0+1)]
                2 * c0 + 1,     # C[1, c0]   = W[2*c0 + 1]
                2 * c0 + 3,     # C[1, c0+1] = W[2*(c0+1) + 1]
            ]

            for (u_sp, v_sp, w_sp) in strassen_terms:
                for (loc, coeff) in u_sp:
                    U[a_global[loc]][col] += coeff
                for (loc, coeff) in v_sp:
                    V[b_global[loc]][col] += coeff
                for (loc, coeff) in w_sp:
                    W[c_global[loc]][col] += coeff
                col += 1

    if col != rank:
        raise ValueError(f"Expected {rank} columns, got {col}")

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=4, p=6, atol=1e-10, seed=42):
    """Verify the decomposition against random matrices."""
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C_computed[i, k] += np.dot(
                    U_arr[i * m + j, :],
                    V_arr[j * p + k, :] * W_arr[k * n + i, :]
                )

    return np.allclose(C_computed, C_expected, atol=atol)


def construct():
    global _START
    _START = time.time()

    # Try the rank-42 Strassen decomposition
    try:
        result = _build_strassen_42()
        if _verify(result["U"], result["V"], result["W"], seed=42):
            if _verify(result["U"], result["V"], result["W"], seed=123):
                return result
    except Exception:
        pass

    if _deadline_exceeded():
        return _schoolbook(2, 4, 6)

    # Fallback to schoolbook
    return _schoolbook(2, 4, 6)
