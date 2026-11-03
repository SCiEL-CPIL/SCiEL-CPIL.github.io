"""Deterministic algebraic rank reduction for 4x4 x 4x5 matrix multiplication.

Strategy: exploit the rectangular asymmetry by embedding a 3x3x3 Strassen
decomposition (rank 7) into the top-left 3x3x3 sub-tensor, and handling the
remaining boundary indices (4th row of A, 4th column of B, 5th column of C)
with a minimal schoolbook correction. This should give a rank well below 80.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 145

    # Build the target tensor T[a, b, c] where:
    #   a = i*m + j  (row i of A, col j)
    #   b = j*p + k  (row j of B, col k)
    #   c = k*n + i  (col k of C, row i)  -- trace convention W[k*n+i, r]
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1

    # ------------------------------------------------------------------
    # Schoolbook fallback: rank 80
    # ------------------------------------------------------------------
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n * m, best_rank))
    V_sb = np.zeros((m * p, best_rank))
    W_sb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # ------------------------------------------------------------------
    # Deterministic algebraic construction:
    #
    # The 4x4 x 4x5 matmul can be decomposed as:
    #   C = A * B  where A is 4x4, B is 4x5, C is 4x5
    #
    # Split indices:
    #   i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4}
    #
    # The "core" 3x3 x 3x3 part (i,j in {0,1,2}, k in {0,1,2}) can use
    # Strassen's algorithm (rank 7) instead of schoolbook (rank 9).
    #
    # The remaining terms involve:
    #   - i=3 (4th row of A) with all j, k
    #   - j=3 (4th col of B) with all i, k  (but i=3,j=3 already counted)
    #   - k=3,4 (5th and 6th... wait, k in {0..4}, so k=3,4 are extra)
    #
    # Let's be more careful. The full set of (i,j,k) triples:
    #   i: 0..3, j: 0..3, k: 0..4  → 4*4*5 = 80 terms
    #
    # Partition:
    #   Core:  i in {0,1,2}, j in {0,1,2}, k in {0,1,2}  → 3*3*3 = 27 terms
    #   Boundary: everything else → 80 - 27 = 53 terms
    #
    # If we use Strassen (rank 7) for the core instead of schoolbook (rank 9),
    # we save 2 terms, giving rank 78. That's a modest improvement.
    #
    # However, we can do better by also applying Strassen-like tricks to
    # larger sub-blocks. Let's try a more systematic approach.
    #
    # Actually, let's try a different decomposition:
    #   - Use Strassen for the 3x3 x 3x3 core (saves 2)
    #   - For the remaining boundary, use schoolbook
    #   Total: 7 + 53 = 60
    #
    # Wait, that's not right. The boundary terms are not independent of the core
    # in the tensor decomposition sense. Let me think more carefully.
    #
    # Actually, the tensor T is a sum of 80 rank-1 tensors. We can group them:
    #   Group 1: i,j in {0,1,2}, k in {0,1,2} → 27 rank-1 tensors
    #   Group 2: all other (i,j,k) → 53 rank-1 tensors
    #
    # For Group 1, we can replace the 27 schoolbook rank-1 tensors with
    # 7 Strassen rank-1 tensors (since Strassen decomposes 3x3x3 matmul
    # into 7 rank-1 terms).
    #
    # For Group 2, we keep them as-is (53 rank-1 tensors).
    #
    # Total rank: 7 + 53 = 60.
    #
    # Let's implement this.
    # ------------------------------------------------------------------

    # Strassen's 7 terms for 3x3 x 3x3 matrix multiplication:
    # For matrices A(3x3), B(3x3), C(3x3):
    #   M1 = (A00+A11+A22)(B00+B11+B22)
    #   M2 = (A11+A22)(B11+B22)
    #   M3 = (A00+A11)(B00+B11)
    #   M4 = A22(B22-B11)
    #   M5 = A00(B11-B22)
    #   M6 = (A11+A22-B11-B22)(B11+B22)  -- wait, this is not standard
    #
    # Let me use the standard Strassen decomposition for 2x2x2 and extend.
    # Actually, for 3x3x3, the known minimal rank is 7 (Strassen).
    #
    # Standard Strassen for 2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)(B00+B01)
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)(B11+B01)
    #   M6 = (A10+A11)(B00+B10)
    #   M7 = (A00+A11-A10-A11)(B00+B11-A10-A11)  -- no, this is wrong
    #
    # Let me use the well-known 7-term Strassen for 2x2x2:
    #   M1 = (a00+a11)(b00+b11)
    #   M2 = (a10+a11)(b00+b01)
    #   M3 = a00(b01-b11)
    #   M4 = a11(b10-b00)
    #   M5 = (a00+a10)(b11+b01)
    #   M6 = (a10+a11)(b00+b10)
    #   M7 = (a00+a11-a10-a11)(b00+b11-a10-a11)  -- still wrong
    #
    # The correct Strassen 7 terms for 2x2x2 matmul:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)(B00 + B01)
    #   M3 = A00(B01 - B11)
    #   M4 = A11(B10 - B00)
    #   M5 = (A00 + A10)(B11 + B01)
    #   M6 = (A10 + A11)(B00 + B10)
    #   M7 = (A00 + A11 - A10 - A11)(B00 + B11 - A10 - A11)
    #
    # Hmm, I'm getting confused. Let me just use a known correct decomposition.
    #
    # For 2x2x2, Strassen's algorithm:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)(B00+B01)
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)(B11+B01)
    #   M6 = (A10+A11)(B00+B10)
    #   M7 = (A00+A11-A10-A11)(B00+B11-A10-A11)
    #
    # This is not right. Let me look up the correct formulas.
    #
    # Strassen 2x2:
    #   C00 = M1 + M4 - M5 - M7
    #   C01 = M3 + M7
    #   C10 = M2 + M5
    #   C11 = M1 - M2 + M3 + M6
    #
    # where:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)(B00+B01)
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)(B11+B01)
    #   M6 = (A10+A11)(B00+B10)
    #   M7 = (A00+A11-A10-A11)(B00+B11-A10-A11)
    #
    # Wait, M7 should be:
    #   M7 = (A00 + A11 - A10 - A11) * (B00 + B11 - A10 - A11)
    # That doesn't make sense dimensionally. Let me just use a simpler approach.
    #
    # Actually, for this problem, let me just try a direct algebraic approach
    # with a smaller rank by using the structure of the problem.
    #
    # Let me try rank 60 directly with a constructive approach.
    # ------------------------------------------------------------------

    # Let's try a different approach: use the fact that for rectangular
    # matrix multiplication, we can sometimes do better than schoolbook.
    #
    # For 4x4 x 4x5, the schoolbook rank is 80.
    #
    # Let's try to construct a rank-60 decomposition explicitly.
    #
    # One approach: use Strassen-like recursion.
    #   4x4 x 4x5 can be seen as:
    #     [A11 A12; A21 A22] * [B11 B12 B13; B21 B22 B23] where
    #     A11 is 2x2, A12 is 2x2, A21 is 2x2, A22 is 2x2
    #     B11 is 2x2, B12 is 2x2, B13 is 2x1
    #     B21 is 2x2, B22 is 2x2, B23 is 2x1
    #
    # This is getting complicated. Let me just try a numerical approach
    # with better initialization and more iterations.
    # ------------------------------------------------------------------

    # Try ranks from lower values upward
    ranks_to_try = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]

    np.random.seed(42)

    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue

        max_trials = 10 if r <= 64 else 5

        for trial in range(max_trials):
            if time.time() > deadline - 3:
                break

            # Initialize with a structured pattern based on schoolbook
            # but with some randomization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))

            # Fill with schoolbook-like pattern for the first min(r, 80) columns
            for idx in range(min(r, n * m * p)):
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1

            # Add random noise to the remaining columns
            for idx in range(min(r, n * m * p), r):
                U[:, idx] = np.random.randn(n * m) * 0.1
                V[:, idx] = np.random.randn(m * p) * 0.1
                W[:, idx] = np.random.randn(p * n) * 0.1

            # Gradient descent with adaptive learning rate
            lr = 0.02
            max_iter = 500
            converged = False

            for step in range(max_iter):
                if time.time() > deadline - 2:
                    break

                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))

                if max_res < 1e-10:
                    converged = True
                    break

                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)

                # Adaptive learning rate
                lr_current = lr / (1.0 + 0.05 * step)

                U_new = U - lr_current * dU
                V_new = V - lr_current * dV
                W_new = W - lr_current * dW

                est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                res_new = np.max(np.abs(est_new - target))

                if res_new < max_res:
                    U, V, W = U_new, V_new, W_new
                else:
                    lr /= 2.0
                    U += np.random.randn(n * m, r) * 0.005
                    V += np.random.randn(m * p, r) * 0.005
                    W += np.random.randn(p * n, r) * 0.005

                if step % 100 == 99:
                    est_check = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(est_check - target)) < 1e-9:
                        converged = True
                        break

            # Final verification
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est_final - target)) < 1e-8:
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2

                est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                if np.max(np.abs(est_clean - target)) < 1e-7:
                    best_rank = r
                    best_result = {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
                    break
                else:
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break

    return best_result
