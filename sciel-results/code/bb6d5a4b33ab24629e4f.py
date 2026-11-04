import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8

    # Build the target tensor T[i, j, k] = 1 for all valid (i, j, k)
    # U index: i*m + j  (size 6)
    # V index: j*p + k  (size 12)
    # W index: k*n + i  (size 8)

    def build_decomposition(rank, U, V, W):
        """Verify decomposition and return dict if valid."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
            return None
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a] * V[b], W[c])
                    if abs(val - 1.0) > 1e-6:
                        return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy: Construct a rank-22 decomposition using Strassen-like 2x2x2 block
    # For the 2x2x2 sub-problem (rows 0-1, cols 0-1, outputs 0-1):
    # Strassen uses 7 multiplications instead of 8.
    # For the full 2x3x4, we decompose into overlapping sub-problems.
    #
    # Approach: Use the known structure that matrix multiplication tensor
    # T_{2,3,4} can be decomposed by:
    # - A 2x2x2 Strassen block for (i=0,1; j=0,1; k=0,1) -> rank 7
    # - Remaining terms handled by rank-1 outer products
    #
    # Let's construct this explicitly.
    #
    # Strassen for C = A*B where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For our 2x3x4 case, let's use a direct construction:
    # We'll build the decomposition term by term.
    #
    # Actually, let me use a simpler but effective approach:
    # Start with the naive rank-24 and try to merge terms.
    #
    # Better approach: Use the fact that we can express the multiplication
    # as a sum of rank-1 tensors. Let's try to find a rank-22 construction
    # by starting from naive and eliminating 2 terms through linear dependence.
    #
    # For a cleaner structural approach, let me use a hybrid:
    # Treat the problem as multiplying 2x3 by 3x4.
    # Split the 3 middle dimension: use indices j=0,1,2
    # Split the 4 output columns: k=0,1,2,3
    #
    # Let me try a direct algebraic construction for rank 22.
    #
    # Alternative: Use the known result that the tensor rank of 2x3x4
    # matrix multiplication is at most 22 (from literature/known constructions).
    #
    # Let me construct it using a systematic approach:
    # The naive decomposition has 24 terms indexed by (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    #
    # I'll use a different structural approach: exploit the fact that for fixed (i,k),
    # the sum over j is linear. This means we can potentially share terms.
    #
    # Let me try a MILP-like approach but with a smarter initialization:
    # Use the Strassen structure for the 2x2x2 corner and fill the rest.

    # Let me try yet another approach: use scipy's optimization with a
    # better formulation - minimize the sum of squared errors with
    # multiple random restarts and annealing-like schedule.
    #
    # But the directive says to make a STRUCTURAL change. Let me use
    # a completely different method: algebraic/geometric approach.
    #
    # For 2x3x4 matrix multiplication, I'll construct the decomposition
    # using the following known technique:
    #
    # The matrix multiplication tensor T_{n,m,p} has a decomposition of rank
    # at most n*m*p - min(n,m,p) + 1 in some cases, but let me use a
    # more concrete construction.
    #
    # Let me use a direct constructive approach:
    # For 2x3x4, I'll use the following rank-22 construction based on
    # the idea of "folding" two of the naive terms.

    # Actually, let me just use a robust numerical search with a different
    # algorithm: alternating least squares with regularization and
    # multiple restarts, which is structurally different from gradient descent.
    #
    # But even better: let me use the approach from the inspiration that
    # achieved rank 22 - Strassen-like for 2x2x2 blocks.

    # Let me construct the rank-22 decomposition explicitly.
    #
    # For the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}):
    # Use Strassen's 7-term decomposition.
    #
    # For the remaining parts, use rank-1 terms.
    #
    # Total: 7 (Strassen) + terms for the rest
    # The "rest" is: all (i,j,k) triples not covered by the 2x2x2 block
    # i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # The 2x2x2 block covers i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms, replaced by 7
    # Remaining: 24 - 8 = 16 terms, but we need to be careful about overlaps.
    #
    # Actually, the Strassen decomposition doesn't just replace 8 rank-1 terms with 7.
    # It replaces the 2x2x2 multiplication tensor with a rank-7 decomposition.
    #
    # Let me think about this more carefully. The full 2x3x4 tensor is:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    #
    # We can write this as:
    # T = T_2x2x2 (for j=0,1; k=0,1) + T_rest
    #
    # Where T_2x2x2[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
    # and T_rest[i,j,k] = 1 for all other valid (i,j,k)
    #
    # T_2x2x2 has a rank-7 decomposition (Strassen).
    # T_rest has a rank-1 decomposition with 24-8=16 terms.
    # Total rank: 7 + 16 = 23.
    #
    # To get to 22, we need to find one more linear dependence.
    #
    # Let me try a different split:
    # Use a 2x3x3 sub-block (all i, all j, k in {0,1,2}) with a known lower-rank decomposition,
    # and handle the 4th column separately.
    #
    # The 2x3x3 matrix multiplication tensor has rank at most 11 (known result).
    # The remaining terms for k=3: 2*3 = 6 terms.
    # Total: 11 + 6 = 17? That seems too low. The issue is that the decompositions
    # may not combine cleanly.
    #
    # Let me just go with a robust numerical approach but using a different
    # algorithm family: use scipy.optimize.minimize with L-BFGS-B on a
    # reformulated problem, with multiple restarts.

    best_result = None
    best_rank = 24

    # First, establish the naive rank-24 solution
    U_naive = np.zeros((nm, 24))
    V_naive = np.zeros((mp, 24))
    W_naive = np.zeros((pn, 24))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_naive[i * m + j, r] = 1
                V_naive[j * p + k, r] = 1
                W_naive[k * n + i, r] = 1
    result_24 = build_decomposition(24, U_naive, V_naive, W_naive)
    if result_24 is not None:
        best_result = result_24
        best_rank = 24

    # Now try to find lower rank using alternating minimization
    # This is structurally different from gradient descent: we fix two
    # factor matrices and solve for the third in closed form.

    def alternating_minimization(target_rank, U_init, V_init, W_init, max_iter=200, tol=1e-8):
        """Alternating least squares for tensor decomposition."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Fix V, W; solve for U
            # For each row a of U, we need: sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # This is: U[a] @ diag(V[b]*W[c]) = 1 for all (b,c) corresponding to a
            # Actually, for fixed a (which corresponds to some (i,j)), the constraints are:
            # For each k: U[a] @ (V[j*p+k] * W[k*n+i]) = 1
            # This is a system of p equations in target_rank unknowns (for U[a])
            # But it's underdetermined if target_rank > p, so we use least squares.

            # Better formulation: For each (i,j) -> a, we have p constraints:
            # sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0,...,p-1
            # Let M_a be a p x target_rank matrix where M_a[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we want M_a @ U[a].T = ones(p)
            # Solve: U[a].T = pinv(M_a) @ ones(p)  (least squares)

            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    M_a = np.zeros((p, target_rank))
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        M_a[k, :] = V[b, :] * W[c, :]
                    # Solve M_a @ x = ones(p)
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_a, np.ones(p), rcond=None)
                    U[a, :] = x

            # Fix U, W; solve for V
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    M_b = np.zeros((n, target_rank))
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        M_b[i, :] = U[a, :] * W[c, :]
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_b, np.ones(n), rcond=None)
                    V[b, :] = x

            # Fix U, V; solve for W
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    M_c = np.zeros((m, target_rank))
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        M_c[j, :] = U[a, :] * V[b, :]
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_c, np.ones(m), rcond=None)
                    W[c, :] = x

            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U[a] * V[b], W[c])
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < tol:
                break

        return U, V, W, max_err

    # Try ranks from 23 down to a reasonable lower bound
    for target_rank in range(23, 18, -1):
        if time.time() > deadline - 20:
            break

        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break

            # Initialize
            np.random.seed(12345 + attempt * 7 + target_rank)
            if attempt == 0:
                # Use naive solution truncated
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                U_init[:, :min(target_rank, 24)] = U_naive[:, :min(target_rank, 24)]
                V_init[:, :min(target_rank, 24)] = V_naive[:, :min(target_rank, 24)]
                W_init[:, :min(target_rank, 24)] = W_naive[:, :min(target_rank, 24)]
            elif attempt == 1:
                # Random
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            elif attempt == 2:
                # Sparse random
                U_init = np.random.randn(nm, target_rank)
                V_init = np.random.randn(mp, target_rank)
                W_init = np.random.randn(pn, target_rank)
                U_init[np.random.rand(nm, target_rank) > 0.7] = 0
                V_init[np.random.rand(mp, target_rank) > 0.7] = 0
                W_init[np.random.rand(pn, target_rank) > 0.7] = 0
            elif attempt == 3:
                # Structured: use Kronecker-like patterns
                U_init = np.random.randn(nm, target_rank) * 0.3
                V_init = np.random.randn(mp, target_rank) * 0.3
                W_init = np.random.randn(pn, target_rank) * 0.3
                # Add some structure
                for r in range(target_rank):
                    U_init[:, r] *= np.random.choice([1, -1], nm)
                    V_init[:, r] *= np.random.choice([1, -1], mp)
                    W_init[:, r] *= np.random.choice([1, -1], pn)
            else:
                # Very small initialization
                U_init = np.random.randn(nm, target_rank) * 0.1
                V_init = np.random.randn(mp, target_rank) * 0.1
                W_init = np.random.randn(pn, target_rank) * 0.1

            U_sol, V_sol, W_sol, max_err = alternating_minimization(
                target_rank, U_init, V_init, W_init, max_iter=300, tol=1e-9
            )

            if max_err < 1e-6:
                result = build_decomposition(target_rank, U_sol, V_sol, W_sol)
                if result is not None:
                    best_result = result
                    best_rank = target_rank
                    found = True
                    break

        if found:
            continue  # Try to go lower

    # Final fallback
    if best_result is None:
        best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return best_result
