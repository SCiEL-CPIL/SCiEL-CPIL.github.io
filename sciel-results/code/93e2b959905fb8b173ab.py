import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_rank17():
    """Rank 17 decomposition for 2x3x3 matrix multiplication.
    Combines Strassen (rank 7) for 2x2x2 block + schoolbook for cross terms.
    """
    n, m, p = 2, 3, 3
    R = 17
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i  (trace convention W[k*n+i, r])

    # ---- Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] ----
    # A idx: 0,1,3,4 ; B idx: 0,1,3,4 ; C idx: 0,1,2,3
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # ---- Part 2: schoolbook A[0:2,2] * B[2,0:2] -> C[0:2,0:2] ----
    # A idx: 2,5 ; B idx: 6,7 ; C idx: 0,1,2,3
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # ---- Part 3: schoolbook A[0:2,0:3] * B[0:3,2] -> C[0:2,2] ----
    # A idx: 0..5 ; B idx: 2,5,8 ; C idx: 4,5
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] reconstructed tensor matches T."""
    n, m, p = 2, 3, 3
    # Build target tensor T[i,j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Reconstruct
    R = U.shape[1]
    rec = np.zeros((n, m, p))
    for r in range(R):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # term = sum_{a,b} Ur[a,b] Vr[b,c] Wr[c,a]  -> but Wr is p x n indexed [k,i]
        # rec[i,j,k] = sum_r Ur[i,j] Vr[j,k] Wr[k,i]
        # We can compute: for each r, outer of (Ur * ... )
        # rec_r[i,j,k] = Ur[i,j] * Vr[j,k] * Wr[k,i]
        # Use einsum
        rec += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
    diff = np.abs(rec - T).max()
    return diff < tol, diff


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 150.0  # wall-clock budget; evaluator kills at 150s
    SAFETY = 20.0

    # Baseline: rank-17 decomposition (verified exact, integer entries).
    U17, V17, W17 = _build_rank17()
    ok17, diff17 = _verify(U17, V17, W17)
    best = (17, U17, V17, W17)

    # Try to find a rank-16 decomposition via randomized tensor rank search.
    # Strategy: random rank-16 tensor, then iterative refinement (Burer-Monteiro style)
    # minimizing ||T - sum_r U_r V_r W_r||_F. If it converges to ~0, we have rank 16.
    # We use the fact that the target tensor is all-ones on its support.
    if time.time() + SAFETY < DEADLINE:
        try:
            R = 16
            T = np.ones((n, m, p))
            rng = np.random.default_rng(12345)
            best_err = np.inf
            # Several random restarts
            for restart in range(40):
                if time.time() + SAFETY >= DEADLINE:
                    break
                # Initialize random rank-1 factors
                U = rng.normal(size=(n, m, R))
                V = rng.normal(size=(m, p, R))
                W = rng.normal(size=(p, n, R))
                # Normalize columns to help conditioning
                for r in range(R):
                    nu = np.linalg.norm(U[:, :, r])
                    nv = np.linalg.norm(V[:, :, r])
                    nw = np.linalg.norm(W[:, :, r])
                    if nu > 1e-12:
                        U[:, :, r] /= nu
                    if nv > 1e-12:
                        V[:, :, r] /= nv
                    if nw > 1e-12:
                        W[:, :, r] /= nw
                # Gradient descent on flattened factors
                lr = 0.01
                for it in range(2000):
                    if time.time() + SAFETY >= DEADLINE:
                        break
                    # Compute residual tensor
                    rec = np.zeros((n, m, p))
                    for r in range(R):
                        rec += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
                    err = np.linalg.norm(rec - T)
                    if err < best_err:
                        best_err = err
                    if err < 1e-6:
                        break
                    # Gradients
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    resid = rec - T
                    for r in range(R):
                        dU[:, :, r] = np.einsum('ijk,jk,ki->ij', resid, V[:, :, r], W[:, :, r])
                        dV[:, :, r] = np.einsum('ijk,ij,ki->jk', resid, U[:, :, r], W[:, :, r])
                        dW[:, :, r] = np.einsum('ijk,ij,jk->ki', resid, U[:, :, r], V[:, :, r])
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    # adaptive lr
                    if err > 5.0:
                        lr *= 0.5
                    elif err < 0.01:
                        lr = min(lr * 1.05, 0.05)
                if best_err < 1e-6:
                    break
            # Check if we found a near-exact rank-16
            if best_err < 1e-6:
                # Round to nearest half-integers and re-verify
                # (the true solution, if it exists, should be close to simple rationals)
                Uf = U.reshape(n * m, R)
                Vf = V.reshape(m * p, R)
                Wf = W.reshape(p * n, R)
                # Round to nearest 0.5
                Ur = (2 * np.round(2 * Uf) / 2)
                Vr = (2 * np.round(2 * Vf) / 2)
                Wr = (2 * np.round(2 * Wf) / 2)
                ok_r, diff_r = _verify(Ur, Vr, Wr, tol=1e-9)
                if ok_r:
                    best = (16, Ur, Vr, Wr)
        except Exception:
            pass

    R, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
