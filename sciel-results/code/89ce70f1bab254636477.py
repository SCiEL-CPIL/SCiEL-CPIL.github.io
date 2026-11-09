"""Rewritten program: search for a rank-minimal exact tensor decomposition of 2x5 by 5x5 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Return the matmul tensor T[i, j, k] = 1 if output (i,k) uses A[i,j]*B[j,k].
    Convention: U indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    Tensor T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i, else 0."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _decompose(T, U, V, W):
    """Reconstruct tensor from factor matrices (a,b,c) = sum_r U[a,r]V[b,r]W[c,r]."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _verify(T, U, V, W, tol=1e-9):
    R = _decompose(T, U, V, W)
    return bool(np.allclose(R, T, atol=tol, rtol=0.0))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _mat_to_halfint(x):
    """Convert a float to the nearest half-integer (multiple of 0.5)."""
    return round(x * 2.0) / 2.0


def _round_factors(U, V, W):
    """Round all entries to nearest half-integer."""
    U = np.array([[_mat_to_halfint(v) for v in row] for row in U])
    V = np.array([[_mat_to_halfint(v) for v in row] for row in V])
    W = np.array([[_mat_to_halfint(v) for v in row] for row in W])
    return U, V, W


def _try_reduce(T, U, V, W, max_steps, deadline):
    """Attempt to reduce rank by removing one factor column at a time via least-squares
    refit. Returns (U,V,W,success_count)."""
    n, m, p = T.shape
    a, b, c = n * m, m * p, p * n
    Tflat = T.reshape(a * b, c)
    rank = U.shape[1]
    reductions = 0
    # Work on float copies
    Uf = U.astype(np.float64).copy()
    Vf = V.astype(np.float64).copy()
    Wf = W.astype(np.float64).copy()
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        if rank <= 0:
            break
        # Try removing each column; keep the one that best preserves the tensor.
        best_err = None
        best_col = -1
        best_UVW = None
        for col in range(rank):
            if time.time() >= deadline:
                break
            # Build the residual tensor with column `col` zeroed, then refit.
            Uc = Uf.copy()
            Vc = Vf.copy()
            Wc = Wf.copy()
            Uc[:, col] = 0.0
            Vc[:, col] = 0.0
            Wc[:, col] = 0.0
            # Residual = T - decompose(Uc,Vc,Wc)
            R = np.einsum('ar,br,cr->abc', Uc, Vc, Wc).reshape(a * b, c)
            resid = Tflat - R
            # Refit column `col` as a rank-1 term: find u,v,w minimizing ||resid - u v w^T||
            # Use SVD on the residual matrix (a*b) x c.
            try:
                Us, s, Vt = np.linalg.svd(resid, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            if s.size == 0 or s[0] <= 1e-12:
                # residual ~ 0, column can be dropped for free
                new_err = 0.0
            else:
                # best rank-1 approx error = sqrt(sum of s[1:]^2)
                new_err = float(np.sqrt(np.sum(s[1:] ** 2)))
                u = Us[:, 0] * np.sqrt(s[0])
                w = Vt[0, :] * np.sqrt(s[0])
                # v is the middle index: resid = u (a*b) x v (b) x w (c)?
                # Actually resid is (a*b) x c. To factor as u[a] v[b] w[c],
                # reshape u to (a,b) and solve for v.
                # resid ≈ (u ⊗ v) w^T  =>  for each c, resid[:,c] ≈ w[c] * (u⊗v)
                # u⊗v = u reshaped (a,b) times v -> (a*b). So resid[:,c] ≈ w[c]* (Uvec ⊗ Vvec)
                # We have u from SVD as the (a*b) vector; need to split into (a,b) and (b).
                # This is a bilinear factorization; use the structure: u = Uvec ⊗ Vvec.
                # Recover via: Uvec = u / ||u||, then Vvec via least squares.
                # But u from SVD is already the left singular vector (a*b).
                # We need u = Uvec ⊗ Vvec. Solve: for the chosen scaling, 
                # let Uvec be (a,b) and Vvec be (b): u[a*b] = sum? No, outer: u = Uvec.flatten() ⊙? 
                # Actually (Uvec ⊗ Vvec) where Uvec is (a,b), Vvec is (b): 
                # u = np.einsum('ab,b->ab', Uvec, Vvec).flatten()
                # This is a matrix (a,b) times diagonal(Vvec). So u reshaped (a,b) = Uvec @ diag(Vvec).
                # Given u (a*b), factor u.reshape(a,b) = X @ diag(y).
                # Take X = u.reshape(a,b), and we need y such that X = X0 @ diag(y) for some X0.
                # Simplest: set X0 = X (absorb), y = 1? That doesn't reduce. 
                # We need a genuine split. Use: choose X0 = X / row-norms? 
                # Better: factor X = X0 @ diag(y) by setting y = row-wise? 
                # Let y = 1/b * ones? Not exact.
                # Use SVD of X: X = Xs S Xt. Then set X0 = Xs S[:, :1], y = Xt[0] * s[0]? 
                # X = X0 @ diag(y) with X0 (a,b), y (b). 
                # Let X0 = X, y = 1? Then X = X @ I, valid but y all ones.
                # That's fine! v = ones(b). Then u⊗v = u, w from SVD.
                # So the rank-1 term is u (a*b) x w (c) with v = ones.
                # But v must be (b,), and the decomposition needs V indexed by b.
                # Set Vvec = ones(b). Then (Uvec ⊗ Vvec) = Uvec (a*b) = u. Good.
                u_reshaped = u.reshape(a, b)
                # Vvec = ones, Uvec = u_reshaped
            # Build new column: Uvec (a,b) -> but U is (a, rank) per a? No!
            # U has shape (a, rank) where a = n*m. The column r of U is indexed by a.
            # Wait: U[a, r] where a = i*m+j. So U column is a vector of length a = n*m.
            # But the tensor term is U[a,r] V[b,r] W[c,r]. So U column is length a, V column length b, W column length c.
            # The rank-1 tensor is outer(Ucol, Vcol, Wcol) = einsum. So Ucol is (a,), Vcol (b,), Wcol (c).
            # resid = Tflat - R is (a*b, c). We need to express as Ucol[a] Vcol[b] Wcol[c].
            # resid[a*b, c] = Ucol[a] Vcol[b] Wcol[c].
            # For each c: resid[:, c] = Wcol[c] * (Ucol ⊗ Vcol).
            # So (Ucol ⊗ Vcol) = resid[:,0] / Wcol[0] (if Wcol[0]!=0).
            # From SVD: resid = Umat diag(s) Vt. Rank-1: u = Umat[:,0]*s[0], w = Vt[0]*s[0].
            # u is (a*b,). Factor u = Ucol ⊗ Vcol where Ucol (a,), Vcol (b,):
            # u[a*b] = Ucol[a] Vcol[b]. Reshape u to (a,b): u2 = u.reshape(a,b).
            # u2[a,b] = Ucol[a] Vcol[b]. Factor u2 = Ucol[:,None] * Vcol[None,:].
            # Use SVD of u2: u2 = us s vt. Rank-1: Ucol = us[:,0]*s[0], Vcol = vt[0]*s[0].
            u2 = u.reshape(a, b)
            try:
                us2, s2, vt2 = np.linalg.svd(u2, full_matrices=False)
                Ucol = us2[:, 0] * s2[0]
                Vcol = vt2[0, :] * s2[0]
            except np.linalg.LinAlgError:
                continue
            # w = Vt[0] * s[0] from first SVD
            Wcol = Vt[0, :] * s[0]
            # Assign
            Uc[:, col] = Ucol
            Vc[:, col] = Vcol
            Wc[:, col] = Wcol
            # Compute actual error
            R2 = np.einsum('ar,br,cr->abc', Uc, Vc, Wc).reshape(a * b, c)
            err = float(np.sqrt(np.sum((Tflat - R2) ** 2)))
            if best_err is None or err < best_err:
                best_err = err
                best_col = col
                best_UVW = (Uc, Vc, Wc)
        if best_col < 0 or best_err is None:
            break
        if best_err > 1e-8:
            # Can't remove this column exactly; try a different strategy or stop.
            break
        Uf, Vf, Wf = best_UVW
        # Drop the column
        mask = np.ones(rank, dtype=bool)
        mask[best_col] = False
        Uf = Uf[:, mask]
        Vf = Vf[:, mask]
        Wf = Wf[:, mask]
        rank = Uf.shape[1]
        reductions += 1
    return Uf, Vf, Wf, reductions


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    T = _build_tensor(n, m, p)
    a, b, c = n * m, m * p, p * n

    # Start from schoolbook rank 50
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_UVW = (U.copy(), V.copy(), W.copy())

    # Try rank reduction via column elimination
    U2, V2, W2, red = _try_reduce(T, U, V, W, max_steps=50, deadline=deadline)
    if red > 0 and _verify(T, U2, V2, W2):
        best_rank = U2.shape[1]
        best_UVW = (U2.copy(), V2.copy(), W2.copy())

    # Randomized local search: perturb and re-eliminate
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() >= deadline:
            break
        # Start from current best or schoolbook with random init
        if trial % 2 == 0:
            U0, V0, W0 = best_UVW
        else:
            U0, V0, W0 = _schoolbook(n, m, p)
        # Add small random perturbation to create non-trivial structure
        U0 = U0 + rng.gauss(0, 0.01) * np.ones_like(U0)
        V0 = V0 + rng.gauss(0, 0.01) * np.ones_like(V0)
        W0 = W0 + rng.gauss(0, 0.01) * np.ones_like(W0)
        # Try to reduce
        U1, V1, W1, red1 = _try_reduce(T, U0, V0, W0, max_steps=30, deadline=deadline)
        if red1 > 0:
            # Check if it's exact after rounding to half-integers
            Ur, Vr, Wr = _round_factors(U1, V1, W1)
            if _verify(T, Ur, Vr, Wr):
                if U1.shape[1] < best_rank:
                    best_rank = U1.shape[1]
                    best_UVW = (Ur, Vr, Wr)
            elif _verify(T, U1, V1, W1):
                # exact in float, try to snap to half-integers
                Ur, Vr, Wr = _round_factors(U1, V1, W1)
                if _verify(T, Ur, Vr, Wr):
                    if U1.shape[1] < best_rank:
                        best_rank = U1.shape[1]
                        best_UVW = (Ur, Vr, Wr)

    Ubest, Vbest, Wbest = best_UVW
    # Final: convert to lists of lists of half-integers
    Uout = [[_mat_to_halfint(v) for v in row] for row in Ubest]
    Vout = [[_mat_to_halfint(v) for v in row] for row in Vbest]
    Wout = [[_mat_to_halfint(v) for v in row] for row in Wbest]
    return {"U": Uout, "V": Vout, "W": Wout}
# EVOLVE-BLOCK-END
