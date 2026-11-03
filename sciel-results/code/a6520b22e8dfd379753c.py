"""Rewritten program: search for a rank-minimal exact tensor decomposition
for 3x3 * 3x7 matrix multiplication, starting from schoolbook and reducing rank
by removing linearly dependent columns of the factor matrices.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 150.0


def _tensors_equal(A, B, tol=1e-8):
    """Check whether two flattened tensors are equal within tolerance."""
    return np.allclose(A, B, atol=tol, rtol=0.0)


def _verify(U, V, W, n=3, m=3, p=7, tol=1e-8):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] equals the matmul tensor."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We compare against the canonical matmul tensor:
    # T_canon[i,j,k] = 1 if ... actually the matmul tensor is the structure
    # tensor: M_{(i,k),(j)} = delta... Let's build it properly.
    # The matrix multiplication tensor T has indices (i, j, k) with
    # T[i,j,k] = 1 (it's the structure tensor where C[i,k] = sum_j A[i,j] B[j,k])
    # In trace convention W[k*n+i, r], U[i*m+j, r], V[j*p+k, r].
    # The decomposition gives: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # should equal 1 for all (i,j,k)?? No.
    # Actually the matmul tensor T_{i,j,k} = 1 for all i,j,k is wrong.
    # The correct structure: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right either. The tensor M has M[i,j,k] = 1 meaning
    # the product A[i,j]*B[j,k] feeds into C[i,k]. So yes, M[i,j,k] = 1 for all i,j,k.
    # Let me verify: the decomposition must satisfy
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # That's the standard formulation. Let's check with schoolbook:
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for the single column c=(i,j,k).
    # So for a given (i,j,k), only the column c=(i,j,k) contributes 1*1*1=1. Correct.
    T = np.zeros((n, m, p))
    for r in range(rank):
        Uc = U[:, r]
        Vc = V[:, r]
        Wc = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return _tensors_equal(T, target, tol)


def _schoolbook(n=3, m=3, p=7):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_reduce(U, V, W, n=3, m=3, p=7, max_rounds=5, tol=1e-8):
    """Try to reduce rank by removing columns that are linear combinations
    of others (i.e., find a minimal subset of columns that still spans the
    same tensor). We greedily remove columns whose removal still yields a
    valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(max_rounds):
        removed = False
        for c in range(rank):
            # Try removing column c
            mask = [i for i in range(rank) if i != c]
            U2 = U[:, mask]
            V2 = V[:, mask]
            W2 = W[:, mask]
            if _verify(U2, V2, W2, n, m, p, tol):
                U, V, W = U2, V2, W2
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W


def _greedy_random_search(U0, V0, W0, n=3, m=3, p=7, budget=10.0, tol=1e-8):
    """Randomly perturb columns and try to find lower-rank representations.
    We add random linear combinations of existing columns to new columns and
    see if we can drop others."""
    U = np.array(U0, dtype=float)
    V = np.array(V0, dtype=float)
    W = np.array(W0, dtype=float)
    rank = U.shape[1]
    best = (U, V, W)
    best_rank = rank
    start = time.time()
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() - start < budget and attempts < 200:
        attempts += 1
        # Pick a random column to try to eliminate by expressing it as a combo
        c = rng.integers(0, rank)
        # Try to find coefficients for other columns
        # This is complex; instead try random small perturbations
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        # Zero out a random column and see if we can compensate
        # Actually, let's try a different approach: random linear combinations
        # Create a new column as a random combo and replace an existing one
        new_col = np.zeros(rank)
        idx = rng.choice(rank, size=3, replace=False)
        coeffs = rng.standard_normal(3)
        for i, ci in enumerate(idx):
            new_col += coeffs[i] * U[:, ci]
        # Replace column c with new_col
        U2[:, c] = new_col
        # Similarly for V and W with same coefficients
        V2[:, c] = np.zeros(rank)
        W2[:, c] = np.zeros(rank)
        for i, ci in enumerate(idx):
            V2[:, c] += coeffs[i] * V[:, ci]
            W2[:, c] += coeffs[i] * W[:, ci]
        if _verify(U2, V2, W2, n, m, p, tol):
            # Now try to reduce
            U3, V3, W3 = _try_reduce(U2, V2, W2, n, m, p, max_rounds=2, tol=tol)
            if U3.shape[1] < best_rank:
                best = (U3, V3, W3)
                best_rank = U3.shape[1]
                U, V, W = U3, V3, W3
                rank = best_rank
    return best


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    # Try to reduce
    U, V, W = _try_reduce(U, V, W, n, m, p)
    # Random search if time permits
    elapsed = time.time() - start
    if elapsed < DEADLINE - 20:
        budget = min(30.0, DEADLINE - elapsed - 10)
        U2, V2, W2 = _greedy_random_search(U, V, W, n, m, p, budget=budget)
        if U2.shape[1] < U.shape[1]:
            U, V, W = U2, V2, W2
    # Final safety check
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    # Convert to lists of integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
