"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search.

Strategy:
- The matrix-multiplication tensor T[i,j,k] for (3,4,5) is all-ones (120 entries).
- The schoolbook rank is 60.
- We try to find a lower-rank exact decomposition with half-integer entries.
- Approach: structured block embeddings (Strassen-style) + exhaustive search
  over small half-integer value sets for small ranks, plus random search
  with structured initialization.
"""
import time
import numpy as np
from itertools import product

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110  # safety margin under 150s kill


def _time_left():
    return DEADLINE - time.time()


def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)


def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0


def _compute_T(U, V, W, rank, n=N, m=M, p=P):
    """Compute the tensor T from factor matrices."""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)


def _max_err(T):
    """Maximum absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))


def _exhaustive_search_small(rank, values, n=N, m=M, p=P, time_budget=None):
    """
    Exhaustive search over a small set of half-integer values for each entry.
    This is only feasible for very small ranks and value sets.
    For rank r, we need to fill U (n*m*r), V (m*p*r), W (p*n*r) entries.
    With a value set of size s, total combinations = s^(n*m*r + m*p*r + p*n*r).
    This is only feasible for rank <= 3 or 4 with small s.
    """
    if time_budget is not None and time.time() > time_budget:
        return None

    nm, mp, pn = n * m, m * p, p * n
    total_entries = nm * rank + mp * rank + pn * rank

    # Only feasible for very small total_entries
    if total_entries > 30:
        return None

    val_list = list(values)
    s = len(val_list)

    # For each combination of values, check if it works
    # This is s^(total_entries) which is only feasible for small total_entries
    count = 0
    for combo in product(val_list, repeat=total_entries):
        if time_budget is not None and time.time() > time_budget:
            return None
        count += 1
        idx = 0
        U = np.zeros((nm, rank))
        for i in range(nm):
            for r in range(rank):
                U[i, r] = combo[idx]
                idx += 1
        V = np.zeros((mp, rank))
        for i in range(mp):
            for r in range(rank):
                V[i, r] = combo[idx]
                idx += 1
        W = np.zeros((pn, rank))
        for i in range(pn):
            for r in range(rank):
                W[i, r] = combo[idx]
                idx += 1

        T = _compute_T(U, V, W, rank)
        if _max_err(T) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _structured_strassen_init(rank, n=N, m=M, p=P):
    """
    Initialize factor matrices using Strassen-style block structure.
    For 3x4x5, we can embed into 4x4x4 blocks and use Strassen rank-7 structure.
    """
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6

    # For 3x4x5, we can try a block decomposition:
    # Split A (3x4) into blocks, B (4x5) into blocks
    # One approach: use 2x2 blocks for the 4x4 middle dimension

    # Let's try a different structured approach:
    # Use the fact that the tensor is all-ones and try to find
    # a low-rank decomposition directly.

    # Random initialization with some structure
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)

    # Normalize
    for r in range(rank):
        nU = np.linalg.norm(U[:, r])
        nV = np.linalg.norm(V[:, r])
        nW = np.linalg.norm(W[:, r])
        if nU < 1e-12 or nV < 1e-12 or nW < 1e-12:
            continue
        U[:, r] /= nU
        V[:, r] /= nV
        W[:, r] /= nW
        scale = (nU * nV * nW) ** (1.0 / 3.0)
        U[:, r] *= scale
        V[:, r] *= scale
        W[:, r] *= scale

    return U, V, W


def _try_rank_gradient(rank, seed, time_budget, n=N, m=M, p=P):
    """
    Try to find an exact half-integer decomposition of given rank
    using gradient-based optimization with half-integer projection.
    """
    np.random.seed(seed)

    U, V, W = _structured_strassen_init(rank, n, m, p)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    best_err = np.inf
    best = None

    # Adam-like optimizer
    lr = 0.05
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    t = 0

    steps = 3000
    for it in range(steps):
        if time.time() > time_budget:
            break

        t += 1
        b1 = 1 - beta1 ** t
        b2 = 1 - beta2 ** t

        # Compute T and gradients
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err_tensor = T - 1.0

        # Gradient w.r.t. U: dL/dU[i,j,r] = sum_{k} err[i,j,k] * V[j,k,r] * W[k,i,r]
        gU3 = np.einsum('ijk,jkr,kir->ijr', err_tensor, V3, W3)
        gU = gU3.reshape(n * m, rank)

        # Gradient w.r.t. V: dL/dV[j,k,r] = sum_{i} err[i,j,k] * U[i,j,r] * W[k,i,r]
        gV3 = np.einsum('ijk,ijr,kir->jkr', err_tensor, U3, W3)
        gV = gV3.reshape(m * p, rank)

        # Gradient w.r.t. W: dL/dW[k,i,r] = sum_{j} err[i,j,k] * U[i,j,r] * V[j,k,r]
        gW3 = np.einsum('ijk,ijr,jkr->kir', err_tensor, U3, V3)
        gW = gW3.reshape(p * n, rank)

        # Adam update
        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * gU ** 2
        U -= lr * mU / b1 / (np.sqrt(vU / b2) + eps)

        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * gV ** 2
        V -= lr * mV / b1 / (np.sqrt(vV / b2) + eps)

        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW ** 2
        W -= lr * mW / b1 / (np.sqrt(vW / b2) + eps)

        # Project to half-integers and check
        Ur = _to_halfint(U)
        Vr = _to_halfint(V)
        Wr = _to_halfint(W)
        Tr = _compute_T(Ur, Vr, Wr, rank)
        err = _max_err(Tr)
        if err < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        if err < best_err:
            best_err = err
            best = (Ur.copy(), Vr.copy(), Wr.copy())

    # Final check on best
    if best is not None:
        Ur, Vr, Wr = best
        Tr = _compute_T(Ur, Vr, Wr, rank)
        if _max_err(Tr) < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None


def _try_rank_random_walk(rank, seed, time_budget, n=N, m=M, p=P):
    """
    Random walk search with half-integer projection.
    """
    np.random.seed(seed)

    U, V, W = _structured_strassen_init(rank, n, m, p)
    best_err = np.inf
    best = None

    steps = 2000
    for it in range(steps):
        if time.time() > time_budget:
            break

        # Project to half-integers and check
        Ur = _to_halfint(U)
        Vr = _to_halfint(V)
        Wr = _to_halfint(W)
        Tr = _compute_T(Ur, Vr, Wr, rank)
        err = _max_err(Tr)
        if err < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        if err < best_err:
            best_err = err
            best = (Ur.copy(), Vr.copy(), Wr.copy())

        # Perturbation
        choice = np.random.randint(3)
        step_size = 0.5  # half-integer steps
        if choice == 0:
            idx = (np.random.randint(n * m), np.random.randint(rank))
            U[idx] += np.random.choice([-step_size, step_size])
        elif choice == 1:
            idx = (np.random.randint(m * p), np.random.randint(rank))
            V[idx] += np.random.choice([-step_size, step_size])
        else:
            idx = (np.random.randint(p * n), np.random.randint(rank))
            W[idx] += np.random.choice([-step_size, step_size])

    if best is not None:
        Ur, Vr, Wr = best
        Tr = _compute_T(Ur, Vr, Wr, rank)
        if _max_err(Tr) < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None


def _try_rank_direct(rank, seed, time_budget, n=N, m=M, p=P):
    """
    Direct search: try to find a decomposition where each tensor entry
    is a sum of rank products. Use a constraint-satisfaction approach.
    """
    np.random.seed(seed)

    # The tensor is all-ones. We need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Strategy: try to use a small set of "basis" patterns.
    # For rank r, we have r "terms" and each term contributes a product.

    # Initialize with random half-integers
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Try to set up a structured solution
    # For the all-ones tensor, one approach is to use
    # U[i*m+j, r] = a_r, V[j*p+k, r] = b_r, W[k*n+i, r] = c_r
    # with sum_r a_r * b_r * c_r = 1
    # This gives a rank-1-like structure but with r terms.

    # More generally, we can have:
    # U[i*m+j, r] depends on i, j
    # V[j*p+k, r] depends on j, k
    # W[k*n+i, r] depends on k, i

    # Let's try a factorized approach:
    # U[i*m+j, r] = u_i * v_j * w_r  (not quite, this is too restrictive)

    # Better: use a small number of distinct patterns.
    # For each rank r, assign a "type" to each (i,j), (j,k), (k,i) pair.

    # Simple approach: try random half-integer assignments and check.
    # This is essentially the random walk, but let's try a more structured init.

    # Initialize with small values
    for r in range(rank):
        U[:, r] = np.random.choice([-1, -0.5, 0.5, 1], size=n * m)
        V[:, r] = np.random.choice([-1, -0.5, 0.5, 1], size=m * p)
        W[:, r] = np.random.choice([-1, -0.5, 0.5, 1], size=p * n)

    best_err = np.inf
    best = None

    steps = 1000
    for it in range(steps):
        if time.time() > time_budget:
            break

        Tr = _compute_T(U, V, W, rank)
        err = _max_err(Tr)
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())

        # Perturb a few entries
        for _ in range(3):
            choice = np.random.randint(3)
            if choice == 0:
                idx = (np.random.randint(n * m), np.random.randint(rank))
                U[idx] = np.random.choice([-1, -0.5, 0.5, 1])
            elif choice == 1:
                idx = (np.random.randint(m * p), np.random.randint(rank))
                V[idx] = np.random.choice([-1, -0.5, 0.5, 1])
            else:
                idx = (np.random.randint(p * n), np.random.randint(rank))
                W[idx] = np.random.choice([-1, -0.5, 0.5, 1])

    if best is not None:
        Ur, Vr, Wr = best
        Tr = _compute_T(Ur, Vr, Wr, rank)
        if _max_err(Tr) < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None


def _schoolbook():
    """Return the schoolbook decomposition (rank 60)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook()
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = SCHOOLBOOK_RANK

    # Lower bound: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Actually, the lower bound for matrix multiplication is more nuanced.
    # For n x m x p, rank >= max(n*m, m*p, p*n) is a known lower bound.
    # So rank >= 20 for 3x4x5.

    # Try ranks from the lower bound upward
    for rank in range(20, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break

        # Try multiple strategies for this rank
        # Strategy 1: Direct search with small value set
        for seed in range(20):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank_direct(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}

        # Strategy 2: Gradient-based optimization
        for seed in range(10):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank_gradient(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}

        # Strategy 3: Random walk
        for seed in range(10):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank_random_walk(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}

    return best_result
