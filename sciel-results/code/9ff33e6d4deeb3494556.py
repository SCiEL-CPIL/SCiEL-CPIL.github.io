"""Matrix multiplication 3,3,6 – rank-minimal exact tensor decomposition.

Strategy:
  1.  Try to construct a valid decomposition with rank < 54 (the schoolbook bound).
      We use a randomized / heuristic search over the space of decompositions,
      verifying exactness with the standard tensor identity:
          T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
      where T is the multiplication tensor (T_{i,j,k} = 1 if the (i,j,k) entry
      of A*B is non-zero, which for our case is always 1).
  2.  If no better decomposition is found before the deadline, return the
      schoolbook decomposition (rank 54), which is guaranteed correct.

The program is designed to run within 120 s.
"""

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

DIMENSIONS = [3, 3, 6]

DEADLINE_S = 120
SAFETY_MARGIN_S = 5  # return at least this much before the deadline

def _build_tensor():
    """Return the multiplication tensor T of shape (n, m, p).
    T[i, j, k] = 1 always (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    n, m, p = DIMENSIONS
    return np.ones((n, m, p), dtype=np.int64)

def _tensor_from_decomposition(U, V, W):
    """Compute the tensor implied by the decomposition (U, V, W).
    U: shape (n*m, rank)
    V: shape (m*p, rank)
      W: shape (p*n, rank)
    Returns T of shape (n, m, p) where
        T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, p, r] * W3[p, n, r]
    # Use einsum for clarity
    T = np.einsum('ijr,jpr,pnr->ijk', U3, V3, W3)
    return T

def _verify(U, V, W, tol=1e-9):
    """Check whether the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    T_true = _build_tensor()
    T_calc = _tensor_from_decomposition(U, V, W)
    return np.allclose(T_calc, T_true, atol=tol, rtol=0)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _random_decomposition(rank, seed=None):
    """Generate a random decomposition with integer entries in {-1, 0, 1}."""
    if seed is not None:
        random.seed(seed)
    n, m, p = DIMENSIONS
    U = np.random.randint(-1, 2, size=(n * m, rank))
    V = np.random.randint(-1, 2, size=(m * p, rank))
    W = np.random.randint(-1, 2, size=(p * n, rank))
    return U, V, W

def _local_search(U, V, W, max_iters=5000, tol=1e-9):
    """Attempt to reduce the error via local search.
    We try to flip individual entries to improve agreement with T_true.
    Returns the best (U, V, W) found.
    """
    n, m, p = DIMENSIONS
    T_true = _build_tensor()
    rank = U.shape[1]

    # Compute initial error
    T_calc = _tensor_from_decomposition(U, V, W)
    current_err = np.sum((T_calc - T_true) ** 2)

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = current_err

    # Precompute some useful slices
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    # We'll do a simple coordinate-descent-like search.
    # For efficiency, we only consider flipping entries that are non-zero
    # or entries that, when flipped, might reduce the error.
    # Given the small size, we can afford a moderate number of iterations.

    for it in range(max_iters):
        # Pick a random matrix and random entry
        mat_choice = random.choice(['U', 'V', 'W'])
        if mat_choice == 'U':
            r_idx = random.randint(0, n * m - 1)
            c_idx = random.randint(0, rank - 1)
            old_val = U[r_idx, c_idx]
            # Try flipping to the other value in {-1, 0, 1}
            new_val = random.choice([v for v in (-1, 0, 1) if v != old_val])
            U[r_idx, c_idx] = new_val
            T_calc = _tensor_from_decomposition(U, V, W)
            new_err = np.sum((T_calc - T_true) ** 2)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[r_idx, c_idx] = old_val
        elif mat_choice == 'V':
            r_idx = random.randint(0, m * p - 1)
            c_idx = random.randint(0, rank - 1)
            old_val = V[r_idx, c_idx]
            new_val = random.choice([v for v in (-1, 0, 1) if v != old_val])
            V[r_idx, c_idx] = new_val
            T_calc = _tensor_from_decomposition(U, V, W)
            new_err = np.sum((T_calc - T_true) ** 2)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[r_idx, c_idx] = old_val
        else:  # W
            r_idx = random.randint(0, p * n - 1)
            c_idx = random.randint(0, rank - 1)
            old_val = W[r_idx, c_idx]
            new_val = random.choice([v for v in (-1, 0, 1) if v != old_val])
            W[r_idx, c_idx] = new_val
            T_calc = _tensor_from_decomposition(U, V, W)
            new_err = np.sum((T_calc - T_true) ** 2)
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[r_idx, c_idx] = old_val

        if best_err < tol:
            break

    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S - SAFETY_MARGIN_S

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54

    # We'll try to find a decomposition with rank < schoolbook_rank.
    # Start from a rank that is reasonably below 54 and go up.
    # Known lower bound for 3x3x3 is 19, so 3x3x6 should be at most 38 (two 3x3x3).
    # But we don't know if 38 is achievable or if we can do better.
    # Let's try ranks from 20 up to 53.

    best_result = None
    best_rank = schoolbook_rank

    # We'll try a few random starts for each rank
    max_random_starts = 10

    for rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break

        for rs in range(max_random_starts):
            if time.time() > deadline:
                break

            U, V, W = _random_decomposition(rank, seed=rank * 1000 + rs)
            # Quick check: if the error is already zero, we're done
            T_calc = _tensor_from_decomposition(U, V, W)
            T_true = _build_tensor()
            err = np.sum((T_calc - T_true) ** 2)
            if err < 1e-9:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break

            # Otherwise, do a local search
            U_ls, V_ls, W_ls, err_ls = _local_search(U, V, W, max_iters=2000)
            if err_ls < 1e-9:
                best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
                best_rank = rank
                break

        if best_result is not None:
            break  # Found a decomposition, no need to try higher ranks

    # If no better decomposition found, use schoolbook
    if best_result is None:
        best_result = _schoolbook()

    return best_result

# EVOLVE-BLOCK-END
