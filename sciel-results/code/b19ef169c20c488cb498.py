import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_tensor():
    n, m, p = 2, 2, 6
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for l in range(m):
            a = i * m + l
            for k in range(p):
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _build_rank21():
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((12, rank), dtype=np.float64)
    W = np.zeros((12, rank), dtype=np.float64)

    # Strassen 7-term for 2x2
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1}),
    ]

    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                v_idx = l * p + 2 * b + (0 if l == 0 else 0)
                # B indexing: B[l, k] where l in {0,1}, k in {0..5}
                # In our Strassen, the 2x2 B block for column-block b is B[:, 2b:2b+2]
                # l=0 -> B[0, 2b], l=1 -> B[0, 2b+1], l=2 -> B[1, 2b], l=3 -> B[1, 2b+1]
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1

    return U, V, W

def _verify(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    recon = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(recon - T, ord=np.inf) < tol

def _find_decomposition(T, target_rank, max_time=30.0):
    """Try to find a rank-target_rank decomposition of tensor T using ALS."""
    n_a, n_b, n_c = T.shape
    best_error = float('inf')
    best_result = None
    start = time.time()
    n_restarts = 200

    # Pre-flatten
    T_flat = T.reshape(n_a, n_b * n_c)

    for restart in range(n_restarts):
        if time.time() - start > max_time:
            break

        # Random initialization
        U = np.random.randn(n_a, target_rank)
        V = np.random.randn(n_b, target_rank)
        W = np.random.randn(n_c, target_rank)

        for iteration in range(200):
            if time.time() - start > max_time:
                break

            # Build VW matrix: (n_b * n_c, target_rank)
            VW = np.zeros((n_b * n_c, target_rank))
            for r in range(target_rank):
                VW[:, r] = (V[:, r] * W[:, r]).flatten()

            # Solve for U: T[a, :] = U[a, :] @ VW^T  ->  VW @ U[a,:]^T = T[a,:]^T
            U_new = np.zeros((n_a, target_rank))
            for a in range(n_a):
                rhs = T[a, :, :].flatten()
                U_new[a, :], _, _, _ = np.linalg.lstsq(VW, rhs, rcond=None)
            U = U_new

            # Build UW matrix: (n_a * n_c, target_rank)
            UW = np.zeros((n_a * n_c, target_rank))
            for r in range(target_rank):
                UW[:, r] = (U[:, r] * W[:, r]).flatten()

            V_new = np.zeros((n_b, target_rank))
            for b in range(n_b):
                rhs = T[:, b, :].flatten()
                V_new[b, :], _, _, _ = np.linalg.lstsq(UW, rhs, rcond=None)
            V = V_new

            # Build UV matrix: (n_a * n_b, target_rank)
            UV = np.zeros((n_a * n_b, target_rank))
            for r in range(target_rank):
                UV[:, r] = (U[:, r] * V[:, r]).flatten()

            W_new = np.zeros((n_c, target_rank))
            for c in range(n_c):
                rhs = T[:, :, c].flatten()
                W_new[c, :], _, _, _ = np.linalg.lstsq(UV, rhs, rcond=None)
            W = W_new

        # Check reconstruction error
        recon = np.zeros((n_a, n_b, n_c))
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        error = np.linalg.norm(recon - T, ord=np.inf)

        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())

        if error < 1e-12:
            break

    if best_error < 1e-10:
        U, V, W = best_result
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r, V_r, W_r, T):
            return U_r, V_r, W_r
        return U, V, W
    else:
        return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()

    U21, V21, W21 = _build_rank21()
    best_U, best_V, best_W = U21, V21, W21
    best_rank = 21

    deadline = time.time() + 130  # Safety margin before 150s limit

    # Try to find lower-rank decompositions
    for target_rank in range(20, 12, -1):
        if time.time() > deadline - 10:
            break
        time_budget = min(30.0, deadline - time.time() - 5)
        if time_budget <= 0:
            break
        found = _find_decomposition(T, target_rank, max_time=time_budget)
        if found is not None:
            U, V, W = found
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
