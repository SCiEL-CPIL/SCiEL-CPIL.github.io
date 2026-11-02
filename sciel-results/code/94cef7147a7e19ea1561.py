"""Matrix multiplication 2,3,8: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix mult."""
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline with margin (evaluator kills at 150s)

    n, m, p = DIMENSIONS
    U_dims = n * m
    V_dims = m * p
    W_dims = p * n

    # Build the target tensor non-zeros
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))

    K = len(non_zeros)
    
    # Precompute indices grouped by each mode for fast verification/solving
    indices_by_w = {i: [] for i in range(W_dims)}
    indices_by_u = {i: [] for i in range(U_dims)}
    indices_by_v = {i: [] for i in range(V_dims)}

    for idx, (u, v, w) in enumerate(non_zeros):
        indices_by_w[w].append(idx)
        indices_by_u[u].append(idx)
        indices_by_v[v].append(idx)

    def verify_decomp(U, V, W, tol=1e-9):
        """
        Checks if U, V, W exactly decompose the tensor.
        Returns True if valid.
        """
        r = U.shape[1]
        # Check all non-zero entries
        for u, v, w in non_zeros:
            # Sum over c: U[u, c] * V[v, c] * W[w, c]
            prod = np.sum(U[u, :] * V[v, :] * W[w, :])
            if abs(prod - 1.0) > tol:
                return False

        # Check all zero entries
        # We can sample a few zero entries to save time, but for small dims, check all.
        nz_set = set(non_zeros)
        for u in range(U_dims):
            for v in range(V_dims):
                for w in range(W_dims):
                    if (u, v, w) not in nz_set:
                        prod = np.sum(U[u, :] * V[v, :] * W[w, :])
                        if abs(prod) > tol:
                            return False
        return True

    def round_to_rational(X, denom=2):
        # Round to nearest multiple of 1/denom
        return np.round(X * denom) / denom

    def al_solve(U, V, W, rank, iters=50, tol=1e-10):
        """
        Alternating Least Squares for CP decomposition.
        Minimizes sum (T - sum U_c V_c W_c)^2
        """
        for it in range(iters):
            # Update W
            for w in range(W_dims):
                if not indices_by_w[w]:
                    continue
                idxs = indices_by_w[w]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, v, _ = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * V[v, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol

            # Update U
            for u in range(U_dims):
                if not indices_by_u[u]:
                    continue
                idxs = indices_by_u[u]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    _, v, w = non_zeros[idx]
                    A[r_idx, :] = V[v, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol

            # Update V
            for v in range(V_dims):
                if not indices_by_v[v]:
                    continue
                idxs = indices_by_v[v]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, _, w = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol

            # Check convergence
            if it > 0 and it % 5 == 0:
                err = 0.0
                for idx in range(K):
                    u, v, w = non_zeros[idx]
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    err += (val - 1.0)**2
                if err < tol:
                    break

        return U, V, W

    def try_rank(rank, max_restarts=10):
        """Try to find a valid decomposition of given rank."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None

            # Initialize with random values
            # Scaling helps convergence
            U = np.random.randn(U_dims, rank) * 0.5
            V = np.random.randn(V_dims, rank) * 0.5
            W = np.random.randn(W_dims, rank) * 0.5

            # Run ALS
            U, V, W = al_solve(U, V, W, rank, iters=50)

            # Try rounding to integers and half-integers
            for denom in [1, 2]:
                U_r = round_to_rational(U, denom)
                V_r = round_to_rational(V, denom)
                W_r = round_to_rational(W, denom)

                # Verify
                if verify_decomp(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r

            # Also try without rounding in case the solution is already close to rational
            if verify_decomp(U, V, W, tol=1e-9):
                return U, V, W

        return None

    # Schoolbook rank is n*m*p = 48
    # We try ranks below 48
    # Lower bound is max(n*m, m*p, p*n) = 24
    # Let's try from 40 down to 30? No, we want MINIMAL rank.
    # So we should try 30, 31, ... 47.
    # If we find a valid one, we return it.
    # If none, we return 48.
    
    # To save time, we prioritize ranks that are more likely to be minimal.
    # However, without knowing the exact rank, we must search.
    # Let's try a few restarts for each rank.
    
    best_result = None
    
    # We'll try ranks from 35 to 47. Ranks below 35 are less likely to be valid 
    # for this size, but we'll try a few if time permits.
    # Actually, let's just iterate from 30 to 47.
    
    for rank in range(30, 48):
        if time.time() > deadline - 5:  # Leave 5s for schoolbook
            break
        
        # Use more restarts for lower ranks? Or fixed?
        # Let's use 5 restarts per rank to keep it fast.
        result = try_rank(rank, max_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            U_list = [[float(U_r[i, j]) for j in range(rank)] for i in range(U_dims)]
            V_list = [[float(V_r[i, j]) for j in range(rank)] for i in range(V_dims)]
            W_list = [[float(W_r[i, j]) for j in range(rank)] for i in range(W_dims)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook decomposition
    rank = n * m * p
    U = np.zeros((U_dims, rank))
    V = np.zeros((V_dims, rank))
    W = np.zeros((W_dims, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(U_dims)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(V_dims)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(W_dims)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
