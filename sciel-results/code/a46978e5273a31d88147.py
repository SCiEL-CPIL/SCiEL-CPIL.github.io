"""Matrix multiplication 2x3x9: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    # Precompute tensor entries
    tensor_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.add((a, b, c))

    # Build target tensor as flat array for fast access
    nm, mp, pn = n * m, m * p, p * n
    T_flat = np.zeros(nm * mp * pn)
    for (a, b, c) in tensor_entries:
        T_flat[a * mp * pn + b * pn + c] = 1.0

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition using Fraction arithmetic."""
        U_frac = [[Fraction(int(U[a, s]), 1) for s in range(rank)] for a in range(nm)]
        V_frac = [[Fraction(int(V[b, s]), 1) for s in range(rank)] for b in range(mp)]
        W_frac = [[Fraction(W[c, s]) for s in range(rank)] for c in range(pn)]

        for (a, b, c) in tensor_entries:
            val = sum(U_frac[a][s] * V_frac[b][s] * W_frac[c][s] for s in range(rank))
            if val != 1:
                return False
        # Check a sample of zero entries
        import random
        random.seed(42)
        all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
        zero_indices = [idx for idx in all_indices if idx not in tensor_entries]
        sample_size = min(200, len(zero_indices))
        for (a, b, c) in random.sample(zero_indices, sample_size):
            val = sum(U_frac[a][s] * V_frac[b][s] * W_frac[c][s] for s in range(rank))
            if val != 0:
                return False
        return True

    def als_refine(U, V, W, rank, max_iter=30, tol=1e-10):
        """Alternating Least Squares refinement for tensor decomposition."""
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)

        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Update U: for each a, solve M_a^T U[a,:] = T[a,:,:].flatten()
            for a in range(nm):
                M_a = np.zeros(mp * pn, rank)
                target = T_flat[a * mp * pn:(a + 1) * mp * pn].copy()
                for b in range(mp):
                    for c in range(pn):
                        M_a[b * pn + c, :] = V[b, :] * W[c, :]
                try:
                    U[a, :], res, rnk, sv = np.linalg.lstsq(M_a.T, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Update V
            for b in range(mp):
                M_b = np.zeros(nm * pn, rank)
                target = np.zeros(nm * pn)
                for a in range(nm):
                    for c in range(pn):
                        M_b[a * pn + c, :] = U[a, :] * W[c, :]
                        target[a * pn + c] = T_flat[a * mp * pn + b * pn + c]
                try:
                    V[b, :], res, rnk, sv = np.linalg.lstsq(M_b.T, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Update W
            for c in range(pn):
                M_c = np.zeros(nm * mp, rank)
                target = np.zeros(nm * mp)
                for a in range(nm):
                    for b in range(mp):
                        M_c[a * mp + b, :] = U[a, :] * V[b, :]
                        target[a * mp + b] = T_flat[a * mp * pn + b * pn + c]
                try:
                    W[c, :], res, rnk, sv = np.linalg.lstsq(M_c.T, target, rcond=None)
                except np.linalg.LinAlgError:
                    pass

            # Check convergence
            recon_err = 0.0
            for (a, b, c) in tensor_entries:
                val = sum(U[a, s] * V[b, s] * W[c, s] for s in range(rank))
                recon_err += (val - 1.0) ** 2
            if recon_err < tol:
                break

        return U, V, W

    def try_find_decomposition(rank, num_trials=20):
        """Try to find a rank-r decomposition with multiple random initializations."""
        prime = 257

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None

            np.random.seed(rank * 1000 + trial * 7 + 42)
            U = np.random.randint(0, prime, size=(nm, rank)).astype(np.float64)
            V = np.random.randint(0, prime, size=(mp, rank)).astype(np.float64)
            W = np.random.randint(0, prime, size=(pn, rank)).astype(np.float64)

            # Refine with ALS
            U, V, W = als_refine(U, V, W, rank, max_iter=20)

            # Check if the decomposition is close to exact
            max_err = 0.0
            for (a, b, c) in tensor_entries:
                val = sum(U[a, s] * V[b, s] * W[c, s] for s in range(rank))
                max_err = max(max_err, abs(val - 1.0))

            if max_err < 1e-6:
                # Try to round to integers/half-integers
                U_int = np.round(U * 2) / 2.0
                V_int = np.round(V * 2) / 2.0
                W_int = np.round(W * 2) / 2.0

                # Verify with exact arithmetic
                if verify_decomposition(U_int, V_int, W_int, rank):
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

                # Also try pure integer rounding
                U_int2 = np.round(U).astype(int).astype(np.float64)
                V_int2 = np.round(V).astype(int).astype(np.float64)
                W_int2 = np.round(W).astype(int).astype(np.float64)
                if verify_decomposition(U_int2, V_int2, W_int2, rank):
                    U_list = U_int2.tolist()
                    V_list = V_int2.tolist()
                    W_list = W_int2.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks in increasing order (lower rank = better score)
    # Start from a reasonable lower bound
    min_rank = max(n, m, p) * min(n, m)  # heuristic lower bound
    ranks_to_try = sorted(set(range(min_rank, 54)))

    # Allocate more trials to lower ranks
    trials_per_rank = {
        r: max(5, 30 - r) if r < 40 else max(3, 20 - r)
        for r in ranks_to_try
    }

    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        result = try_find_decomposition(rank, num_trials=trials_per_rank.get(rank, 10))
        if result is not None:
            return result

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
