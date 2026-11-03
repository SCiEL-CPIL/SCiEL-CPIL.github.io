"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_numpy(U, V, W, n, m, p):
    """Fast exact verification using numpy einsum."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _snap_half(x):
    return np.rint(x * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Schoolbook decomposition as guaranteed baseline (rank 100)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    best_rank = rank_sb
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    np.random.seed(42)

    # Lower bound: max(n*m, m*p, p*n) = 25
    # Try ranks from 25 upward, but prioritize lower ranks first
    # Use a structured approach: fix U and W randomly, solve for V, snap all, verify
    for target_rank in range(25, min(rank_sb, 80)):
        if time.time() > deadline - 5:
            break

        # Multiple random initializations
        n_attempts = max(1, int((deadline - 5 - time.time()) / 2))
        n_attempts = min(n_attempts, 20)

        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break

            # Random U and W with values biased toward small integers/half-integers
            # Use a mix: some entries 0, ±1, ±0.5
            def random_matrix(rows, cols):
                mat = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        val = np.random.choice([0, 0.5, 1, -0.5, -1, 1.5, -1.5], 
                                               p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05])
                        mat[i, j] = val
                return mat

            U_rand = random_matrix(n * m, target_rank)
            W_rand = random_matrix(p * n, target_rank)

            # Build linear system: for each (i, j, k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Unknowns: V[j*p+k, r] for all j, k, r
            num_eqs = n * m * p
            num_unknowns = m * p * target_rank

            if num_unknowns < num_eqs:
                continue

            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)

            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            col_idx = (j * p + k) * target_rank + r
                            A_mat[eq_idx, col_idx] = U_rand[i * m + j, r] * W_rand[k * n + i, r]
                        eq_idx += 1

            try:
                V_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue

            V_rand = V_flat.reshape(m * p, target_rank)

            # Snap all three to half-integers
            U_snap = _snap_half(U_rand)
            V_snap = _snap_half(V_rand)
            W_snap = _snap_half(W_rand)

            # Verify
            if _verify_numpy(U_snap, V_snap, W_snap, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                break  # Found valid decomposition for this rank, move to next (lower) rank

    # If we found a good rank, also try to refine by searching slightly below
    if best_rank < rank_sb:
        for target_rank in range(max(25, best_rank - 5), best_rank):
            if time.time() > deadline - 5:
                break
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                U_rand = np.random.randn(n * m, target_rank) * 0.5
                W_rand = np.random.randn(p * n, target_rank) * 0.5
                U_rand = _snap_half(U_rand)
                W_rand = _snap_half(W_rand)

                num_eqs = n * m * p
                num_unknowns = m * p * target_rank
                if num_unknowns < num_eqs:
                    continue

                A_mat = np.zeros((num_eqs, num_unknowns))
                b_vec = np.ones(num_eqs)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                col_idx = (j * p + k) * target_rank + r
                                A_mat[eq_idx, col_idx] = U_rand[i * m + j, r] * W_rand[k * n + i, r]
                            eq_idx += 1

                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    continue
                V_rand = V_flat.reshape(m * p, target_rank)
                V_snap = _snap_half(V_rand)

                if _verify_numpy(U_rand, V_snap, W_rand, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {
                            "U": U_rand.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_rand.tolist()
                        }
                    break

    return best_result

# EVOLVE-BLOCK-END
