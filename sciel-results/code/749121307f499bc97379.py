"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8

    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))

    num_entries = len(tensor_entries)  # 24

    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    entry_set = set(tensor_entries)
    all_uvw = [(u, v, w) for u in range(nm) for v in range(mp) for w in range(pn)]
    non_entries = [t for t in all_uvw if t not in entry_set]

    def verify_solution(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = float(np.dot(U_arr[u], V_arr[v] * W_arr[w]))
            if abs(val - 1.0) > tol:
                return False
        # Check all non-entries
        for (u, v, w) in non_entries:
            val = float(np.dot(U_arr[u], V_arr[v] * W_arr[w]))
            if abs(val) > tol:
                return False
        return True

    def solve_for_W(U_init, V_init, target_rank, tol=1e-9):
        """Given U, V, solve linear system for W and snap to half-integers.
        Returns (W_snapped, error) or (None, inf) if not solvable cleanly."""
        # Build A_W: for each entry (u,v,w), the coefficient of W[w,r] is U[u,r]*V[v,r]
        A_W = np.zeros((num_entries, pn * target_rank))
        b_W = np.ones(num_entries)

        for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
            for r in range(target_rank):
                col = w_idx * target_rank + r
                A_W[e_idx, col] = U_init[u_idx, r] * V_init[v_idx, r]

        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A_W, b_W, rcond=None)
        except np.linalg.LinAlgError:
            return None, float('inf')

        W_curr = W_flat.reshape(pn, target_rank)

        # Snap to half-integers
        W_check = np.round(W_curr * 2) / 2.0

        # Check error on tensor entries with snapped W
        U_arr = np.array(U_init, dtype=float)
        V_arr = np.array(V_init, dtype=float)
        W_arr = np.array(W_check, dtype=float)

        error = 0.0
        for (u, v, w) in tensor_entries:
            val = float(np.dot(U_arr[u], V_arr[v] * W_arr[w]))
            error += abs(val - 1.0)

        if error > 1e-6:
            return None, error

        return W_check, error

    def polish_solution(U_init, V_init, W_init, target_rank, tol=1e-10):
        """High-precision polish: re-solve W given U,V with tighter tolerance,
        then re-solve U given V,W, and re-verify."""
        U_curr = np.array(U_init, dtype=float)
        V_curr = np.array(V_init, dtype=float)
        W_curr = np.array(W_init, dtype=float)

        for _ in range(5):
            # Solve for W given U, V
            A_W = np.zeros((num_entries, pn * target_rank))
            b_W = np.ones(num_entries)
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                for r in range(target_rank):
                    col = w_idx * target_rank + r
                    A_W[e_idx, col] = U_curr[u_idx, r] * V_curr[v_idx, r]
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_curr = W_flat.reshape(pn, target_rank)
            except np.linalg.LinAlgError:
                break

            # Solve for U given V, W
            A_U = np.zeros((num_entries, nm * target_rank))
            b_U = np.ones(num_entries)
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                for r in range(target_rank):
                    col = u_idx * target_rank + r
                    A_U[e_idx, col] = V_curr[v_idx, r] * W_curr[w_idx, r]
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_curr = U_flat.reshape(nm, target_rank)
            except np.linalg.LinAlgError:
                break

        # Snap all to half-integers
        U_snap = np.round(U_curr * 2) / 2.0
        V_snap = np.round(V_curr * 2) / 2.0
        W_snap = np.round(W_curr * 2) / 2.0

        # Verify
        if verify_solution(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), target_rank, tol=tol):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        return None

    # Generate a pool of structured basis vectors for each mode
    # Values: -1, -0.5, 0, 0.5, 1 (small magnitude, half-integer)
    def generate_basis_pool(dim, num_patterns=200, seed=0):
        """Generate a pool of structured vectors of length dim."""
        rng = np.random.RandomState(seed)
        pool = []
        # Unit vectors
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            pool.append(e.copy())
            e[i] = -1.0
            pool.append(e.copy())
        # Sparse vectors with 1-2 nonzeros
        for _ in range(num_patterns):
            v = np.zeros(dim)
            nnz = rng.randint(1, min(3, dim + 1))
            idx = rng.choice(dim, size=nnz, replace=False)
            for i in idx:
                v[i] = rng.choice([-1.0, -0.5, 0.5, 1.0])
            if np.any(v != 0):
                pool.append(v)
        # Dense vectors with small entries
        for _ in range(num_patterns // 2):
            v = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=dim)
            pool.append(v)
        return pool

    pool_U = generate_basis_pool(nm, num_patterns=100, seed=1)
    pool_V = generate_basis_pool(mp, num_patterns=100, seed=2)

    def sample_from_pool(pool, num_cols, rng):
        """Sample num_cols vectors from the pool."""
        idx = rng.randint(0, len(pool), size=num_cols)
        return np.array([pool[i] for i in idx])

    # Search for minimal rank using randomized linear solving with structured U,V
    # Theoretical lower bound is 18 for 2x3x4. We start from 18 to save time.
    # If 18 fails, we go up to 24.

    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break

        found = False
        np.random.seed(42 + target_rank)
        rng = np.random.RandomState(42 + target_rank)

        # Number of random attempts
        num_attempts = 3000

        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break

            # Generate structured U and V from the pool
            U_init = sample_from_pool(pool_U, target_rank, rng)
            V_init = sample_from_pool(pool_V, target_rank, rng)

            # Ensure U and V are not zero vectors
            if np.all(U_init == 0) or np.all(V_init == 0):
                continue

            # Solve for W
            W_check, error = solve_for_W(U_init, V_init, target_rank)
            if W_check is None:
                continue

            # Coarse full verification
            if verify_solution(U_init.tolist(), V_init.tolist(), W_check.tolist(), target_rank, tol=1e-8):
                # Polish pass
                polished = polish_solution(U_init, V_init, W_check, target_rank, tol=1e-10)
                if polished is not None:
                    U_pol, V_pol, W_pol = polished
                    if verify_solution(U_pol, V_pol, W_pol, target_rank, tol=1e-10):
                        best_rank = target_rank
                        best_result = {"U": U_pol, "V": V_pol, "W": W_pol}
                        found = True
                        break
                else:
                    # Use the coarse solution if it passes
                    best_rank = target_rank
                    best_result = {
                        "U": U_init.tolist(),
                        "V": V_init.tolist(),
                        "W": W_check.tolist()
                    }
                    found = True
                    break

        if found:
            break

    # If we found a solution, return it
    if best_rank < n * m * p:
        return best_result

    # Return schoolbook if nothing better found
    return best_result
# EVOLVE-BLOCK-END
