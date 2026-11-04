import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Try to find a rank-17 decomposition for the 2x3x3 sub-tensor
    # If found, use it to construct a rank-51 decomposition for 2x3x9
    U_233, V_233, W_233 = search_rank_233(17, deadline)

    if U_233 is not None:
        return construct_239_from_233(U_233, V_233, W_233)

    # Fallback: Schoolbook 54
    return construct_schoolbook(n, m, p)

def construct_239_from_233(U_233, V_233, W_233):
    n, m, p = 2, 3, 9
    rank_233 = U_233.shape[1]
    total_rank = 3 * rank_233

    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    # U is repeated across the 3 blocks
    for block in range(3):
        U_full[:, block * rank_233 : (block + 1) * rank_233] = U_233

    # V mapping: row j*9+k in full -> row j*3+k_local in 233, block k//3
    for j in range(3):
        for k in range(9):
            block = k // 3
            k_local = k % 3
            v_row_full = j * 9 + k
            v_row_233 = j * 3 + k_local
            col_start = block * rank_233
            V_full[v_row_full, col_start:col_start+rank_233] = V_233[v_row_233, :]

    # W mapping: row k*2+i in full -> row k_local*2+i in 233, block k//3
    for k in range(9):
        for i in range(2):
            block = k // 3
            k_local = k % 3
            w_row_full = k * 2 + i
            w_row_233 = k_local * 2 + i
            col_start = block * rank_233
            W_full[w_row_full, col_start:col_start+rank_233] = W_233[w_row_233, :]

    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

def search_rank_233(target_rank, deadline):
    n, m, p = 2, 3, 3
    num_u = n * m  # 6
    num_v = m * p  # 9
    num_w = p * n  # 6

    # Precompute target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Flatten T for efficient solving
    T_flat = T.reshape(num_u, num_v * num_w)

    random.seed(42)
    np.random.seed(42)

    def verify_solution(U, V, W):
        """Check if the decomposition is exact and has half-integer entries."""
        # Check half-integer property
        for mat in [U, V, W]:
            scaled = 2.0 * mat
            rounded = np.round(scaled)
            if not np.allclose(scaled, rounded, atol=1e-6):
                return False

        # Snap to nearest half-integer
        U_s = np.round(2.0 * U) / 2.0
        V_s = np.round(2.0 * V) / 2.0
        W_s = np.round(2.0 * W) / 2.0

        # Verify exactness using einsum
        T_approx = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        return np.allclose(T_approx, T, atol=1e-9)

    # Efficient error calculation for a given U, V, W
    def calc_error(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        return np.sum(diff**2)

    # Strategy: Fix V and W to random small integers, solve for U via least squares
    # Check if U is half-integer and the solution is exact
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')

    # Precompute V and W random initializations
    # We'll try many random (V, W) pairs
    trial = 0
    max_trials = 50000  # Will be limited by deadline

    while time.time() < deadline - 5:
        # Generate random V and W with small integer entries
        # Use -1, 0, 1 as entries (known to work well for structured tensors)
        V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)

        # Build M matrix: M[b*num_w+c, r] = V[b, r] * W[c, r]
        # Shape: (num_v * num_w) x target_rank
        M = (V[:, None, :] * W[None, :, :]).reshape(num_v * num_w, target_rank)

        # Check if M has full column rank
        # Use SVD for numerical stability
        try:
            rank_M = np.linalg.matrix_rank(M)
        except np.linalg.LinAlgError:
            trial += 1
            continue

        if rank_M < target_rank:
            trial += 1
            continue

        # Solve for U: For each row a, solve M @ U[a]^T = T_flat[a]^T
        U = np.zeros((num_u, target_rank))
        valid = True
        for a in range(num_u):
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M, T_flat[a], rcond=None)
                U[a, :] = sol
            except np.linalg.LinAlgError:
                valid = False
                break

        if not valid:
            trial += 1
            continue

        # Check if U is close to half-integers
        U_scaled = 2.0 * U
        U_rounded = np.round(U_scaled)
        if not np.allclose(U_scaled, U_rounded, atol=1e-4):
            trial += 1
            continue

        # Snap U to nearest half-integer
        U_snapped = U_rounded / 2.0

        # Verify exactness
        if verify_solution(U_snapped, V, W):
            # Found a valid solution!
            return U_snapped, V, W

        # Track best error found so far (in case we get close)
        err = calc_error(U_snapped, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_snapped.copy(), V.copy(), W.copy()

        trial += 1

    # If we found a very close solution, try to polish it
    if best_err < 1e-6 and best_U is not None:
        # Try a local search around the best solution
        # Use simulated annealing with half-integer steps
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_err = best_err

        temp = 0.1
        cooling_rate = 0.999
        min_temp = 0.001
        deltas = [0.5, -0.5, 1.0, -1.0]

        while time.time() < deadline - 2:
            # Pick a random matrix, row, col
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = current_U
                rows = num_u
            elif mat_idx == 1:
                mat = current_V
                rows = num_v
            else:
                mat = current_W
                rows = num_w

            row = random.randint(0, rows - 1)
            col = random.randint(0, target_rank - 1)

            old_val = mat[row, col]
            delta = random.choice(deltas)
            new_val = old_val + delta

            # Keep values in reasonable range
            if new_val > 10: new_val = 10
            if new_val < -10: new_val = -10

            mat[row, col] = new_val
            new_err = calc_error(current_U, current_V, current_W)

            delta_err = new_err - current_err

            if delta_err < 0 or random.random() < np.exp(-delta_err / temp):
                current_err = new_err
                if current_err < 1e-9:
                    # Check if it's half-integer
                    if verify_solution(current_U, current_V, current_W):
                        return current_U, current_V, current_W
                temp = max(min_temp, temp * cooling_rate)
            else:
                mat[row, col] = old_val
                temp = max(min_temp, temp * cooling_rate)

    return None, None, None

def construct_schoolbook(n, m, p):
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
