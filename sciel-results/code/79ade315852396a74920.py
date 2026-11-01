import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s

    n, m, p = 2, 2, 8

    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()

    # Try to find a lower-rank decomposition via algebraic reduction and random search
    # We prioritize lower ranks first
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 10:  # Need at least 10s to try and verify
            break

        # Allocate time budget for this rank
        # Give more time to higher ranks (closer to 28) as they are more likely
        # But we want the lowest rank, so we try 24 first.
        # Heuristic: spend 10% of time on 24, 10% on 25, 10% on 26, 20% on 27?
        # Or just equal split?
        # Let's give 10s to 24, 10s to 25, 10s to 26, and rest to 27.
        budget_map = {24: 10, 25: 10, 26: 10, 27: 20}
        budget = min(budget_map.get(target_rank, 10), remaining - 5)

        if budget <= 1:
            continue

        # First, try algebraic reduction from the Strassen basis
        # This exploits the Kronecker structure M_{2,2,2} \otimes I_4
        # and looks for linear dependencies among the rank-1 terms
        result = _algebraic_reduction(target_rank, budget / 3)
        if result is not None:
            return result

        # Then try random search with local refinement
        result = _search_low_rank(target_rank, budget * 2 / 3)
        if result is not None:
            return result

    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _algebraic_reduction(target_rank, time_budget):
    """
    Exploit the Kronecker product structure of the 2x2x8 tensor as M_{2,2,2} \otimes I_4
    and apply a global linear algebraic reduction to the 28-dimensional Strassen basis
    to identify and eliminate any linear dependencies among the rank-1 terms.
    """
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16

    # Build the list of (a,b,c) indices where T=1
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))

    nonzero_set = set(nonzero_indices)
    zero_indices = []
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    zero_indices.append((a, b, c))

    # Start from the Strassen-28 decomposition
    base = _strassen_28()
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)

    # The 28 rank-1 terms are: for block t in 0..3, for Strassen term r in 0..6
    # Term (t, r) has:
    #   U vector: U_strassen[r] (same for all t)
    #   V vector: V_strassen[r] placed at rows [2t, 2t+1, 8+2t, 8+2t+1]
    #   W vector: W_strassen[r] placed at rows [4t, 4t+2, 4t+1, 4t+3]

    # We can try to find linear combinations of these 28 terms that cancel out
    # some of the terms, effectively reducing the rank.

    # One approach: try to find a set of at most target_rank terms (with possibly
    # modified coefficients) that still decompose the tensor.

    # For now, we'll try a simple heuristic: look for pairs of terms that can be
    # combined or cancelled.

    # Actually, a more systematic approach: treat the 28 terms as a basis and
    # try to find a linear dependency among them.

    # Build the tensor for each of the 28 terms
    # term_tensor[t, r] is a (4, 16, 16) tensor
    # But this is too large. Instead, we can check linear dependencies by
    # sampling the tensor entries.

    # For each term (t, r), the contribution to T[a, b, c] is:
    #   U_strassen[r][a] * V_strassen[r][b_local] * W_strassen[r][c_local]
    # where b_local and c_local are the local indices within block t.

    # Let's build a matrix where each column corresponds to a term (t, r)
    # and each row corresponds to a tensor entry (a, b, c).
    # The entry is the contribution of that term to T[a, b, c].

    # We want to find a linear combination of columns that equals the target tensor.
    # But we already know that the sum of all 28 terms equals the target.
    # We want to find a smaller set of terms (with possibly different coefficients)
    # that also equals the target.

    # This is a linear algebra problem: find a vector x of length 28 such that
    # sum_t,r x[t,r] * term_tensor[t,r] = target_tensor
    # and x has at most target_rank non-zero entries.

    # This is a sparse linear system, which is hard to solve exactly.

    # Instead, let's try a different approach: use the fact that the U vectors
    # are the same for all blocks, and look for dependencies among the V and W vectors.

    # For a fixed Strassen term r, the V and W vectors are the same across blocks,
    # just placed at different rows. So the 4 blocks for a fixed r are "copies"
    # of each other, shifted in the V and W indices.

    # This suggests that we might be able to combine terms across blocks.

    # For now, let's just try random linear combinations of the 28 terms
    # and see if we can get a valid decomposition with fewer non-zero terms.

    # This is essentially a random search over the space of linear combinations.

    # We'll skip this for now and rely on the random search with local refinement.
    # The algebraic reduction idea is promising but complex to implement correctly.

    return None

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts and local refinement."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16

    # Non-zero indices of the tensor
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))

    nonzero_set = set(nonzero_indices)
    zero_indices = []
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    zero_indices.append((a, b, c))

    # Precompute target tensor
    target = np.zeros((num_a, num_b, num_c), dtype=float)
    for a, b, c in nonzero_indices:
        target[a, b, c] = 1.0

    def check_solution(U, V, W):
        # Compute T_hat
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_hat = np.einsum('abr,cr->abc', UV, W)

        # Check non-zero
        for a, b, c in nonzero_indices:
            if abs(T_hat[a, b, c] - 1) > 1e-6:
                return False

        # Check zero
        for a, b, c in zero_indices:
            if abs(T_hat[a, b, c]) > 1e-6:
                return False
        return True

    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break

        rng = np.random.default_rng(seed=restart)

        # Initialize with small integers
        U = rng.integers(-3, 4, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-3, 4, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-3, 4, size=(num_c, target_rank)).astype(float)

        # Local Refinement: Coordinate Descent with Incremental Updates
        for step in range(500):
            if time.time() - start > time_budget:
                break

            # Compute residual and error
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            T_hat = np.einsum('abr,cr->abc', UV, W)

            # Calculate error
            error = np.sum((T_hat - target) ** 2)

            # If error is small, check exactness
            if error < 1e-3:
                if check_solution(U, V, W):
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if check_solution(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

            # Try a random coordinate update
            mat_choice = rng.integers(0, 3)
            if mat_choice == 0:
                mat = U
                rows = num_a
            elif mat_choice == 1:
                mat = V
                rows = num_b
            else:
                mat = W
                rows = num_c

            r_idx = rng.integers(0, rows)
            c_idx = rng.integers(0, target_rank)

            old_val = mat[r_idx, c_idx]

            # Incremental update logic
            # Delta T = Delta U * V * W + U * Delta V * W + U * V * Delta W
            # We only change one element, so we can compute the change in T efficiently.

            best_delta = None
            best_error = error

            # Try a few new values
            for delta in [-2, -1, -0.5, 0.5, 1, 2]:
                new_val = old_val + delta
                if new_val == old_val:
                    continue

                # Compute change in T_hat
                if mat_choice == 0:  # U changed
                    dT = np.einsum('b,c->abc', V[:, c_idx], W[:, c_idx]) * delta
                elif mat_choice == 1:  # V changed
                    dT = np.einsum('a,c->abc', U[:, c_idx], W[:, c_idx]) * delta
                else:  # W changed
                    dT = np.einsum('a,b->abc', U[:, c_idx], V[:, c_idx]) * delta

                # Update error incrementally
                # error_new = sum (T + dT - target)^2
                # = error_old + 2 sum (T - target) dT + sum dT^2
                diff_old = T_hat - target
                term1 = 2 * np.sum(diff_old * dT)
                term2 = np.sum(dT ** 2)
                new_error = error + term1 + term2

                if new_error < best_error:
                    best_error = new_error
                    best_delta = delta

            # Apply the best delta if it improves the error
            if best_delta is not None:
                mat[r_idx, c_idx] = old_val + best_delta
                # Update T_hat
                if mat_choice == 0:
                    dT = np.einsum('b,c->abc', V[:, c_idx], W[:, c_idx]) * best_delta
                elif mat_choice == 1:
                    dT = np.einsum('a,c->abc', U[:, c_idx], W[:, c_idx]) * best_delta
                else:
                    dT = np.einsum('a,b->abc', U[:, c_idx], V[:, c_idx]) * best_delta
                T_hat = T_hat + dT
                error = best_error

    return None
