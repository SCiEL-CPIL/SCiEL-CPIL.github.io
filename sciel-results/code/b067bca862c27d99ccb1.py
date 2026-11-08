import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    n, m, p = 2, 3, 3
    num_outputs = p * n      # 6
    num_inputs_a = n * m     # 6
    num_inputs_b = m * p     # 9

    def make_tensor(U, V, W, rank):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def target_tensor():
        T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ia = i * m + j
                    ib = j * p + k
                    out_idx = k * n + i
                    T[ia, ib, out_idx] = 1.0
        return T

    target = target_tensor()

    # ------------------------------------------------------------------
    # Verified rank-17 block decomposition (guaranteed baseline)
    #   Strassen(7) for 2x2x2 + schoolbook(4) for 2x2x1
    # + schoolbook(4) for 2x1x2 + schoolbook(2) for 2x1x1 = 17
    # ------------------------------------------------------------------
    def build_rank17():
        rank = 17
        U = np.zeros((num_inputs_a, rank))
        V = np.zeros((num_inputs_b, rank))
        W = np.zeros((num_outputs, rank))

        # Strassen for A[:,:2] * B[:2,:2]
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1

        # Schoolbook A[:,:2] * B[:2,2]
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
        U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
        U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1

        # Schoolbook A[:,2] * B[2,:2]
        U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
        U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
        U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
        U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1

        # Schoolbook A[:,2] * B[2,2]
        U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
        U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

        return U, V, W, rank

    # ------------------------------------------------------------------
    # Rank-16 search: try to find a decomposition of rank 16 by
    # randomized rank-revealing factorization of the tensor.
    # We attempt to express the tensor as a sum of 16 rank-1 terms
    # using a greedy/QR approach on the matricizations.
    # ------------------------------------------------------------------
    def try_rank16(rng, max_iter=200):
        # Matricize the target tensor
        # T as (num_inputs_a, num_inputs_b*num_outputs)
        T_a = target.reshape(num_inputs_a, num_inputs_b * num_outputs)
        # T as (num_inputs_b, num_inputs_a*num_outputs)
        T_b = target.reshape(num_inputs_b, num_inputs_a * num_outputs)
        # T as (num_outputs, num_inputs_a*num_inputs_b)
        T_c = target.reshape(num_outputs, num_inputs_a * num_inputs_b)

        rank = 16
        best = None
        best_err = np.inf

        for it in range(max_iter):
            # Randomized rank-revealing: pick random projections
            # to get an approximate factorization
            # Use random matrix to project and then QR
            # T_a = U @ V^T approximately
            # Strategy: random projection to reduce dimension, then SVD
            # Actually, use a direct approach: random basis for the
            # column space, then solve.

            # Random projection matrix of size (num_inputs_b*num_outputs, rank)
            P = rng.standard_normal((num_inputs_b * num_outputs, rank))
            # Projected tensor: T_a @ P  -> (num_inputs_a, rank)
            A_proj = T_a @ P  # (6, 16)

            # Now we want U (6,16) such that T_a ≈ U @ Q^T for some Q
            # and V, W consistent.
            # This is getting complex. Let's use a simpler greedy approach.

            # Greedy: iteratively find rank-1 components that best
            # approximate the residual.
            residual = target.copy()
            U_list = []
            V_list = []
            W_list = []

            for r in range(rank):
                # Find best rank-1 approximation of residual
                # Use SVD of the flattened tensor
                flat = residual.reshape(-1)
                # Try random rank-1 vectors
                best_r_err = np.inf
                best_Ur = None
                best_Vr = None
                best_Wr = None

                # Use a few random tries
                for trial in range(20):
                    ur = rng.standard_normal(num_inputs_a)
                    vr = rng.standard_normal(num_inputs_b)
                    wr = rng.standard_normal(num_outputs)
                    # Scale to best fit
                    # residual ≈ ur ⊗ vr ⊗ wr * scalar
                    # scalar = <residual, ur⊗vr⊗wr> / ||ur⊗vr⊗wr||^2
                    prod = np.einsum('i,j,k->ijk', ur, vr, wr)
                    denom = np.sum(prod * prod)
                    if denom < 1e-12:
                        continue
                    scalar = np.sum(residual * prod) / denom
                    err = np.sum((residual - scalar * prod) ** 2)
                    if err < best_r_err:
                        best_r_err = err
                        best_Ur = ur * np.sqrt(abs(scalar))
                        best_Vr = vr * (1 if scalar >= 0 else -1) * np.sqrt(abs(scalar))
                        best_Wr = wr

                if best_Ur is None:
                    break

                residual -= np.einsum('i,j,k->ijk', best_Ur, best_Vr, best_Wr)
                U_list.append(best_Ur)
                V_list.append(best_Vr)
                W_list.append(best_Wr)

                # Check if residual is small enough
                res_norm = np.linalg.norm(residual)
                if res_norm < 1e-10:
                    break

            if len(U_list) < rank:
                continue

            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)

            # Verify
            computed = make_tensor(U, V, W, rank)
            err = np.max(np.abs(computed - target))
            if err < best_err:
                best_err = err
                best = (U, V, W)

            if best_err < 1e-10:
                break

            if time.time() - start_time > deadline - 30:
                break

        if best is not None and best_err < 1e-10:
            return best[0], best[1], best[2], rank
        return None

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------
    # First, verify the rank-17 baseline
    U17, V17, W17, r17 = build_rank17()
    computed17 = make_tensor(U17, V17, W17, r17)
    err17 = np.max(np.abs(computed17 - target))

    if err17 < 1e-10:
        # Try rank-16 search if we have time
        if time.time() - start_time < deadline - 40:
            rng = np.random.default_rng(42)
            result = try_rank16(rng, max_iter=50)
            if result is not None:
                U16, V16, W16, r16 = result
                computed16 = make_tensor(U16, V16, W16, r16)
                err16 = np.max(np.abs(computed16 - target))
                if err16 < 1e-10:
                    return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}

        # Return verified rank-17
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    # Fallback: naive rank-18
    rank = 18
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U[ia, r] = 1
                V[ib, r] = 1
                W[out_idx, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
