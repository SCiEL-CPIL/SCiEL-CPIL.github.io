import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    _start = time.time()
    _deadline = _start + 130  # 20s safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Try to find a lower rank decomposition
    # First, verify the known rank-22 solution works, then try to improve

    def build_rank22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def is_half_integer(val):
        return val == int(val) or abs(val * 2 - round(val * 2)) < 1e-10

    def check_valid(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        return True

    def residual_tensor(U_cols, V_cols, W_cols):
        """Compute the residual tensor T - sum of rank-1 terms.
        U_cols: list of (n*m,) arrays
        V_cols: list of (m*p,) arrays
        W_cols: list of (p*n,) arrays
        Returns (n, m, p) array where T[i,j,k]=1.
        """
        res = np.ones((n, m, p))
        for u, v, w in zip(U_cols, V_cols, W_cols):
            # u[i*m+j], v[j*p+k], w[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]
        return res

    def greedy_reduce(U, V, W, rank, max_iters=50):
        """Try to reduce rank by greedily selecting best rank-1 term to subtract
        from residual, then snapping to half-integers."""
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build residual
        res = np.ones((n, m, p))
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]

        # Greedy: at each step, find the best rank-1 term to subtract
        # Strategy: pick a random (i,j,k) with large residual, then solve
        # for the optimal v and w given u (or use mode-3 flattening)

        best_rank_found = rank
        best_U_list = [list(U_np[:, r]) for r in range(rank)]
        best_V_list = [list(V_np[:, r]) for r in range(rank)]
        best_W_list = [list(W_np[:, r]) for r in range(rank)]

        for iteration in range(max_iters):
            if time.time() > _deadline - 20:
                break

            # Try multiple random seeds for the greedy step
            for seed in range(10):
                if time.time() > _deadline - 20:
                    break

                np.random.seed(seed * 1000 + iteration)
                U_try = [U_np[:, r].copy() for r in range(rank)]
                V_try = [V_np[:, r].copy() for r in range(rank)]
                W_try = [W_np[:, r].copy() for r in range(rank)]

                # Rebuild residual
                res_try = np.ones((n, m, p))
                for r in range(rank):
                    u = U_try[r]
                    v = V_try[r]
                    w = W_try[r]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res_try[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]

                # Greedy elimination: try to remove one term at a time
                # For each term r, check if we can replace it with a better
                # rank-1 approximation of the remaining residual
                for r_remove in range(rank):
                    if time.time() > _deadline - 20:
                        break

                    # Compute residual without term r_remove
                    res_wo = np.ones((n, m, p))
                    for r2 in range(rank):
                        if r2 == r_remove:
                            continue
                        u = U_try[r2]
                        v = V_try[r2]
                        w = W_try[r2]
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    res_wo[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]

                    # Now we want to find a rank-1 tensor u⊗v⊗w that best
                    # approximates res_wo (to replace the removed term)
                    # Mode-3 flattening: for each (i,j), we have a vector of length p
                    # res_wo[i,j,:] = u[i,j] * (v[j,:] ⊙ w[:,i])
                    # This is hard to solve directly. Instead, use a simpler approach:
                    # Fix u, solve for v and w via least squares on mode-3.

                    # Try a few random u vectors
                    for u_seed in range(5):
                        if time.time() > _deadline - 20:
                            break

                        # Initialize u from the removed term or randomly
                        if u_seed == 0:
                            u_new = U_try[r_remove].copy()
                        elif u_seed == 1:
                            u_new = np.random.randn(n*m) * 0.1
                        else:
                            u_new = U_try[r_remove].copy() + np.random.randn(n*m) * 0.1

                        # For fixed u, the problem is:
                        # res_wo[i,j,k] ≈ u[i*m+j] * v[j*p+k] * w[k*n+i]
                        # Mode-3 flatten: for each (i,j), vector of length p:
                        # res_wo[i,j,:] = u[i*m+j] * (v[j,:] ⊙ w[:,i])
                        # Let M[i,j] = u[i*m+j], then we need:
                        # res_wo[i,j,:] = M[i,j] * (v[j,:] ⊙ w[:,i])

                        # This is still bilinear. Let's use a simpler approach:
                        # Just check if the existing term r_remove is already
                        # a good approximation of res_wo.

                        # Compute the "error" if we keep term r_remove
                        # vs trying a modified version

                        # Simple heuristic: try scaling the removed term
                        # to best fit res_wo
                        # res_wo ≈ scale * (u_r ⊗ v_r ⊗ w_r)
                        # Inner product: <res_wo, u_r⊗v_r⊗w_r> / ||u_r⊗v_r⊗w_r||^2

                        u_r = U_try[r_remove]
                        v_r = V_try[r_remove]
                        w_r = W_try[r_remove]

                        # Compute inner product of res_wo with u_r⊗v_r⊗w_r
                        ip = 0.0
                        norm_sq = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                                    ip += res_wo[i, j, k] * val
                                    norm_sq += val * val

                        if norm_sq > 1e-15:
                            scale = ip / norm_sq
                            # New term: scale * (u_r, v_r, w_r)
                            u_new = scale * u_r
                            v_new = scale * v_r
                            w_new = scale * w_r

                            # Check if this reduces the residual norm
                            res_after = res_wo.copy()
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        res_after[i, j, k] -= u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i]

                            # Compare: is res_after closer to zero than res_wo?
                            # Actually we want the total residual (with all other terms)
                            # to be closer to zero. Since res_wo is the residual
                            # without term r_remove, and we're adding back a modified
                            # term, the new total residual is res_after.
                            # We want ||res_after|| < ||res_wo|| (which means the
                            # modified term is better than having no term)
                            # But actually we always want to keep a term, so we
                            # just need the modified term to be a good fit.

                            # The key question: can we reduce the total rank?
                            # If the best rank-1 approximation of res_wo has very
                            # small norm, then the term is nearly redundant.

                            # For now, just track if we found a better fit
                            # and try to snap to half-integers

                            # Try snapping the new term to half-integers
                            u_snap = np.round(u_new * 2) / 2
                            v_snap = np.round(v_new * 2) / 2
                            w_snap = np.round(w_new * 2) / 2

                            # Check if the snapped version is still good
                            res_snap = res_wo.copy()
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        res_snap[i, j, k] -= u_snap[i*m+j] * v_snap[j*p+k] * w_snap[k*n+i]

                            # Total residual with snapped term replacing r_remove
                            # (other terms unchanged)
                            # If ||res_snap|| is small, we have a valid decomposition
                            # at the same rank but with a different term.
                            # This doesn't reduce rank, but might enable further reduction.

                            # For rank reduction, we need the term to be exactly zero
                            # in the snapped version. Check if snapping gives zeros.
                            if np.allclose(u_snap, 0) and np.allclose(v_snap, 0) and np.allclose(w_snap, 0):
                                # Term is redundant!
                                new_rank = rank - 1
                                U_new = [U_try[r2].copy() for r2 in range(rank) if r2 != r_remove]
                                V_new = [V_try[r2].copy() for r2 in range(rank) if r2 != r_remove]
                                W_new = [W_try[r2].copy() for r2 in range(rank) if r2 != r_remove]

                                # Verify
                                valid = True
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            s = 0
                                            for r2 in range(new_rank):
                                                s += U_new[r2][i*m+j] * V_new[r2][j*p+k] * W_new[r2][k*n+i]
                                            if abs(s - 1) > 1e-10:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        break

                                if valid:
                                    best_rank_found = new_rank
                                    best_U_list = [list(x) for x in U_new]
                                    best_V_list = [list(x) for x in V_new]
                                    best_W_list = [list(x) for x in W_new]
                                    # Update for next iteration
                                    rank = new_rank
                                    U_try = U_new
                                    V_try = V_new
                                    W_try = W_new
                                    break

            if best_rank_found < rank:
                rank = best_rank_found
                U_try = best_U_list
                V_try = best_V_list
                W_try = best_W_list
                # Continue trying to reduce further
                if time.time() > _deadline - 20:
                    break
            else:
                # No reduction found in this iteration, try a different approach
                break

        return best_rank_found, best_U_list, best_V_list, best_W_list

    # Start with rank-22
    U, V, W, rank = build_rank22()

    if not verify(U, V, W, rank):
        raise RuntimeError("Rank-22 decomposition failed verification")

    if not check_valid(U, V, W):
        raise RuntimeError("Rank-22 has non-half-integer entries")

    best_rank = rank
    best_U, best_V, best_W = U[:], V[:], W[:]

    # Try to reduce rank using greedy method
    if time.time() < _deadline - 25:
        new_rank, new_U, new_V, new_W = greedy_reduce(U, V, W, rank, max_iters=30)
        if new_rank < best_rank:
            # Verify the new solution
            U_chk = [[float(x) for x in row] for row in new_U]
            V_chk = [[float(x) for x in row] for row in new_V]
            W_chk = [[float(x) for x in row] for row in new_W]
            if verify(U_chk, V_chk, new_rank) and check_valid(U_chk, V_chk, W_chk):
                best_rank = new_rank
                best_U, best_V, best_W = U_chk, V_chk, W_chk

    # Also try the simple removal approach
    U_cur, V_cur, W_cur, rank_cur = U[:], V[:], W[:], rank
    for r_remove in range(rank_cur):
        if time.time() > _deadline - 10:
            break

        U_test = [row[:] for row in U_cur]
        V_test = [row[:] for row in V_cur]
        W_test = [row[:] for row in W_cur]

        for i in range(n * m):
            U_test[i][r_remove] = 0
        for j in range(m * p):
            V_test[j][r_remove] = 0
        for k in range(p * n):
            W_test[k][r_remove] = 0

        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank_cur):
                        if r == r_remove:
                            continue
                        s += U_test[i*m+j][r] * V_test[j*p+k][r] * W_test[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            new_rank = rank_cur - 1
            U_new = [[U_test[i][r] for r in range(rank_cur) if r != r_remove] for i in range(n*m)]
            V_new = [[V_test[j][r] for r in range(rank_cur) if r != r_remove] for j in range(m*p)]
            W_new = [[W_test[k][r] for r in range(rank_cur) if r != r_remove] for k in range(p*n)]

            if verify(U_new, V_new, new_rank) and check_valid(U_new, V_new, W_new):
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    U_cur, V_cur, W_cur, rank_cur = U_new, V_new, W_new, new_rank

    return {"U": best_U, "V": best_V, "W": best_W}
