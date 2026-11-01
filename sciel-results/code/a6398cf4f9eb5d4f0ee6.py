"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor entries: T[(i,j),(j,k),(k,i)] = 1
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    num_entries = len(tensor_entries)  # 24

    # Precompute index arrays for speed
    u_idxs = np.array([e[0] for e in tensor_entries])
    v_idxs = np.array([e[1] for e in tensor_entries])
    w_idxs = np.array([e[2] for e in tensor_entries])

    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    def verify(U, V, W, rank):
        """Fast exact verification. Returns True if all 24 entries sum to 1."""
        # For each entry, sum over rank: U[u,r]*V[v,r]*W[w,r]
        # Vectorize: for each entry e, compute dot over r
        # Build matrix M[e, r] = U[u_e, r] * V[v_e, r], then sum M * W[w_e, r]
        # M shape (num_entries, rank), W_sel shape (num_entries, rank)
        U_sel = U[u_idxs, :]  # (num_entries, rank)
        V_sel = V[v_idxs, :]  # (num_entries, rank)
        W_sel = W[w_idxs, :]  # (num_entries, rank)
        # sum_r U_sel[e,r] * V_sel[e,r] * W_sel[e,r]
        vals = np.einsum('er,er,er->e', U_sel, V_sel, W_sel)
        return bool(np.all(np.abs(vals - 1.0) < 1e-9))

    def snap_half(x):
        return np.round(x * 2) / 2.0

    def solve_linear(W_fixed, U, V, rank, solve_for):
        """Solve for one factor given the other two. Returns snapped solution."""
        # Equations: for each entry e: sum_r U[u_e,r]*V[v_e,r]*W[w_e,r] = 1
        # If solving for W: linear in W[w_e, r] with coeff U[u_e,r]*V[v_e,r]
        # Group: for each (w_idx, r) unknown, it appears in entries where w_e == w_idx
        # Build A (num_entries, pn*rank), b (num_entries)
        A = np.zeros((num_entries, pn * rank))
        for e in range(num_entries):
            u_e, v_e, w_e = u_idxs[e], v_idxs[e], w_idxs[e]
            for r in range(rank):
                A[e, w_e * rank + r] = U[u_e, r] * V[v_e, r]
        b = np.ones(num_entries)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return snap_half(sol.reshape(pn, rank))

    def solve_linear_U(U_fixed, V, W, rank):
        A = np.zeros((num_entries, nm * rank))
        for e in range(num_entries):
            u_e, v_e, w_e = u_idxs[e], v_idxs[e], w_idxs[e]
            for r in range(rank):
                A[e, u_e * rank + r] = V[v_e, r] * W[w_e, r]
        b = np.ones(num_entries)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return snap_half(sol.reshape(nm, rank))

    def solve_linear_V(V_fixed, U, W, rank):
        A = np.zeros((num_entries, mp * rank))
        for e in range(num_entries):
            u_e, v_e, w_e = u_idxs[e], v_idxs[e], w_idxs[e]
            for r in range(rank):
                A[e, v_e * rank + r] = U[u_e, r] * W[w_e, r]
        b = np.ones(num_entries)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return snap_half(sol.reshape(mp, rank))

    def polish(U, V, W, rank, max_iter=30):
        """Final high-precision polish: tighter tolerance, more iterations, exact local solve."""
        U = U.copy(); V = V.copy(); W = W.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            W_new = solve_linear(W, U, V, rank)
            U_new = solve_linear_U(U, V, W_new, rank)
            V_new = solve_linear_V(V, U_new, W_new, rank)
            if np.array_equal(U_new, U) and np.array_equal(V_new, V) and np.array_equal(W_new, W):
                break
            U, V, W = U_new, V_new, W_new
        return U, V, W

    np.random.seed(42)
    half_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        found = False

        # Structured + random initializations
        inits = []

        # Initialization 0: schoolbook-derived (use first target_rank columns, zero-pad)
        U_s = np.zeros((nm, target_rank))
        V_s = np.zeros((mp, target_rank))
        W_s = np.zeros((pn, target_rank))
        # Try to pack schoolbook columns into target_rank by combining
        # For now, just use a random subset pattern
        for r in range(target_rank):
            U_s[:, r] = np.random.choice([-1, 0, 1], size=nm)
            V_s[:, r] = np.random.choice([-1, 0, 1], size=mp)
            W_s[:, r] = np.random.choice([-1, 0, 1], size=pn)
        inits.append((U_s, V_s, W_s))

        num_restarts = 25
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            U_init = np.random.choice(half_vals, size=(nm, target_rank))
            V_init = np.random.choice(half_vals, size=(mp, target_rank))
            W_init = np.random.choice(half_vals, size=(pn, target_rank))
            inits.append((U_init, V_init, W_init))

        for U0, V0, W0 in inits:
            if time.time() > deadline - 4:
                break
            U_curr, V_curr, W_curr = U0.copy(), V0.copy(), W0.copy()

            for iteration in range(80):
                if time.time() > deadline - 3:
                    break
                # Coordinate descent: solve each factor in turn
                W_curr = solve_linear(W_curr, U_curr, V_curr, target_rank)
                U_curr = solve_linear_U(U_curr, V_curr, W_curr, target_rank)
                V_curr = solve_linear_V(V_curr, U_curr, W_curr, target_rank)

                # Quick check
                if verify(U_curr, V_curr, W_curr, target_rank):
                    # Polish for extra confidence
                    U_p, V_p, W_p = polish(U_curr, V_curr, W_curr, target_rank)
                    if verify(U_p, V_p, W_p, target_rank):
                        best_rank = target_rank
                        best_result = {
                            "U": U_p.tolist(),
                            "V": V_p.tolist(),
                            "W": W_p.tolist()
                        }
                        found = True
                        break

            if found:
                break

        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
