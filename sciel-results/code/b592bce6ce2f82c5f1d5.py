"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 140  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, r, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U[:, :r], V[:, :r], W[:, :r])
        return float(np.max(np.abs(recon - T))) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # ---- Build the verified rank-28 Strassen decomposition ----
    # Strassen 2x2*2x2 rank-1 components: (U_entries, V_entries, W_entries)
    # P,Q,R are 2x2 indexed (row,col).
    strassen_components = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def build_strassen28():
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        # 4 block terms: (a_col_off, b_row_off, b_col_off, c_col_off)
        # A block s in {0,1} (cols 2s,2s+1); B block (br,bc) in {0,1}^2 (rows 2br,2br+1; cols 2bc,2bc+1)
        # C block: term (s, br, bc) -> C row i, cols 2bc,2bc+1  (since C = sum_s A_s * B_{s?})
        # Actually C0 = A0*B00 + A1*B10 ; C1 = A0*B01 + A1*B11
        # So term index: (a_block, b_row_block, b_col_block) with a_block in {0,1},
        #   b_row_block = a_block (since A0 pairs with B00,B01 and A1 with B10,B11)
        #   b_col_block in {0,1}
        # C col offset = 2*b_col_block
        terms = [
            (0, 0, 0, 0),  # A0 * B00 -> C cols 0,1
            (0, 0, 1, 2),  # A0 * B01 -> C cols 2,3
            (1, 1, 0, 0),  # A1 * B10 -> C cols 0,1
            (1, 1, 1, 2),  # A1 * B11 -> C cols 2,3
        ]
        for ti, (a_off, b_row_off, b_col_off, c_col_off) in enumerate(terms):
            for t in range(7):
                col = ti * 7 + t
                u_ent, v_ent, w_ent = strassen_components[t]
                # U: A element (ai, aj) where aj local in {0,1} -> global col a_off+aj
                for (ai, aj), val in u_ent.items():
                    u_idx = ai * m + (a_off + aj)
                    U_full[u_idx, col] += val
                # V: B element (bi, bj) where bi local row in {0,1} -> global row b_row_off+bi
                #    bj local col in {0,1} -> global col b_col_off+bj
                for (bi, bj), val in v_ent.items():
                    v_idx = (b_row_off + bi) * p + (b_col_off + bj)
                    V_full[v_idx, col] += val
                # W: C element (ci, cj) where ci row in {0,1}, cj local col in {0,1}
                #    global C col = c_col_off + cj ; W index = (global_col)*n + ci
                for (ci, cj), val in w_ent.items():
                    w_idx = (c_col_off + cj) * n + ci
                    W_full[w_idx, col] += val
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()

    if not verify(U28, V28, W28, 28, tol=1e-8):
        # Fallback to schoolbook rank 32
        U = np.zeros((n * m, schoolbook_rank)); V = np.zeros((m * p, schoolbook_rank)); W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0; V[j * p + k, col] = 1.0; W[k * n + i, col] = 1.0; col += 1
        return to_lists(U, V, W, schoolbook_rank)

    # ---- Try to reduce to rank 27 ----
    # Strategy: fix 27 of the 28 columns, treat the remaining as a free variable
    # (Uf, Vf, Wf). Optimize (Uf, Vf, Wf) so that the full reconstruction matches T.
    # If a solution exists with Uf,Vf,Wf snap-able to half-integers, we get rank 27.
    #
    # The free column is a rank-1 tensor u⊗v⊗w. The equation is:
    #   base_27 + u⊗v⊗w = T   =>   u⊗v⊗w = T - base_27
    # where base_27 = sum of the 27 fixed rank-1 terms.
    # So we need T - base_27 to be a rank-1 tensor. We search over which column to
    # free and optimize u,v,w to minimize ||u⊗v⊗w - (T - base_27)||.

    def smooth_loss(F, beta):
        absF = np.abs(F)
        m = float(absF.max())
        if m < 1e-15:
            return 0.0
        shifted = absF - m
        lse = m + np.log(np.sum(np.exp(beta * shifted))) / beta
        return lse

    def refine_rank27(free_col, seed, max_iters=6000):
        rng = np.random.RandomState(seed)
        # Fixed columns: all except free_col
        fixed = [t for t in range(28) if t != free_col]
        base = np.einsum('at,bt,ct->abc', U28[:, fixed], V28[:, fixed], W28[:, fixed])
        target = T - base  # this must equal u⊗v⊗w

        # Initialize free column
        # Heuristic: scale target by a rank-1 approximation
        # Use SVD of target reshaped to get initial u,v,w
        # target is (n*m, m*p, p*n). For rank-1, pick u = vec of first mode, etc.
        # Simple init: random small, or use the freed column as a warm start.
        u0 = U28[:, free_col].copy()
        v0 = V28[:, free_col].copy()
        w0 = W28[:, free_col].copy()
        # Perturb to escape the (incorrect) fixed point
        u = u0 + rng.randn(n * m) * 0.3
        v = v0 + rng.randn(m * p) * 0.3
        w = w0 + rng.randn(p * n) * 0.3

        best_loss = np.inf
        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()

        beta = 60.0
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            recon = np.einsum('a,b,c->abc', u, v, w)
            F = recon - target
            loss = smooth_loss(F, beta)
            absF = np.abs(F)
            m = float(absF.max())
            if m < 1e-15:
                dF = np.zeros_like(F)
            else:
                shifted = absF - m
                weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
                dF = weights * np.sign(F)
            gU = np.einsum('abc,b,c->a', dF, v, w)
            gV = np.einsum('abc,a,c->b', dF, u, w)
            gW = np.einsum('abc,a,b->c', dF, u, v)
            err_l2 = float(np.sum(F * F))
            gnorm = float(np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))) + 1e-12
            lr = min(0.1 / (1.0 + err_l2), 0.3 / gnorm)
            u -= lr * gU
            v -= lr * gV
            w -= lr * gW

            recon2 = np.einsum('a,b,c->abc', u, v, w)
            F2 = recon2 - target
            cur_loss = smooth_loss(F2, beta)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                if best_loss < 1e-12:
                    break

        # High-precision L2 polish (exact gradient of ||u⊗v⊗w - target||^2)
        u, v, w = best_u, best_v, best_w
        for it in range(3000):
            if time.time() > deadline - 8:
                break
            recon = np.einsum('a,b,c->abc', u, v, w)
            F = recon - target
            gU = np.einsum('abc,b,c->a', F, v, w)
            gV = np.einsum('abc,a,c->b', F, u, w)
            gW = np.einsum('abc,a,b->c', F, u, v)
            err = float(np.sum(F * F))
            lr = 0.05 / (1.0 + err)
            u -= lr * gU
            v -= lr * gV
            w -= lr * gW
            if err < 1e-16:
                break

        # Assemble full rank-28 matrices with the optimized free column,
        # then snap to half-integers and verify at rank 27.
        U_full = U28.copy(); V_full = V28.copy(); W_full = W28.copy()
        U_full[:, free_col] = u; V_full[:, free_col] = v; W_full[:, free_col] = w

        for snap_factor in [2, 4, 1]:
            Us = np.round(U_full * snap_factor) / snap_factor
            Vs = np.round(V_full * snap_factor) / snap_factor
            Ws = np.round(W_full * snap_factor) / snap_factor
            if verify(Us, Vs, Ws, 28, tol=1e-8):
                # Check if the free column is effectively zero (rank 27)
                # or if we can drop it. Actually we want rank 27: drop the free col
                # and verify the remaining 27 reproduce T.
                # But the free col may be needed. Instead: build a rank-27 matrix
                # by using the 27 fixed cols only, and check.
                # The true rank-27 test: does sum of 27 fixed cols == T?
                # That's base == T, i.e., target == 0.
                # If target ~ 0, then 27 cols suffice.
                if float(np.max(np.abs(target))) < 1e-6:
                    U27 = U_full[:, fixed]; V27 = V_full[:, fixed]; W27 = W_full[:, fixed]
                    if verify(U27, V27, W27, 27, tol=1e-8):
                        return U27, V27, W27, 27
                # Otherwise the free col carries info; we cannot drop it.
                # But we can still return rank 28 (no improvement). Skip.
                return None
        return None

    # Try freeing each column with a few seeds
    for free_col in range(28):
        for seed in [0, 1, 2, 3, 5]:
            if time.time() > deadline - 20:
                break
            res = refine_rank27(free_col, seed)
            if res is not None:
                U27, V27, W27, r27 = res
                return to_lists(U27, V27, W27, r27)
        if time.time() > deadline - 20:
            break

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
