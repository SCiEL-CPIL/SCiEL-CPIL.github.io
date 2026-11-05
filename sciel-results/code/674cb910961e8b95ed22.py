import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Exact-rational greedy rank-1 extraction (rank minimization) ----------
    def greedy_extract(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        # Target tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,n+i)
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

        def matmul(a, b):
            # a: num_u x num_v, b: num_v x num_w -> num_u x num_w
            out = [[Fraction(0)] * num_w for _ in range(num_u)]
            for a_idx in range(num_u):
                ra = a[a_idx]
                row = out[a_idx]
                for b_idx in range(num_v):
                    x = ra[b_idx]
                    if x:
                        rb = b[b_idx]
                        for c_idx in range(num_w):
                            row[c_idx] += x * rb[c_idx]
            return out

        def matvec(a, v):
            out = [Fraction(0)] * num_u
            for a_idx in range(num_u):
                ra = a[a_idx]
                s = Fraction(0)
                for b_idx in range(num_v):
                    x = ra[b_idx]
                    if x:
                        s += x * v[b_idx]
                out[a_idx] = s
            return out

        def outer(a, b):
            return [[a[i] * b[j] for j in range(num_v)] for i in range(num_u)]

        def residual_norm2(R):
            s = Fraction(0)
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    x = ra[b_idx]
                    if x:
                        s += x * x
            return s

        def shrink(M):
            # M: num_u x num_v, return (num_u, num_v, factor) all integer
            den = 1
            for i in range(num_u):
                for j in range(num_v):
                    f = M[i][j]
                    if f:
                        d = f.denominator
                        if d > 1:
                            den *= d // np.gcd(den, d)
            den = int(den)
            if den == 1:
                return M, 1
            S = [[int(M[i][j] * den) for j in range(num_v)] for i in range(num_u)]
            return S, den

        def matvec_int(S, v):
            # S integer, v Fraction -> list Fraction
            out = [Fraction(0)] * num_u
            for i in range(num_u):
                row = S[i]
                s = Fraction(0)
                for j in range(num_v):
                    x = row[j]
                    if x:
                        s += x * v[j]
                out[i] = s
            return out

        def outer_int(S, w):
            # S integer, w Fraction -> num_u x num_v Fraction
            out = []
            for i in range(num_u):
                row = S[i]
                out.append([row[j] * w[i] for j in range(num_v)])
            return out

        # Start from the Strassen solution as a valid initial decomposition
        base = strassen_block_solution()
        U0 = [[Fraction(x) for x in row] for row in base["U"]]
        V0 = [[Fraction(x) for x in row] for row in base["V"]]
        W0 = [[Fraction(x) for x in row] for row in base["W"]]
        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for r in range(len(U0[0])):
            for a_idx in range(num_u):
                ua = U0[a_idx][r]
                if ua:
                    for b_idx in range(num_v):
                        vb = V0[b_idx][r]
                        if vb:
                            uv = ua * vb
                            if uv:
                                for c_idx in range(num_w):
                                    wc = W0[c_idx][r]
                                    if wc:
                                        R[a_idx][b_idx][c_idx] -= uv * wc
        # R should now be exactly T
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def pick_candidate():
            # Pick candidate that minimizes the "cost" of extraction
            # Prefer candidates where the slice R[:, b_idx, c_idx] has low rank
            best_cost = None
            best_cand = None
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        if rb[c_idx] != 0:
                            # Estimate cost: count nonzeros in R[:, b_idx, c_idx] and R[a_idx, :, c_idx]
                            cost1 = sum(1 for u_i in range(num_u) if R[u_i][b_idx][c_idx] != 0)
                            cost2 = sum(1 for b_i in range(num_v) if R[a_idx][b_i][c_idx] != 0)
                            cost = cost1 + cost2
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_cand = (a_idx, b_idx, c_idx, rb[c_idx])
            return best_cand

        def try_extract(a_idx, b_idx, c_idx, val, den):
            # build S = (1/den) * R[a_idx, :, c_idx]  (num_u x num_v)
            S = [[0] * num_v for _ in range(num_u)]
            for u_i in range(num_u):
                row = S[u_i]
                src = R[u_i][b_idx][c_idx]
                for b_i in range(num_v):
                    f = R[u_i][b_i][c_idx]
                    if f:
                        row[b_i] = int(f * den)
            # u vector: u[a_idx]=1, u[u_i] = -R[u_i,a_idx,c_idx] * den
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][a_idx][c_idx]
                if f:
                    u[u_i] = Fraction(-int(f * den), den)
            # verify S * v == u  (v = S[b_idx,:])
            v = S[b_idx][:]
            pred = matvec_int(S, u)  # placeholder, recompute properly below
            # Actually compute S @ v where v is integer row -> integer vector
            Sv = [0] * num_u
            for u_i in range(num_u):
                row = S[u_i]
                s = 0
                for b_i in range(num_v):
                    if row[b_i] and v[b_i]:
                        s += row[b_i] * v[b_i]
                Sv[u_i] = s
            # u_int = u * den
            u_int = [0] * num_u
            for u_i in range(num_u):
                if u[u_i]:
                    u_int[u_i] = int(u[u_i] * den)
            if any(Sv[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False
            # v = S[b_idx,:], w[c_idx]=1, w[w_i] = -R[a_idx,b_idx,w_i]*den
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = Fraction(-int(f * den), den)
            # verify S2 = (1/den) R[a_idx,b_idx,:] satisfies S2 @ w == u
            S2 = [[0] * num_w for _ in range(num_u)]
            for u_i in range(num_u):
                row = S2[u_i]
                for w_i in range(num_w):
                    f = R[u_i][b_idx][w_i]
                    if f:
                        row[w_i] = int(f * den)
            S2w = [0] * num_u
            for u_i in range(num_u):
                row = S2[u_i]
                s = 0
                for w_i in range(num_w):
                    if row[w_i] and int(w[w_i] * den):
                        s += row[w_i] * int(w[w_i] * den)
                S2w[u_i] = s
            if any(S2w[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False

            # update R -= (1/den^2) S ⊗ v ⊗ w
            # R[u,b,w] -= (1/den^2) S[u,b] * v[b] * w[w]
            for u_i in range(num_u):
                Su = S[u_i]
                for b_i in range(num_v):
                    if not Su[b_i]:
                        continue
                    vv = v[b_i]
                    if not vv:
                        continue
                    base_val = Fraction(Su[b_i] * vv, den * den)
                    for w_i in range(num_w):
                        if w[w_i]:
                            R[u_i][b_i][w_i] -= base_val * w[w_i]
            U_list.append(u)
            V_list.append([Fraction(x) for x in v])
            W_list.append(w)
            return True

        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            cand = pick_candidate()
            if cand is None:
                break
            a_idx, b_idx, c_idx, val = cand
            if not try_extract(a_idx, b_idx, c_idx, val, val.denominator):
                # Try a few other candidates before giving up
                found = False
                for a_idx2 in range(num_u):
                    if found:
                        break
                    ra2 = R[a_idx2]
                    for b_idx2 in range(num_v):
                        if found:
                            break
                        rb2 = ra2[b_idx2]
                        for c_idx2 in range(num_w):
                            if rb2[c_idx2] != 0 and (a_idx2, b_idx2, c_idx2) != (a_idx, b_idx, c_idx):
                                if try_extract(a_idx2, b_idx2, c_idx2, rb2[c_idx2], rb2[c_idx2].denominator):
                                    found = True
                                    break
                if not found:
                    return None

        if pick_candidate() is not None:
            return None  # did not reach zero residual within target_rank

        # Build final integer/half-integer matrices
        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        for r in range(rank):
            for u_i in range(num_u):
                if U_list[r][u_i]:
                    Uf[u_i][r] = float(U_list[r][u_i])
            for b_i in range(num_v):
                if V_list[r][b_i]:
                    Vf[b_i][r] = float(V_list[r][b_i])
            for w_i in range(num_w):
                if W_list[r][w_i]:
                    Wf[w_i][r] = float(W_list[r][w_i])
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Direct rank-40 construction (guaranteed valid) ----------
    def rank40_solution():
        """
        Trivial decomposition with rank n*m*p.
        r corresponds to (i0, j0, k0).
        U[(i,j), r] = 1 if i==i0 and j==j0
        V[(j,k), r] = 1 if j==j0 and k==k0
        W[(k,i), r] = 1 if k==k0 and i==i0
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    r = (i0 * m + j0) * p + k0
                    # U
                    U[i0 * m + j0][r] = 1
                    # V
                    V[j0 * p + k0][r] = 1
                    # W
                    W[k0 * n + i0][r] = 1
        
        return {"U": U, "V": V, "W": W}

    # ---------- Main: try Strassen first, fallback to Rank 40 ----------
    # Try Strassen block solution (Rank 36)
    best = strassen_block_solution()
    best_rank = len(best["U"][0])
    
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to guaranteed Rank 40
        best = rank40_solution()
        best_rank = len(best["U"][0])

    return best
# EVOLVE-BLOCK-END
