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

    # ---------- Beam Search Greedy Rank-1 Extraction ----------
    def beam_greedy_extract(start_rank, beam_width, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        
        # Initialize Residual R with the target tensor T
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

        base = strassen_block_solution()
        U0 = [[Fraction(x) for x in row] for row in base["U"]]
        V0 = [[Fraction(x) for x in row] for row in base["V"]]
        W0 = [[Fraction(x) for x in row] for row in base["W"]]
        
        # Calculate Residual R = T - Sum(U0*V0*W0)
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
        
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        # State: (R, U_list, V_list, W_list)
        # We start with one state
        states = [(R, [], [], [])]
        
        def pick_candidates_fast(R, limit):
            candidates = []
            count = 0
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        val = rb[c_idx]
                        if val != 0:
                            count += 1
                            if count > 500: # Limit search space for speed
                                break
                    if count > 500: break
                if count > 500: break
            # This simple loop is too slow for beam search if not optimized.
            # We will use a simpler heuristic: pick the first few non-zeros with lowest slice density.
            
            best_cands = []
            for a_idx in range(num_u):
                for b_idx in range(num_v):
                    for c_idx in range(num_w):
                        val = R[a_idx][b_idx][c_idx]
                        if val != 0:
                            # Calculate score (density)
                            nz_v = sum(1 for b_i in range(num_v) if R[a_idx][b_i][c_idx] != 0)
                            nz_u = sum(1 for u_i in range(num_u) if R[u_i][b_idx][c_idx] != 0)
                            score = nz_v + nz_u
                            best_cands.append((score, a_idx, b_idx, c_idx, val))
            
            best_cands.sort(key=lambda x: x[0])
            return best_cands[:limit]

        def try_extract(R, a_idx, b_idx, c_idx, val):
            inv_val = Fraction(1) / val
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx: continue
                f = R[u_i][b_idx][c_idx]
                if f: u[u_i] = f * inv_val
            
            v = [Fraction(0)] * num_v
            v[b_idx] = val
            for b_i in range(num_v):
                if b_i == b_idx: continue
                f = R[a_idx][b_i][c_idx]
                if f: v[b_i] = f
            
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx: continue
                f = R[a_idx][b_idx][w_i]
                if f: w[w_i] = f * inv_val
            
            # Verify extraction
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui: continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi: continue
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        wi = w[w_i]
                        if not wi:
                            if row_r[w_i] != 0: return None
                        else:
                            expected = ui * bi * wi
                            if row_r[w_i] != expected: return None
            
            # Update R (copy)
            R_new = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
            for u_i in range(num_u):
                ui = u[u_i]
                if not ui: continue
                for b_i in range(num_v):
                    bi = v[b_i]
                    if not bi: continue
                    base = ui * bi
                    row_r = R[u_i][b_i]
                    for w_i in range(num_w):
                        R_new[u_i][b_i][w_i] = row_r[w_i]
                        if wi: R_new[u_i][b_i][w_i] -= base * wi
            
            return R_new, u, v, w

        # Beam Search Loop
        for step in range(start_rank):
            if time.time() > time_budget:
                break
            if not states:
                break
                
            new_states = []
            seen_R = set() # To avoid duplicates? R is large, so this is expensive. Skip for now.
            
            for R, U_list, V_list, W_list in states:
                # Check if R is zero
                is_zero = True
                for a_idx in range(num_u):
                    for b_idx in range(num_v):
                        for c_idx in range(num_w):
                            if R[a_idx][b_idx][c_idx] != 0:
                                is_zero = False
                                break
                        if not is_zero: break
                    if not is_zero: break
                
                if is_zero:
                    # Found a solution
                    rank = len(U_list)
                    Uf = [[0] * rank for _ in range(num_u)]
                    Vf = [[0] * rank for _ in range(num_v)]
                    Wf = [[0] * rank for _ in range(num_w)]
                    for r in range(rank):
                        for u_i in range(num_u):
                            if U_list[r][u_i]: Uf[u_i][r] = float(U_list[r][u_i])
                        for b_i in range(num_v):
                            if V_list[r][b_i]: Vf[b_i][r] = float(V_list[r][b_i])
                        for w_i in range(num_w):
                            if W_list[r][w_i]: Wf[w_i][r] = float(W_list[r][w_i])
                    return {"U": Uf, "V": Vf, "W": Wf}

                # Expand
                cands = pick_candidates_fast(R, beam_width * 2)
                for score, a_idx, b_idx, c_idx, val in cands:
                    res = try_extract(R, a_idx, b_idx, c_idx, val)
                    if res:
                        R_new, u, v, w = res
                        new_states.append((R_new, U_list + [u], V_list + [v], W_list + [w]))
            
            # Prune states
            if not new_states:
                break
            
            # Keep top beam_width states based on some metric? 
            # For now, just keep the first beam_width valid ones.
            states = new_states[:beam_width]

        return None

    best = strassen_block_solution()
    
    # Try to reduce rank using beam search
    if time.time() < deadline - 10:
        # Start with rank 36, try to reduce
        # Beam width 2 is a good balance
        res = beam_greedy_extract(36, 2, deadline - 10)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res

    return best
# EVOLVE-BLOCK-END
