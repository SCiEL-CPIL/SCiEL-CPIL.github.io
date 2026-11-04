import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0

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

    # ---------- Direct construction: try to build rank-r decomposition ----------
    def direct_construction(target_rank, time_budget):
        """Try to construct a rank-target_rank decomposition directly using 
        a structured approach based on the tensor's algebraic properties."""
        num_u, num_v, num_w = n * m, m * p, p * n
        
        # The tensor T has T[i*m+j][j*p+k][k*n+i] = 1
        # We want: T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # For small ranks, try a structured approach
        # Key insight: the tensor has a specific pattern we can exploit
        
        # Try a block-based construction
        # For 2x4x5, we can think of it as blocks
        
        # Approach: use the fact that matrix multiplication can be decomposed
        # using fewer multiplications via structured algorithms
        
        # Let's try a direct search with smart initialization
        # Initialize with random-like but structured vectors
        
        import random
        random.seed(42)
        
        # We'll use a greedy approach but with better initialization
        # Start from the identity-like structure
        
        # Actually, let's try a completely different approach:
        # Use the known lower bound and try to hit it
        
        # Lower bound for 2x4x5 matrix mult is at least 2*4+4*5-4 = 20 (from Strassen-type bounds)
        # But actual minimum might be higher
        
        # Let's try ranks from 20 upward
        # For each rank, try to find a decomposition
        
        # For rank r, we need to solve:
        # sum_{r=1}^R U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        
        # This is a system of 40 equations in r*(8+20+10) = 38r unknowns
        # For r >= 2, we have more unknowns than equations, so solutions exist generically
        
        # Let's use a numerical approach with rational reconstruction
        
        # For the given dimensions, try a structured solution
        # The key is that T[i*m+j][j*p+k][k*n+i] = 1 only when indices match
        
        # Let's try to build this step by step
        
        # For 2x4x5, we have:
        # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
        # T[i*4+j][j*5+k][k*2+i] = 1
        
        # Let's enumerate all 40 non-zero entries
        # (0,0,0): T[0][0][0] = 1
        # (0,1,0): T[1][5][0] = 1
        # (0,2,0): T[2][10][0] = 1
        # (0,3,0): T[3][15][0] = 1
        # (1,0,0): T[4][0][1] = 1
        # (1,1,0): T[5][5][1] = 1
        # (1,2,0): T[6][10][1] = 1
        # (1,3,0): T[7][15][1] = 1
        # (0,0,1): T[0][1][2] = 1
        # (0,1,1): T[1][6][2] = 1
        # (0,2,1): T[2][11][2] = 1
        # (0,3,1): T[3][16][2] = 1
        # (1,0,1): T[4][1][3] = 1
        # (1,1,1): T[5][6][3] = 1
        # (1,2,1): T[6][11][3] = 1
        # (1,3,1): T[7][16][3] = 1
        # (0,0,2): T[0][2][4] = 1
        # (0,1,2): T[1][7][4] = 1
        # (0,2,2): T[2][12][4] = 1
        # (0,3,2): T[3][17][4] = 1
        # (1,0,2): T[4][2][5] = 1
        # (1,1,2): T[5][7][5] = 1
        # (1,2,2): T[6][12][5] = 1
        # (1,3,2): T[7][17][5] = 1
        # (0,0,3): T[0][3][6] = 1
        # (0,1,3): T[1][8][6] = 1
        # (0,2,3): T[2][13][6] = 1
        # (0,3,3): T[3][18][6] = 1
        # (1,0,3): T[4][3][7] = 1
        # (1,1,3): T[5][8][7] = 1
        # (1,2,3): T[6][13][7] = 1
        # (1,3,3): T[7][18][7] = 1
        # (0,0,4): T[0][4][8] = 1
        # (0,1,4): T[1][9][8] = 1
        # (0,2,4): T[2][14][8] = 1
        # (0,3,4): T[3][19][8] = 1
        # (1,0,4): T[4][4][9] = 1
        # (1,1,4): T[5][9][9] = 1
        # (1,2,4): T[6][14][9] = 1
        # (1,3,4): T[7][19][9] = 1
        
        # So we need: sum_r U[i*4+j][r] * V[j*5+k][r] * W[k*2+i][r] = 1
        
        # For rank r, this is 40 equations.
        # Let's try to find a solution for small r.
        
        # For r=1: 38 unknowns, 40 equations - overdetermined, likely no solution
        # For r=2: 76 unknowns, 40 equations - underdetermined, solutions exist
        # But we need the solution to be exact and have the right structure
        
        # Let's try a different approach: use the structure of the problem
        # The matrix multiplication tensor for 2x4x5 can be viewed as
        # a combination of smaller matrix multiplication tensors
        
        # Actually, let's just try the greedy approach but much faster
        # by using numpy for the inner loops
        
        # Precompute the target tensor
        T = np.zeros((num_u, num_v, num_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        
        # Start with the Strassen decomposition as initial approximation
        base = strassen_block_solution()
        U0 = np.array(base["U"], dtype=np.float64)
        V0 = np.array(base["V"], dtype=np.float64)
        W0 = np.array(base["W"], dtype=np.float64)
        
        # Compute the residual
        R = np.zeros((num_u, num_v, num_w), dtype=np.float64)
        for r in range(U0.shape[1]):
            R -= np.einsum('ir,jr,kr->ijk', U0[:, r], V0[:, r], W0[:, r])
        R += T
        
        # Now try to reduce the rank by extracting rank-1 terms
        U_list = []
        V_list = []
        W_list = []
        
        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            
            # Find the largest entry in R
            max_idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            if R[max_idx] == 0:
                break
            
            a_idx, b_idx, c_idx = max_idx
            val = R[a_idx, b_idx, c_idx]
            
            if abs(val) < 1e-10:
                break
            
            # Extract a rank-1 term
            # u[a_idx] = 1, v[b_idx] = 1, w[c_idx] = 1
            # Then adjust the other entries
            
            u = np.zeros(num_u)
            v = np.zeros(num_v)
            w = np.zeros(num_w)
            
            u[a_idx] = 1.0
            v[b_idx] = 1.0
            w[c_idx] = 1.0
            
            # For each other u_i, set u[u_i] to cancel R[u_i, b_idx, c_idx]
            for u_i in range(num_u):
                if u_i != a_idx:
                    u[u_i] = -R[u_i, b_idx, c_idx] / val
            
            # For each other v_j, set v[v_j] to cancel R[a_idx, v_j, c_idx]
            for v_j in range(num_v):
                if v_j != b_idx:
                    v[v_j] = -R[a_idx, v_j, c_idx] / val
            
            # For each other w_k, set w[w_k] to cancel R[a_idx, b_idx, w_k]
            for w_k in range(num_w):
                if w_k != c_idx:
                    w[w_k] = -R[a_idx, b_idx, w_k] / val
            
            # Update R
            R -= np.einsum('i,j,k->ijk', u, v, w) * val
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        # Check if we've reduced R to zero
        if np.max(np.abs(R)) > 1e-8:
            return None
        
        # Convert to the required format
        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        for r in range(rank):
            for u_i in range(num_u):
                Uf[u_i][r] = U_list[r][u_i]
            for v_i in range(num_v):
                Vf[v_i][r] = V_list[r][v_i]
            for w_i in range(num_w):
                Wf[w_i][r] = W_list[r][w_i]
        
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Exact-rational greedy rank-1 extraction (rank minimization) ----------
    def greedy_extract(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

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
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def pick_candidate():
            # Pick the entry with maximum absolute value
            best_a, best_b, best_c, best_val = None, None, None, Fraction(0)
            best_abs = Fraction(0)
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        v = rb[c_idx]
                        if v != 0:
                            av = abs(v)
                            if av > best_abs:
                                best_abs = av
                                best_a, best_b, best_c, best_val = a_idx, b_idx, c_idx, v
            if best_a is None:
                return None
            return best_a, best_b, best_c, best_val

        def try_extract(a_idx, b_idx, c_idx, val, den):
            S = [[0] * num_v for _ in range(num_u)]
            for u_i in range(num_u):
                row = S[u_i]
                for b_i in range(num_v):
                    f = R[u_i][b_i][c_idx]
                    if f:
                        row[b_i] = int(f * den)
            
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][a_idx][c_idx]
                if f:
                    u[u_i] = Fraction(-int(f * den), den)
            
            v = S[b_idx][:]
            
            Sv = [0] * num_u
            for u_i in range(num_u):
                row = S[u_i]
                s = 0
                for b_i in range(num_v):
                    if row[b_i] and v[b_i]:
                        s += row[b_i] * v[b_i]
                Sv[u_i] = s
            
            u_int = [0] * num_u
            for u_i in range(num_u):
                if u[u_i]:
                    u_int[u_i] = int(u[u_i] * den)
            
            if any(Sv[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False
            
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = Fraction(-int(f * den), den)
            
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
                    if row[w_i]:
                        w_int = int(w[w_i] * den)
                        if w_int:
                            s += row[w_i] * w_int
                S2w[u_i] = s
            
            if any(S2w[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False

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
                            R[u_i][b_idx][w_i] -= base_val * w[w_i]
            
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
                return None

        if pick_candidate() is not None:
            return None

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

    # ---------- Main: fallback first, then attempt to trim rank ----------
    best = strassen_block_solution()
    best_rank = len(best["U"][0])

    # Try direct construction for lower ranks first (faster)
    for tr in range(20, best_rank):
        if time.time() > deadline:
            break
        res = direct_construction(tr, deadline)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    # Then try greedy extraction
    for tr in range(best_rank - 1, max(best_rank - 10, 1), -1):
        if time.time() > deadline:
            break
        res = greedy_extract(tr, deadline)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    return best
# EVOLVE-BLOCK-END
