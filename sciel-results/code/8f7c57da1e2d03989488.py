import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _try_rank17_structured(n, m, p, seed, max_iters=500):
    """
    Try to find rank-17 decomposition using structured initialization.
    
    Key idea: Start from schoolbook (rank 18) and try to merge two terms.
    For 2x3 x 3x3, we have 18 terms. If we can express two of the rank-1
    tensors as a combination that still satisfies all equations with one fewer term,
    we get rank 17.
    
    Alternative: Use the algebraic structure. Fix w_1 = e_{(0,0)}.
    Then term 1 contributes only to (i=0, j=*, k=0) entries: u_1[0,j]*v_1[j,0]*1.
    So it covers (0,0,0), (0,1,0), (0,2,0) with values u_1[0,0]*v_1[0,0], etc.
    
    Let's try: set u_1[0,0]=1, u_1[0,1]=1, u_1[0,2]=1, v_1[0,0]=1, v_1[1,0]=1, v_1[2,0]=1
    Then term 1 gives: (0,0,0)=1, (0,1,0)=1, (0,2,0)=1. 
    But we need each to be exactly 1, so this works for those 3 entries.
    
    Now we need 16 more terms to cover the remaining 15 entries:
    (0,0,1), (0,0,2), (0,1,1), (0,1,2), (0,2,1), (0,2,2),
    (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1), (1,2,2)
    
    That's 15 entries with 16 terms - we have one extra degree of freedom!
    """
    rng = random.Random(seed)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    R = 17
    
    # Fix term 0 to cover (0,0,0), (0,1,0), (0,2,0)
    u0 = [0.0] * (n * m)
    v0 = [0.0] * (m * p)
    w0 = [0.0] * (p * n)
    u0[u_idx(0, 0)] = 1.0
    u0[u_idx(0, 1)] = 1.0
    u0[u_idx(0, 2)] = 1.0
    v0[v_idx(0, 0)] = 1.0
    v0[v_idx(1, 0)] = 1.0
    v0[v_idx(2, 0)] = 1.0
    w0[w_idx(0, 0)] = 1.0
    
    # Remaining 16 terms (r=1..16) need to satisfy 15 equations
    # For each (i,j,k) not covered by term 0:
    #   sum_{r=1}^{16} u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
    # (i,j,k) covered by term 0: (0,0,0), (0,1,0), (0,2,0) -> these are already satisfied
    
    # Actually wait - term 0 contributes u_0[i,j]*v_0[j,k]*w_0[k,i] to ALL (i,j,k),
    # not just those where it's nonzero. Let me recompute.
    # w_0[k,i] = 1 only when k=0, i=0. So term 0 only contributes to (i=0, j, k=0) for j=0,1,2.
    # For (i=0, j, k=0): contribution = u_0[0,j]*v_0[j,0]*1 = 1*1 = 1. Good.
    # For all other (i,j,k): contribution = 0.
    
    # So we need terms 1-16 to satisfy:
    # For (i,j,k) in {(0,0,1),(0,0,2),(0,1,1),(0,1,2),(0,2,1),(0,2,2),
    #                 (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2),
    #                 (1,2,0),(1,2,1),(1,2,2)}:
    #   sum_{r=1}^{16} u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
    
    # That's 15 equations, 16 terms. We have 1 extra term which gives flexibility.
    
    # Strategy: Use 15 of the 16 terms as a "schoolbook-like" decomposition for the 15 entries,
    # and use the 16th term as a "correction" that can be tuned.
    
    # Initialize: for each of the 15 target entries, create a term that covers it.
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not (i == 0 and k == 0):  # Skip those covered by term 0
                    targets.append((i, j, k))
    
    # 15 targets, 16 terms. Assign first 15 terms to cover each target.
    u = [[0.0] * (n * m) for _ in range(R)]
    v = [[0.0] * (m * p) for _ in range(R)]
    w = [[0.0] * (p * n) for _ in range(R)]
    
    # Term 0 fixed
    u[0] = u0
    v[0] = v0
    w[0] = w0
    
    # Terms 1-15: one per target
    for idx, (i, j, k) in enumerate(targets[:15]):
        r = idx + 1
        u[r][u_idx(i, j)] = 1.0
        v[r][v_idx(j, k)] = 1.0
        w[r][w_idx(k, i)] = 1.0
    
    # Term 16: free, initialize randomly
    for i in range(n * m):
        u[16][i] = rng.uniform(-0.5, 0.5)
    for j in range(m * p):
        v[16][j] = rng.uniform(-0.5, 0.5)
    for k in range(p * n):
        w[16][k] = rng.uniform(-0.5, 0.5)
    
    def compute_residual(u, v, w):
        res = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    res[(i, j, k)] = s - 1.0
        return res
    
    # Alternating minimization:
    # Given v, w: solve for u (linear)
    # Given u, w: solve for v (linear)  
    # Given u, v: solve for w (linear)
    
    def solve_u_given_vw(v, w):
        # For each (i,j,k): sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
        # Variables: u_r[i,j] for r=0..16, but u_0 is fixed.
        # Actually, let's keep all u_r free except we fix w_0.
        # Build linear system: 18 equations, 17*6 = 102 unknowns (u_r[i,j] for r=1..16, i in 0..1, j in 0..2)
        # Plus u_0 is fixed, so its contribution is known.
        
        A = np.zeros((n * m * p, (R - 1) * n * m))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Subtract term 0's contribution
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * n * m + i * m + j] = v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    eq += 1
        
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        
        u_new = [row[:] for row in u]
        for r in range(1, R):
            for i in range(n):
                for j in range(m):
                    u_new[r][i * m + j] = sol[(r - 1) * n * m + i * m + j]
        return u_new
    
    def solve_v_given_uw(u, w):
        A = np.zeros((n * m * p, (R - 1) * m * p))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * m * p + j * p + k] = u[r][u_idx(i, j)] * w[r][w_idx(k, i)]
                    eq += 1
        
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        
        v_new = [row[:] for row in v]
        for r in range(1, R):
            for j in range(m):
                for k in range(p):
                    v_new[r][j * p + k] = sol[(r - 1) * m * p + j * p + k]
        return v_new
    
    def solve_w_given_uv(u, v):
        A = np.zeros((n * m * p, (R - 1) * p * n))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * p * n + k * n + i] = u[r][u_idx(i, j)] * v[r][v_idx(j, k)]
                    eq += 1
        
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        
        w_new = [row[:] for row in w]
        for r in range(1, R):
            for k in range(p):
                for i in range(n):
                    w_new[r][k * n + i] = sol[(r - 1) * p * n + k * n + i]
        return w_new
    
    best_res = float('inf')
    best_state = None
    
    for it in range(max_iters):
        if deadline_exceeded():
            break
        
        u_new = solve_u_given_vw(v, w)
        if u_new is None:
            break
        u = u_new
        
        v_new = solve_v_given_uw(u, w)
        if v_new is None:
            break
        v = v_new
        
        w_new = solve_w_given_uv(u, v)
        if w_new is None:
            break
        w = w_new
        
        res = compute_residual(u, v, w)
        norm = sum(x * x for x in res.values()) ** 0.5
        
        if norm < best_res:
            best_res = norm
            best_state = (u[:], v[:], [row[:] for row in w])
            if norm < 1e-10:
                break
    
    if best_state is None:
        return None
    
    u, v, w = best_state
    
    # Try to round to half-integers
    def round_half(x):
        h = x * 2
        r = round(h)
        if abs(h - r) < 1e-6:
            return Fraction(r, 2)
        # Try integer
        ri = round(x)
        if abs(x - ri) < 1e-6:
            return Fraction(ri)
        return Fraction(x)
    
    U_frac = [[round_half(x) for x in row] for row in u]
    V_frac = [[round_half(x) for x in row] for row in v]
    W_frac = [[round_half(x) for x in row] for row in w]
    
    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        return {"U": [[float(x) for x in row] for row in U_frac],
                "V": [[float(x) for x in row] for row in V_frac],
                "W": [[float(x) for x in row] for row in W_frac]}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)
    
    # Try structured rank-17 search with multiple seeds
    for seed in range(30):
        if deadline_exceeded():
            break
        result = _try_rank17_structured(n, m, p, seed, max_iters=300)
        if result is not None:
            return result
    
    return fallback


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
