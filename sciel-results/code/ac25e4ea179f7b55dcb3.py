import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook fallback: rank 36
    rank_school = n * m * p
    U_school = [[0.0] * rank_school for _ in range(n * m)]
    V_school = [[0.0] * rank_school for _ in range(m * p)]
    W_school = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}

    def verify_fast(U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i*m+j][l] * V[j*p+k][l] * W[k*n+i][l]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    n3, m3, p3 = 3, 3, 3
    nm3 = n3 * m3
    mp3 = m3 * p3
    pn3 = p3 * n3

    def solve_W(U_r, V_r, r):
        W = np.zeros((pn3, r))
        for k in range(p3):
            for i in range(n3):
                A = np.zeros((m3, r))
                for j in range(m3):
                    A[j, :] = U_r[i*m3+j, :] * V_r[j*p3+k, :]
                try:
                    A_pinv = np.linalg.pinv(A)
                    w = A_pinv @ np.ones(m3)
                    if np.max(np.abs(A @ w - np.ones(m3))) > 1e-10:
                        return None
                    w2 = w * 2
                    if np.max(np.abs(w2 - np.round(w2))) > 1e-10:
                        return None
                    W[k*n3+i, :] = w
                except Exception:
                    return None
        return W

    def try_rank_3x3x3(r, max_attempts, seed_base=0):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            seed = seed_base + attempt
            rng = np.random.RandomState(seed)
            mod = attempt % 5
            if mod == 0:
                U_r = rng.randint(-1, 2, (nm3, r)).astype(float)
                V_r = rng.randint(-1, 2, (mp3, r)).astype(float)
            elif mod == 1:
                U_r = rng.randint(0, 2, (nm3, r)).astype(float) * 2 - 1
                V_r = rng.randint(0, 2, (mp3, r)).astype(float) * 2 - 1
            elif mod == 2:
                U_r = rng.randint(-2, 3, (nm3, r)).astype(float)
                V_r = rng.randint(-2, 3, (mp3, r)).astype(float)
            elif mod == 3:
                U_r = (rng.rand(nm3, r) < 0.3) * rng.randint(-1, 2, (nm3, r)).astype(float)
                V_r = (rng.rand(mp3, r) < 0.3) * rng.randint(-1, 2, (mp3, r)).astype(float)
            else:
                U_r = rng.randint(-1, 2, (nm3, r)).astype(float) * (1 + (attempt % 2))
                V_r = rng.randint(-1, 2, (mp3, r)).astype(float) * (1 + (attempt % 2))
            
            W_r = solve_W(U_r, V_r, r)
            if W_r is None:
                continue
            
            correct = True
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        s = np.dot(U_r[i*m3+j, :] * V_r[j*p3+k, :], W_r[k*n3+i, :])
                        if abs(s - 1.0) > 1e-9:
                            correct = False
                            break
                    if not correct:
                        break
                if not correct:
                    break
            
            if not correct:
                continue
            
            total_rank = r + 9
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            U_full[:9, :r] = U_r
            V_full[:9, :r] = V_r
            W_full[:9, :r] = W_r
            col = r
            for i in range(n):
                for j in range(m):
                    U_full[i*m+j, col] = 1.0
                    V_full[j*p+3, col] = 1.0
                    W_full[3*n+i, col] = 1.0
                    col += 1
            
            if not verify_fast(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                continue
            
            return {
                "U": [[float(x) for x in row] for row in U_full],
                "V": [[float(x) for x in row] for row in V_full],
                "W": [[float(x) for x in row] for row in W_full]
            }
        return None

    for r1 in range(23, 30):
        if time.time() > deadline:
            break
        if r1 <= 24:
            attempts = 10000
        elif r1 <= 26:
            attempts = 5000
        else:
            attempts = 2000
        result = try_rank_3x3x3(r1, attempts, seed_base=r1 * 1000000)
        if result is not None:
            return result

    return fallback
