import time
import numpy as np
from fractions import Fraction
import random

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

def _construct_rank17(n, m, p):
    if n != 2 or m != 3 or p != 3:
        return None
    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    # Strassen 7 terms
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1
    # Boundary 10 terms
    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]
    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1
    return {"U": U, "V": V, "W": W}

def _search_rank16(n, m, p, time_budget=110.0):
    if n != 2 or m != 3 or p != 3:
        return None
    R = 16
    n_u, n_v, n_w = n*m, m*p, p*n
    
    def error(U, V, W):
        # Calculate max absolute error of the rounded solution
        # U, V, W are lists of lists of floats
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    ui = U[i*m+j]
                    vk = V[j*p+k]
                    wk = W[k*n+i]
                    for r in range(R):
                        s += ui[r] * vk[r] * wk[r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def round_to_half(x):
        return round(x * 2) / 2.0

    # Random restart local search
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    # Initialize with sparse random matrices
    # Strategy: Each column has 2-3 non-zeros
    U = np.zeros((n_u, R))
    V = np.zeros((n_v, R))
    W = np.zeros((n_w, R))
    
    for r in range(R):
        # U
        nz_u = np.random.choice([2, 3])
        idx_u = np.random.choice(n_u, nz_u, replace=False)
        U[idx_u, r] = np.random.choice([-1, 1], nz_u)
        # V
        nz_v = np.random.choice([2, 3])
        idx_v = np.random.choice(n_v, nz_v, replace=False)
        V[idx_v, r] = np.random.choice([-1, 1], nz_v)
        # W
        nz_w = np.random.choice([2, 3])
        idx_w = np.random.choice(n_w, nz_w, replace=False)
        W[idx_w, r] = np.random.choice([-1, 1], nz_w)

    # Local search: Gradient-free, coordinate descent on signs and sparsity
    # We will use a simple hill climbing on the "rounded error"
    
    # Precompute indices for speed
    # Eq: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    best_err = float('inf')
    best_state = None
    
    # Convert to lists for easier modification in local search? 
    # Or stick to numpy and modify specific entries.
    
    # We will use a simple random walk / hill climbing
    # Objective: Minimize error of the *continuous* solution, but we care about the *rounded* one.
    # Let's minimize the continuous error, then round.
    
    # Actually, minimizing continuous error directly might not lead to a good rounded solution.
    # Let's try to minimize the error of the *rounded* solution directly.
    
    def get_rounded(U, V, W):
        Ur = np.round(U * 2) / 2.0
        Vr = np.round(V * 2) / 2.0
        Wr = np.round(W * 2) / 2.0
        return Ur, Vr, Wr

    def calc_rounded_err(U, V, W):
        Ur, Vr, Wr = get_rounded(U, V, W)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += Ur[i*m+j, r] * Vr[j*p+k, r] * Wr[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    current_err = calc_rounded_err(U, V, W)
    
    # If we start with a bad solution, we might need to restart.
    # Let's try a few restarts.
    
    for restart in range(50):
        if time.time() - start_time > time_budget or deadline_exceeded():
            break
            
        # Reset to random if stuck or at start
        if restart > 0:
             # Re-randomize
             U = np.zeros((n_u, R))
             V = np.zeros((n_v, R))
             W = np.zeros((n_w, R))
             for r in range(R):
                nz_u = np.random.choice([2, 3])
                idx_u = np.random.choice(n_u, nz_u, replace=False)
                U[idx_u, r] = np.random.choice([-1, 1], nz_u)
                nz_v = np.random.choice([2, 3])
                idx_v = np.random.choice(n_v, nz_v, replace=False)
                V[idx_v, r] = np.random.choice([-1, 1], nz_v)
                nz_w = np.random.choice([2, 3])
                idx_w = np.random.choice(n_w, nz_w, replace=False)
                W[idx_w, r] = np.random.choice([-1, 1], nz_w)
             current_err = calc_rounded_err(U, V, W)
        
        # Hill Climbing
        for step in range(1000):
            if time.time() - start_time > time_budget or deadline_exceeded():
                break
                
            # Pick a random entry to flip or change
            # We only care about entries that are non-zero or close to being non-zero?
            # Let's just pick a random matrix, row, col
            mat_idx = np.random.choice(3)
            if mat_idx == 0:
                row, col = np.random.choice(n_u), np.random.choice(R)
                # Flip sign or change value
                if U[row, col] != 0:
                    U[row, col] = -U[row, col]
                else:
                    U[row, col] = np.random.choice([-1, 1])
            elif mat_idx == 1:
                row, col = np.random.choice(n_v), np.random.choice(R)
                if V[row, col] != 0:
                    V[row, col] = -V[row, col]
                else:
                    V[row, col] = np.random.choice([-1, 1])
            else:
                row, col = np.random.choice(n_w), np.random.choice(R)
                if W[row, col] != 0:
                    W[row, col] = -W[row, col]
                else:
                    W[row, col] = np.random.choice([-1, 1])
            
            new_err = calc_rounded_err(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-6:
                    # Check exact
                    Ur, Vr, Wr = get_rounded(U, V, W)
                    if _verify_exact(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
            else:
                # Revert
                if mat_idx == 0:
                    if U[row, col] == 0:
                        U[row, col] = 0
                    else:
                        U[row, col] = -U[row, col]
                elif mat_idx == 1:
                    if V[row, col] == 0:
                        V[row, col] = 0
                    else:
                        V[row, col] = -V[row, col]
                else:
                    if W[row, col] == 0:
                        W[row, col] = 0
                    else:
                        W[row, col] = -W[row, col]
                        
    return None

def construct():
    n, m, p = DIMENSIONS
    # First try rank 16
    res16 = _search_rank16(n, m, p, time_budget=110.0)
    if res16:
        return res16
        
    # Fallback to rank 17
    res17 = _construct_rank17(n, m, p)
    if res17:
        return res17
        
    return _schoolbook(n, m, p)

def main():
    return construct()

if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
