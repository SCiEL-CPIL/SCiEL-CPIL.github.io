import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a low-rank decomposition
    # Start with a known good decomposition (Strassen-based, rank 77)
    # Then try to improve
    
    # First, let's construct the Strassen-based decomposition as fallback
    R_fallback = 77
    U_fallback = np.zeros((n * m, R_fallback))
    V_fallback = np.zeros((m * p, R_fallback))
    W_fallback = np.zeros((p * n, R_fallback))
    
    alpha_s = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta_s = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                ia = alpha_s[si].get(a, 0)
                if ia != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        sa = alpha_s[s].get(r, 0)
                        if sa != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U_fallback[u_idx, col] += ia * sa
            
            for b in range(4):
                ib = beta_s[si].get(b, 0)
                if ib != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        sb = beta_s[s].get(c, 0)
                        if sb != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V_fallback[v_idx, col] += ib * sb
            
            for a, b, gamma in recon_s[s]:
                for za, zb, zgamma in recon_s[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W_fallback[w_idx, col] += gamma * zgamma
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    sa = alpha_s[s].get(r, 0)
                    if sa != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U_fallback[u_idx, col] += sa
                for c in range(4):
                    sb = beta_s[s].get(c, 0)
                    if sb != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V_fallback[v_idx, col] += sb
                for a, b, gamma in recon_s[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W_fallback[w_idx, col] += gamma
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                r = round(x)
                if abs(x - r) < 1e-9:
                    new_row.append(int(r))
                else:
                    r2 = round(2 * x) / 2
                    if abs(x - r2) < 1e-9:
                        new_row.append(r2)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result
    
    # Return the fallback for now
    return {"U": to_list(U_fallback), "V": to_list(V_fallback), "W": to_list(W_fallback)}
