import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    for c, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_steps):
        for (li, lj), val in u_coeffs.items():
            U[li * m + lj][c] += val
        for (lj, lk), val in v_coeffs.items():
            V[lj * p + lk][c] += val
        for (li, lk), val in w_coeffs.items():
            W[lk * n + li][c] += val
            
    return {"U": U, "V": V, "W": W}

def _strassen_44():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    s2 = _strassen_2x2x2()
    sU = s2["U"]; sV = s2["V"]; sW = s2["W"]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    if sU[li*2+lj][c] != 0:
                        U[(li+xi)*m + (lj+xj)][col] += sU[li*2+lj][c]
            for lj in range(2):
                for lk in range(2):
                    if sV[lj*2+lk][c] != 0:
                        V[(lj+yj)*p + (lk+yk)][col] += sV[lj*2+lk][c]
            for li in range(2):
                for lk in range(2):
                    if sW[lk*2+li][c] != 0:
                        W[(lk+zk)*n + (li+zi)][col] += sW[lk*2+li][c]
            col += 1
            
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _try_remove_columns(U, V, W, n, m, p, tol=1e-9, max_removals=5):
    """Try removing columns one at a time, greedily, until no more can be removed."""
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    rank = len(current_U[0])
    removed = 0
    
    while removed < max_removals:
        found = False
        for c_remove in range(rank - removed):
            # Build reduced matrices
            U_new = [row[:c_remove] + row[c_remove+1:] for row in current_U]
            V_new = [row[:c_remove] + row[c_remove+1:] for row in current_V]
            W_new = [row[:c_remove] + row[c_remove+1:] for row in current_W]
            
            ok, err = _verify_fast(U_new, V_new, W_new, n, m, p, tol)
            if ok:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                rank = len(current_U[0])
                removed += 1
                found = True
                break
        
        if not found:
            break
    
    if removed > 0:
        return {"U": current_U, "V": current_V, "W": current_W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if ok:
        # Try to reduce rank by removing linearly dependent columns
        reduced = _try_remove_columns(result_44["U"], result_44["V"], result_44["W"], n, m, p)
        if reduced is not None:
            return reduced
        return result_44
    
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    ok, err = _verify_fast(res['U'], res['V'], res['W'], 3, 4, 4)
    print("Valid:", ok, "Err:", err)
