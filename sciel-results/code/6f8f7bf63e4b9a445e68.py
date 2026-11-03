import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_2x2_coeffs():
    """Return Strassen coefficients for 2x2 x 2x2."""
    alpha = [
        [1, 0, 0, 1],   # s=0: A00 + A11
        [0, 0, 1, 1],   # s=1: A10 + A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00 + A01
        [-1, 0, 1, 0],  # s=5: A10 - A00
        [0, 1, 0, -1]   # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s=0: B00 + B11
        [1, 0, 0, 0],   # s=1: B00
        [0, 1, 0, -1],  # s=2: B01 - B11
        [-1, 0, 1, 0],  # s=3: B10 - B00
        [0, 0, 0, 1],   # s=4: B11
        [1, 1, 0, 0],   # s=5: B00 + B01
        [0, 0, 1, 1]    # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    return alpha, beta, recon

def _build_77_rank():
    """Build the 77-rank two-level Strassen decomposition for 4x4 x 4x6."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    alpha, beta, recon = _strassen_2x2_coeffs()
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # U: combine X_s via inner_alpha[si]
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += ca * cr
            
            # V: combine Y_s via inner_beta[si]
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += cb * cc
            
            # W: output Z[i,j] for j in {0,1}
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += cr
                
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += cc
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, R, num_trials=5):
    """Fast vectorized verification of the tensor decomposition."""
    for trial in range(num_trials):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        # Tensor: C[i,r] = sum_k sum_t U[i*m+k, t] * V[k*p+r, t] * W[r*n+i, t]
        # Vectorize over t: for each (i,k,r), accumulate U[i*m+k,:] * V[k*p+r,:] * W[r*n+i,:]
        C_recon = np.zeros((n, p), dtype=float)
        for i in range(n):
            for r in range(p):
                for k in range(m):
                    # Element-wise product of three vectors of length R, then sum
                    C_recon[i, r] += np.sum(U[i*m+k, :] * V[k*p+r, :] * W[r*n+i, :])
        
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True

def _build_schoolbook():
    """Fallback: schoolbook rank-96 decomposition."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for k in range(m):
            for r in range(p):
                U[i * m + k, idx] = 1
                V[k * p + r, idx] = 1
                W[r * n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try 77-rank Strassen decomposition
    try:
        U, V, W = _build_77_rank()
        R = U.shape[1]
        
        # Quick verification
        if time.time() - START_TIME < TIME_LIMIT - 20:
            if _verify_fast(U, V, W, n, m, p, R, num_trials=3):
                # Success! Return the 77-rank decomposition
                def to_list(mat):
                    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat.tolist()]
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        
        # Verification failed or time up, fall through
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat.tolist()]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
