"""Improved program: top-level Strassen with inner Strassen+schoolbook for 4x4*4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check ones
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=5000, tol=1e-9):
    """Try to reduce rank by zeroing columns and re-solving via least squares."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the full tensor constraint system
    # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # For each (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i' and (i',j',k') != (i,j,k) with matching constraints: sum = 0
    
    # We'll try greedy column elimination
    for attempt in range(rank):
        # Try removing each column and see if we can solve
        for col_to_remove in range(rank):
            # Remove this column
            mask = [r for r in range(rank) if r != col_to_remove]
            U_r = U[:, mask]
            V_r = V[:, mask]
            W_r = W[:, mask]
            
            # Check if this reduced set still satisfies constraints
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol):
                U, V, W = U_r, V_r, W_r
                rank = U.shape[1]
                break
        else:
            break
    
    return U, V, W

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Top-level Strassen (7 terms) where each term is a (2x2) x (2x3) product.
    For each (2x2)x(2x3): use inner Strassen for the 2x2x2 part (7 mults) + 
    schoolbook for the 2x2x1 part (4 mults) = 11 mults per term.
    Total rank: 7 * 11 = 77.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    
    # Top-level Strassen coefficients
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                coeff_a = inner_alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b_idx * 3 + zb) * 4 + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx * 3 + 2) * 4 + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank via column elimination
    if time.time() < deadline - 10:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p)
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = np.zeros((n * m, rank))
    V_fb = np.zeros((m * p, rank))
    W_fb = np.zeros((p * n, rank))
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col_fb] = 1
                V_fb[j * p + k, col_fb] = 1
                W_fb[k * n + i, col_fb] = 1
                col_fb += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
