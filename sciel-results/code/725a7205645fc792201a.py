"""Matrix multiplication 2x3x8: find low-rank exact decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
DEADLINE = 150.0
SAFETY = 10.0

def _time_left():
    return DEADLINE - SAFETY - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p):
    """Exact verification using Fraction for exactness."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # Convert to Fraction for exact arithmetic
    U_frac = np.zeros((U.shape[0], rank), dtype=object)
    V_frac = np.zeros((V.shape[0], rank), dtype=object)
    W_frac = np.zeros((W.shape[0], rank), dtype=object)
    for i in range(U.shape[0]):
        for r in range(rank):
            U_frac[i, r] = Fraction(U[i, r])
    for i in range(V.shape[0]):
        for r in range(rank):
            V_frac[i, r] = Fraction(V[i, r])
    for i in range(W.shape[0]):
        for r in range(rank):
            W_frac[i, r] = Fraction(W[i, r])
    
    # Build tensor as dict of nonzeros
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[(a, b, c_idx)] = 1
    
    # Check all support entries
    for (a, b, c_idx), val in T.items():
        s = Fraction(0)
        for r in range(rank):
            s += U_frac[a, r] * V_frac[b, r] * W_frac[c_idx, r]
        if s != val:
            return False
    
    # Check all zero entries
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                if (a, b, c_idx) in T:
                    continue
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[a, r] * V_frac[b, r] * W_frac[c_idx, r]
                if s != 0:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification for preliminary checks."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[(a, b, c_idx)] = 1.0
    
    # Check support entries
    for (a, b, c_idx), val in T.items():
        s = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
        if abs(s - val) > tol:
            return False
    
    # Check zero entries (sample if too many)
    total_zeros = n*m*m*p*p*n - len(T)
    if total_zeros > 10000:
        # Sample
        import random
        sampled = 0
        while sampled < 10000:
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c_idx = random.randint(0, p*n-1)
            if (a, b, c_idx) in T:
                continue
            s = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
            if abs(s) > tol:
                return False
            sampled += 1
    else:
        for a in range(n * m):
            for b in range(m * p):
                for c_idx in range(p * n):
                    if (a, b, c_idx) in T:
                        continue
                    s = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
                    if abs(s) > tol:
                        return False
    return True

def _round_to_half_integer(U, V, W):
    """Round entries to nearest half-integer (multiple of 0.5)."""
    U_r = [[round(u * 2) / 2 for u in row] for row in U]
    V_r = [[round(v * 2) / 2 for v in row] for row in V]
    W_r = [[round(w * 2) / 2 for w in row] for row in W]
    return U_r, V_r, W_r

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    terms = []
    terms.append(([(1,0),(1,3)], [(1,0),(1,3)], [(1,0),(1,3)]))
    terms.append(([(1,2),(1,3)], [(1,0)], [(1,2),(1,3)]))
    terms.append(([(1,0)], [(1,2),(-1,3)], [(1,1),(1,2)]))
    terms.append(([(1,3)], [(-1,1),(1,2)], [(1,0),(1,2)]))
    terms.append(([(1,0),(1,1)], [(1,3)], [(1,1),(1,3)]))
    terms.append(([(1,2),(-1,0)], [(1,0),(1,2)], [(1,3)]))
    terms.append(([(1,1),(-1,3)], [(1,1),(1,3)], [(1,0),(1,3)]))
    return terms

def _build_2x3x2_block():
    blocks = []
    for i in range(2):
        for k in range(2):
            U = [0]*6; V = [0]*6; W = [0]*4
            U[i*3+0] = 1
            V[0*2+k] = 1
            W[k*2+i] = 1
            blocks.append((U, V, W))
    for (a_list, b_list, c_list) in _strassen_2x2x2():
        U = [0]*6; V = [0]*6; W = [0]*4
        for (coef, (si, sj)) in a_list:
            U[si*3 + (1+sj)] += coef
        for (coef, (sj, sk)) in b_list:
            V[(1+sj)*2 + sk] += coef
        for (coef, (sk, si)) in c_list:
            W[sk*2 + si] += coef
        blocks.append((U, V, W))
    return blocks

def _build_44(n, m, p):
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for b in range(p // 2):
        for (Ub, Vb, Wb) in _build_2x3x2_block():
            for a in range(n*m):
                if Ub[a]:
                    U[a][col] += Ub[a]
            for j in range(m):
                for k in range(2):
                    if Vb[j*2+k]:
                        V[j*p + (2*b+k)][col] += Vb[j*2+k]
            for k in range(2):
                for i in range(n):
                    if Wb[k*2+i]:
                        W[(2*b+k)*n + i][col] += Wb[k*2+i]
            col += 1
    return {"U": U, "V": V, "W": W}

def _als_decompose(n, m, p, target_rank, max_iters=200, tol=1e-10, seed=42):
    """ALS-based tensor decomposition to find rank-target_rank decomposition."""
    rng = np.random.RandomState(seed)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
    
    # Initialize randomly
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Update U: for each a, solve least squares
        for a in range(n*m):
            # Build matrix: V^T ⊗ W^T applied to T[a,:,:]
            # T[a,:,:] reshaped as (m*p * p*n), and we want V[b,r]*W[c,r] for all b,c
            # This is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So we solve: [V[:,0]⊗W[:,0], V[:,1]⊗W[:,1], ...] * U[a,:] = T[a,:,:].ravel()
            A_mat = np.zeros((m*p * p*n, target_rank))
            for r in range(target_rank):
                A_mat[:, r] = np.kron(V[:, r], W[:, r])
            b_vec = T[a, :, :].ravel()
            U[a, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update V
        for b in range(m*p):
            A_mat = np.zeros((n*m * p*n, target_rank))
            for r in range(target_rank):
                A_mat[:, r] = np.kron(U[:, r], W[:, r])
            b_vec = T[:, b, :].ravel()
            V[b, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Update W
        for c_idx in range(p*n):
            A_mat = np.zeros((n*m * m*p, target_rank))
            for r in range(target_rank):
                A_mat[:, r] = np.kron(U[:, r], V[:, r])
            b_vec = T[:, :, c_idx].ravel()
            W[c_idx, :], residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Compute error
        err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c_idx in range(p*n):
                    pred = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
                    err += (pred - T[a, b, c_idx]) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank40(n, m, p):
    """Try to find a rank-40 decomposition using ALS."""
    if _time_left() < 5.0:
        return None
    
    # Use multiple random initializations
    for seed in range(10):
        if _time_left() < 3.0:
            break
        
        U, V, W, err = _als_decompose(n, m, p, 40, max_iters=100, seed=seed)
        
        if err < 1e-8:
            # Round to half-integers and verify
            U_r, V_r, W_r = _round_to_half_integer(U.tolist(), V.tolist(), W.tolist())
            if _verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try the 44 construction
    try:
        cand = _build_44(n, m, p)
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            best = cand
    except Exception:
        pass
    
    if _time_left() <= 0:
        return best
    
    # Try to find rank-40
    try:
        cand40 = _try_rank40(n, m, p)
        if cand40 is not None:
            best = cand40
    except Exception:
        pass
    
    return best
