# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # seconds safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where u,v,w are 4-dim vectors (indices 0..3 for 2x2).
    u[i,j], v[j,k], w[k,i] for i,j,k in {0,1}.
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A21+A22)B11
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]))
    # M3 = A11(B21-B22)
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    # M4 = A22(B12-B11)
    terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]))
    # M5 = (A11+A12)B22
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]))
    # M6 = (A21-A11)(B11+B12)
    terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A12-A22)(B21+B22)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return terms


def _rank26_decomposition():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    
    Partition:
    A = [[A11, a1], [a2, a3]]  where A11 is 2x2, a1 is 2x1, a2 is 1x2, a3 is 1x1
    B = [[B11, b1], [b2, b3]]  where B11 is 2x2, b1 is 2x1, b2 is 1x2, b3 is 1x1
    
    C = A*B:
    C11 = A11*B11 + a1*b2   (2x2)
    C12 = A11*b1 + a1*b3    (2x1)
    C21 = a2*B11 + a3*b2    (1x2)
    C22 = a2*b1 + a3*b3     (1x1)
    
    Use Strassen (rank 7) for A11*B11.
    Schoolbook for all other terms:
    - a1*b2: 2*1*2 = 4 terms
    - A11*b1: 2*2*1 = 4 terms
    - a1*b3: 2*1*1 = 2 terms
    - a2*B11: 1*2*2 = 4 terms
    - a3*b2: 1*1*2 = 2 terms
    - a2*b1: 1*2*1 = 2 terms
    - a3*b3: 1*1*1 = 1 term
    Total schoolbook: 4+4+2+4+2+2+1 = 19
    Total rank: 7 + 19 = 26
    """
    n, m, p = 3, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    rank = 26
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    # Strassen for A11*B11 (indices: A rows 0,1; cols 0,1; B rows 0,1; cols 0,1)
    strassen_terms = _strassen_2x2()
    for u_s, v_s, w_s in strassen_terms:
        # u_s: 4-dim for A[0:2, 0:2] -> indices 0,1,3,4 in 3x3 (i*3+j)
        # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        u_map = [0, 1, 3, 4]  # u_s[0]->A[0,0], u_s[1]->A[0,1], u_s[2]->A[1,0], u_s[3]->A[1,1]
        # v_s: 4-dim for B[0:2, 0:2] -> indices 0,1,3,4 in 3x3 (j*3+k)
        # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
        v_map = [0, 1, 3, 4]
        # w_s: 4-dim for C[0:2, 0:2] -> indices 0,1,3,4 in 3x3 (k*3+i)
        # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
        w_map = [0, 1, 3, 4]
        
        for idx, val in enumerate(u_s):
            if val != 0:
                U[u_map[idx], col] = val
        for idx, val in enumerate(v_s):
            if val != 0:
                V[v_map[idx], col] = val
        for idx, val in enumerate(w_s):
            if val != 0:
                W[w_map[idx], col] = val
        col += 1
    
    # Now schoolbook for cross-block terms:
    # a1*b2: A[0:2, 2] * B[2, 0:2]
    # A[0,2]*B[2,0] -> C[0,0]
    U[2, col] = 1.0; V[6, col] = 1.0; W[0, col] = 1.0
    col += 1
    # A[0,2]*B[2,1] -> C[0,1]
    U[2, col] = 1.0; V[7, col] = 1.0; W[1, col] = 1.0
    col += 1
    # A[1,2]*B[2,0] -> C[1,0]
    U[5, col] = 1.0; V[6, col] = 1.0; W[3, col] = 1.0
    col += 1
    # A[1,2]*B[2,1] -> C[1,1]
    U[5, col] = 1.0; V[7, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # A11*b1: A[0:2, 0:2] * B[0:2, 2]
    # A[0,0]*B[0,2] -> C[0,2]
    U[0, col] = 1.0; V[2, col] = 1.0; W[2, col] = 1.0
    col += 1
    # A[0,1]*B[1,2] -> C[0,2]
    U[1, col] = 1.0; V[5, col] = 1.0; W[2, col] = 1.0
    col += 1
    # A[1,0]*B[0,2] -> C[1,2]
    U[3, col] = 1.0; V[2, col] = 1.0; W[5, col] = 1.0
    col += 1
    # A[1,1]*B[1,2] -> C[1,2]
    U[4, col] = 1.0; V[5, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    # a1*b3: A[0:2, 2] * B[2, 2]
    # A[0,2]*B[2,2] -> C[0,2]
    U[2, col] = 1.0; V[8, col] = 1.0; W[2, col] = 1.0
    col += 1
    # A[1,2]*B[2,2] -> C[1,2]
    U[5, col] = 1.0; V[8, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    # a2*B11: A[2, 0:2] * B[0:2, 0:2]
    # A[2,0]*B[0,0] -> C[2,0]
    U[6, col] = 1.0; V[0, col] = 1.0; W[6, col] = 1.0
    col += 1
    # A[2,0]*B[0,1] -> C[2,1]
    U[6, col] = 1.0; V[1, col] = 1.0; W[7, col] = 1.0
    col += 1
    # A[2,1]*B[1,0] -> C[2,0]
    U[7, col] = 1.0; V[3, col] = 1.0; W[6, col] = 1.0
    col += 1
    # A[2,1]*B[1,1] -> C[2,1]
    U[7, col] = 1.0; V[4, col] = 1.0; W[7, col] = 1.0
    col += 1
    
    # a3*b2: A[2,2] * B[2, 0:2]
    # A[2,2]*B[2,0] -> C[2,0]
    U[8, col] = 1.0; V[6, col] = 1.0; W[6, col] = 1.0
    col += 1
    # A[2,2]*B[2,1] -> C[2,1]
    U[8, col] = 1.0; V[7, col] = 1.0; W[7, col] = 1.0
    col += 1
    
    # a2*b1: A[2, 0:2] * B[0:2, 2]
    # A[2,0]*B[0,2] -> C[2,2]
    U[6, col] = 1.0; V[2, col] = 1.0; W[8, col] = 1.0
    col += 1
    # A[2,1]*B[1,2] -> C[2,2]
    U[7, col] = 1.0; V[5, col] = 1.0; W[8, col] = 1.0
    col += 1
    
    # a3*b3: A[2,2] * B[2,2]
    # A[2,2]*B[2,2] -> C[2,2]
    U[8, col] = 1.0; V[8, col] = 1.0; W[8, col] = 1.0
    col += 1
    
    return U, V, W


def _try_solve_for_W(U, V, rank):
    """Given U (nm x rank) and V (mp x rank), solve linear system for W (pn x rank)."""
    n, m, p = 3, 3, 3
    num_vars = (p * n) * rank
    A = np.zeros((n * m * p, num_vars))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A[eq, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq += 1
    W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape((p * n, rank))
    return W


def _try_rank_fixed_UV(rank, seed=0):
    """Fix U,V as first `rank` schoolbook columns, solve for W, verify."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    if rank > U_s.shape[1]:
        return None
    U = U_s[:, :rank].copy()
    V = V_s[:, :rank].copy()
    W = _try_solve_for_W(U, V, rank)
    if _verify(U, V, W):
        return U, V, W
    return None


def _general_search(rank, seed=0):
    """
    General search allowing U,V,W to vary. Uses scipy least-squares on a nonlinear system,
    with a final exact verification.
    """
    n, m, p = 3, 3, 3
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n

    def residuals(flat):
        U = flat[:nm * rank].reshape(nm, rank)
        V = flat[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = flat[nm * rank + mp * rank:].reshape(pn, rank)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    for attempt in range(8):
        if _time_left() < 8.0:
            break
        # Initialize near schoolbook-like random
        U0 = rng.normal(size=(nm, rank))
        V0 = rng.normal(size=(mp, rank))
        W0 = rng.normal(size=(pn, rank))
        # Scale to make products ~O(1)
        scale = 1.0 / (rank ** 0.5)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        flat0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            sol = least_squares(residuals, flat0, method='lm', max_nfev=2000)
        except Exception:
            continue
        U = sol.x[:nm * rank].reshape(nm, rank)
        V = sol.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = sol.x[nm * rank + mp * rank:].reshape(pn, rank)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None


def construct():
    """
    Find a rank-minimal exact decomposition for 3x3 matrix multiplication.
    Tries decreasing ranks; returns the best valid one found before deadline.
    """
    n, m, p = 3, 3, 3

    # Fallback: schoolbook rank 27 (always valid)
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)  # rank 27

    # Try rank 26 (Strassen + schoolbook cross terms)
    if _time_left() > 5.0:
        U26, V26, W26 = _rank26_decomposition()
        if _verify(U26, V26, W26):
            best = (U26, V26, W26)

    # Try ranks from 23 down
    for rank in [23, 22, 21, 20, 19]:
        if _time_left() < 15.0:
            break
        res = _try_rank_fixed_UV(rank)
        if res is not None:
            U, V, W = res
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue
        if _time_left() < 20.0:
            break
        res2 = _general_search(rank)
        if res2 is not None:
            U, V, W = res2
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
