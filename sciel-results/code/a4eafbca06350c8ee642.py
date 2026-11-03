"""Randomized tensor rank search with integer rounding for exact decompositions."""
import time
import numpy as np
from fractions import Fraction

n, m, p = 2, 4, 6
_START = time.time()
_DEADLINE = _START + 145.0
T = None


def _elapsed():
    return time.time() - _START


def _build_tensor():
    """Build the 2x4x4x6 multiplication tensor."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _res(U, V, W):
    """Compute residual norm of decomposition."""
    return np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)


def _check(U, V, W, tol=1e-9):
    if _elapsed() > _DEADLINE:
        return False
    return _res(U, V, W) < tol


def _strassen_2x2():
    """Classic Strassen 2x2x2 decomposition with rank 7."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0]=1;U[3,0]=1;V[0,0]=1;V[3,0]=1;W[0,0]=1;W[3,0]=1
    # M2 = (A21+A22)B11
    U[1,1]=1;U[2,1]=1;V[0,1]=1;W[1,1]=1;W[2,1]=1
    # M3 = A11(B12-B22)
    U[0,2]=1;V[1,2]=1;V[3,2]=-1;W[1,2]=1
    # M4 = A22(B21-B11)
    U[3,3]=1;V[2,3]=1;V[0,3]=-1;W[0,3]=-1;W[2,3]=1
    # M5 = (A11+A12)B22
    U[0,4]=1;U[1,4]=1;V[3,4]=1;W[0,4]=1;W[1,4]=1
    # M6 = (A21-A11)(B11+B12)
    U[2,5]=1;U[0,5]=-1;V[0,5]=1;V[1,5]=1;W[0,5]=1
    # M7 = (A12-A22)(B21+B22)
    U[1,6]=1;U[3,6]=-1;V[2,6]=1;V[3,6]=1;W[3,6]=1
    return U, V, W


def _base_42():
    """Construct the 42-rank decomposition using Strassen blocks."""
    Us, Vs, Ws = _strassen_2x2()
    R = 42
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    
    U0=[0,1,4,5]; U1=[2,3,6,7]
    VB=[[0,1,6,7],[2,3,8,9],[4,5,10,11],
        [12,13,18,19],[14,15,20,21],[16,17,22,23]]
    WC=[[0,2,1,3],[4,6,5,7],[8,10,9,11]]
    cfg=[(U0,VB[0],WC[0],0),(U1,VB[3],WC[0],7),
         (U0,VB[1],WC[1],14),(U1,VB[4],WC[1],21),
         (U0,VB[2],WC[2],28),(U1,VB[5],WC[2],35)]
    for ur,vr,wr,off in cfg:
        for r in range(7):
            for idx in range(4):
                U[ur[idx],off+r]=Us[idx,r]
                V[vr[idx],off+r]=Vs[idx,r]
                W[wr[idx],off+r]=Ws[idx,r]
    return U, V, W


def _alternating(U, V, W, iters=200, tol=1e-9):
    """Alternating least squares to drive residual to zero for fixed R."""
    for it in range(iters):
        if _elapsed() > _DEADLINE - 5:
            break
        # Update U
        VW = np.einsum('br,cr->bcr', V, W)
        for a in range(n*m):
            Aa = VW.transpose(2, 0, 1).reshape(VW.shape[2], -1)
            ta = T[a].reshape(-1)
            U[a] = np.linalg.lstsq(Aa, ta, rcond=None)[0]
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)
        for b in range(m*p):
            Ab = UW.transpose(2, 0, 1).reshape(UW.shape[2], -1)
            tb = T[:, b, :].reshape(-1)
            V[b] = np.linalg.lstsq(Ab, tb, rcond=None)[0]
        # Update W
        UV = np.einsum('ar,br->abr', U, V)
        for c in range(n*p):
            Ac = UV.transpose(2, 0, 1).reshape(UV.shape[2], -1)
            tc = T[:, :, c].reshape(-1)
            W[c] = np.linalg.lstsq(Ac, tc, rcond=None)[0]
        if _res(U, V, W) < tol:
            break
    return U, V, W


def _round_to_half_inters(X):
    """Round entries to nearest half-integer."""
    return np.round(2 * X) / 2.0


def _try_rank_search(R, max_trials=50, seed=42):
    """
    Try to find an exact rank-R decomposition using random initialization
    and alternating least squares, with integer/half-integer rounding.
    """
    if _elapsed() > _DEADLINE - 10:
        return None
    
    rng = np.random.default_rng(seed)
    best_res = float('inf')
    best_result = None
    
    for trial in range(max_trials):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Random initialization with small values
        U = rng.standard_normal((n*m, R)) * 0.5
        V = rng.standard_normal((m*p, R)) * 0.5
        W = rng.standard_normal((p*n, R)) * 0.5
        
        # Run alternating LS
        U, V, W = _alternating(U, V, W, iters=150, tol=1e-9)
        res = _res(U, V, W)
        
        if res < best_res:
            best_res = res
            best_result = (U.copy(), V.copy(), W.copy())
        
        if res < 1e-9:
            # Try rounding to half-integers
            Ur = _round_to_half_inters(U)
            Vr = _round_to_half_inters(V)
            Wr = _round_to_half_inters(W)
            if _check(Ur, Vr, Wr):
                return (Ur, Vr, Wr)
            # Also try integer rounding
            Ui = np.round(U)
            Vi = np.round(V)
            Wi = np.round(W)
            if _check(Ui, Vi, Wi):
                return (Ui, Vi, Wi)
            # Return the unrounded version if it's exact enough
            return (U, V, W)
    
    return best_result


def _try_structured_reduction(R, max_trials=30, seed=123):
    """
    Try to reduce rank by starting from the 42-rank base and projecting
    to lower dimensions with structured perturbations.
    """
    if _elapsed() > _DEADLINE - 10:
        return None
    
    U42, V42, W42 = _base_42()
    rng = np.random.default_rng(seed)
    
    for trial in range(max_trials):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Take first R columns and add random perturbation
        U = U42[:, :R].copy() + rng.standard_normal((n*m, R)) * 0.1
        V = V42[:, :R].copy() + rng.standard_normal((m*p, R)) * 0.1
        W = W42[:, :R].copy() + rng.standard_normal((p*n, R)) * 0.1
        
        # Run alternating LS
        U, V, W = _alternating(U, V, W, iters=150, tol=1e-9)
        res = _res(U, V, W)
        
        if res < 1e-9:
            # Try rounding
            Ur = _round_to_half_inters(U)
            Vr = _round_to_half_inters(V)
            Wr = _round_to_half_inters(W)
            if _check(Ur, Vr, Wr):
                return (Ur, Vr, Wr)
            Ui = np.round(U)
            Vi = np.round(V)
            Wi = np.round(W)
            if _check(Ui, Vi, Wi):
                return (Ui, Vi, Wi)
            return (U, V, W)
    
    return None


def _schoolbook():
    """Fallback schoolbook decomposition with rank 48."""
    R0 = n*m*p
    U = np.zeros((n*m, R0))
    V = np.zeros((m*p, R0))
    W = np.zeros((p*n, R0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def construct():
    global T
    T = _build_tensor()
    
    # Start with the 42-rank base
    U42, V42, W42 = _base_42()
    best_rank = 42
    best = (U42.copy(), V42.copy(), W42.copy())
    
    if not _check(U42, V42, W42):
        Uf, Vf, Wf = _schoolbook()
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    # Try to find lower rank decompositions
    # The lower bound is at least max(matrix ranks)
    M1 = T.reshape(n*m, -1)
    M2 = T.transpose(1,0,2).reshape(m*p, -1)
    M3 = T.transpose(2,0,1).reshape(n*p, -1)
    lower_bound = max(np.linalg.matrix_rank(M1), np.linalg.matrix_rank(M2), np.linalg.matrix_rank(M3))
    
    search_start = max(lower_bound, 28)
    
    for R in range(search_start, 42):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Try structured reduction first (faster)
        result = _try_structured_reduction(R, max_trials=20, seed=100+R)
        if result is not None:
            best_rank = R
            best = result
            break
        
        # Try random search
        result = _try_rank_search(R, max_trials=30, seed=200+R)
        if result is not None:
            best_rank = R
            best = result
            break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
