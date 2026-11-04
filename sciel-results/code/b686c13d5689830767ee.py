"""Find rank-minimal tensor decomposition for 3x3 @ 3x7 matrix multiplication."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _build_tensor():
    """Build the 21x21x21 multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W):
    """Verify that U,V,W exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    # Check against the true multiplication tensor
    TT = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TT[i * m + j, j * p + k, k * n + i] = 1
    return np.array_equal(T, TT)

def _to_halfint(x):
    """Convert a float to an integer or half-integer Fraction if close enough."""
    # Try integer
    xi = int(round(x))
    if abs(x - xi) < 1e-9:
        return Fraction(xi, 1)
    # Try half-integer
    xh = int(round(2 * x)) / 2
    if abs(x - xh) < 1e-9:
        return Fraction(int(xh * 2), 2)
    return None

def _round_matrix(M, tol=1e-8):
    """Try to round a matrix to integer/half-integer values."""
    r, c = M.shape
    out = np.zeros((r, c), dtype=object)
    for i in range(r):
        for j in range(c):
            v = _to_halfint(M[i, j])
            if v is None:
                return None
            out[i, j] = v
    return out

def _matmul_decompose(U, V, W):
    """Verify decomposition and return as list of lists of ints/half-ints."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U_frac = _round_matrix(U)
    V_frac = _round_matrix(V)
    W_frac = _round_matrix(W)
    if U_frac is None or V_frac is None or W_frac is None:
        return None
    # Verify with exact arithmetic
    U_int = np.zeros((U.shape[0], r), dtype=np.int64)
    V_int = np.zeros((V.shape[0], r), dtype=np.int64)
    W_int = np.zeros((W.shape[0], r), dtype=np.int64)
    scale = 1
    for c in range(r):
        for i in range(U.shape[0]):
            U_int[i, c] = int(U_frac[i, c] * 2)
        for i in range(V.shape[0]):
            V_int[i, c] = int(V_frac[i, c] * 2)
        for i in range(W.shape[0]):
            W_int[i, c] = int(W_frac[i, c] * 2)
    # Check: sum_c U[:,c]*V[:,c]*W[:,c] should equal tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U_int[:, c], V_int[:, c], W_int[:, c])
    # Expected tensor (scaled by 2^3 = 8 since we doubled each)
    TT = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TT[i * m + j, j * p + k, k * n + i] = 8
    if not np.array_equal(T, TT):
        return None
    # Convert back to list of lists of Fraction
    U_list = [[str(U_frac[i, c]) for c in range(r)] for i in range(U.shape[0])]
    V_list = [[str(V_frac[i, c]) for c in range(r)] for i in range(V.shape[0])]
    W_list = [[str(W_frac[i, c]) for c in range(r)] for i in range(W.shape[0])]
    return U_list, V_list, W_list

def _randomized_cp(T, r, iters=50, seed=42):
    """
    Attempt to find a rank-r CP decomposition of tensor T using randomized 
    iterative approach. T has shape (a, b, c).
    """
    a, b, c = T.shape
    rng = np.random.default_rng(seed)
    best = None
    best_err = np.inf
    for it in range(iters):
        U = rng.standard_normal((a, r))
        V = rng.standard_normal((b, r))
        W = rng.standard_normal((c, r))
        for inner in range(200):
            # Update U
            T_mat = np.zeros((a, r))
            for k in range(c):
                T_mat += np.einsum('ak,ak->ak', T[:, :, k], W[k, :]) @ V.T
            U = T_mat / (np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0))
            if not np.all(np.isfinite(U)):
                break
            # Update V
            T_mat = np.zeros((b, r))
            for k in range(c):
                T_mat += np.einsum('bk,bk->bk', T[:, k, :], W[k, :]) @ U.T
            V = T_mat / (np.linalg.norm(U, axis=0) * np.linalg.norm(W, axis=0))
            if not np.all(np.isfinite(V)):
                break
            # Update W
            T_mat = np.zeros((c, r))
            for j in range(b):
                T_mat += np.einsum('jk,jk->jk', T[:, j, :], U[:, j]) @ V.T
            W = T_mat / (np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0))
            if not np.all(np.isfinite(W)):
                break
        # Check error
        T_recon = np.zeros((a, b, c))
        for k in range(r):
            T_recon += np.einsum('i,j->ij', U[:, k], V[:, k])
            T_recon = T_recon * W[:, k]  # This is wrong, let me fix
        # Redo properly
        T_recon = np.zeros((a, b, c))
        for k in range(r):
            T_recon += np.einsum('i,j,l->ijl', U[:, k], V[:, k], W[:, k])
        err = np.linalg.norm(T_recon - T)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-6:
            break
    return best, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # First, try the schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    school_result = (U_school, V_school, W_school)
    
    best_result = school_result
    best_rank = rank_school
    
    # Try lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we need at least the "support" rank. For matmul, known lower bounds apply.
    # For 3x3 x 3x3, best known is 23 (Landsberg). For 3x3 x 3x7, let's search.
    
    # Try ranks from 21 upward
    for r in range(21, rank_school):
        if time.time() > DEADLINE - 10:
            break
        # Try randomized CP with several seeds
        found = False
        for seed in range(5):
            if time.time() > DEADLINE - 5:
                break
            try:
                result, err = _randomized_cp(T, r, iters=30, seed=seed)
                if result is not None and err < 1e-6:
                    U, V, W = result
                    # Try to round to half-integers
                    mat_result = _matmul_decompose(U, V, W)
                    if mat_result is not None:
                        best_result = (U, V, W)
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        if found:
            break
    
    # Final verification
    U, V, W = best_result
    # Convert to the required format
    r = U.shape[1]
    U_list = [[int(U[i, c]) for c in range(r)] for i in range(U.shape[0])]
    V_list = [[int(V[i, c]) for c in range(r)] for i in range(V.shape[0])]
    W_list = [[int(W[i, c]) for c in range(r)] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
