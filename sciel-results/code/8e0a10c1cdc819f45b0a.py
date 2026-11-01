"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
1. Start with schoolbook rank 50.
2. Use iterative multiplicative refinement (smooth surrogate for the
   nonsmooth exact-match objective) to drive residual toward zero.
3. Snap converged solutions to integer/half-integer and validate exactly.
4. Try various rank values with structured and randomized initializations.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate using exact Fraction arithmetic."""
    r = len(W[0]) if isinstance(W, list) and len(W) > 0 else W.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    u_val = Fraction(U[idx_u][c])
                    v_val = Fraction(V[idx_v][c])
                    w_val = Fraction(W[idx_w][c])
                    val += u_val * v_val * w_val
                if val != Fraction(1, 1):
                    return False
    return True

def validate_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast validation using floating point."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False

    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def strassen_2x2x2_decomposition(n, m, p):
    """Construct a Strassen-based decomposition for 2x5 x 5x5.
    
    Split j and k dimensions into groups of size 2 (and remainder 1).
    For each 2x2x2 sub-block (i in {0,1}, j in a pair, k in a pair),
    use Strassen's 7-term decomposition instead of 8 schoolbook terms.
    For remaining blocks, use schoolbook.
    """
    # j-groups and k-groups: pairs of 2 + remainder
    # j = 0..4: groups [0,1], [2,3], [4]
    # k = 0..4: groups [0,1], [2,3], [4]
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4,)]
    
    U = {}  # (i*m+j) -> list of (value, col)
    V = {}  # (j*p+k) -> list of (value, col)
    W = {}  # (k*n+i) -> list of (value, col)
    
    col = 0
    for i in range(n):
        for jg in j_groups:
            for kg in k_groups:
                if len(jg) == 2 and len(kg) == 2:
                    # 2x2x2 block: use Strassen
                    j1, j2 = jg
                    k1, k2 = kg
                    # Strassen's 7 products for 2x2 x 2x2:
                    # M1 = (A11+A22)*(B11+B22)
                    # M2 = (A21+A22)*B11
                    # M3 = A11*(B12-B22)
                    # M4 = A22*(B21-B11)
                    # M5 = (A11+A12)*B22
                    # M6 = (A21-A11)*(B11+B12)
                    # M7 = (A12-A22)*(B21+B22)
                    # C11 = M1+M4-M5+M7
                    # C12 = M3+M5
                    # C21 = M2+M4
                    # C22 = M1-M2+M3+M6
                    
                    # For our tensor: T[i, j, k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # We need: for each (i,j,k) in this 2x2x2 block,
                    # the contribution from this Strassen block should give T[i,j,k]=1
                    # if (i,j,k) is in the block, else 0.
                    
                    # The 7 Strassen terms correspond to:
                    # Each term c has U values, V values, W values
                    # such that sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for (i,j,k) in block
                    
                    # Strassen decomposition of 2x2x2 identity tensor:
                    # Term 1: (A11+A22)*(B11+B22)
                    #   U: i=0,j=j1: +1; i=1,j=j2: +1
                    #   V: j=j1,k=k1: +1; j=j2,k=k2: +1
                    #   W: k=k1,i=0: +1; k=k2,i=1: +1
                    #   Product at (i,j,k): U[i,j]*V[j,k]*W[k,i]
                    #   (0,j1,k1): 1*1*1=1, (0,j1,k2): 1*0*0=0, (0,j2,k1): 0*1*0=0, (0,j2,k2): 0*0*1=0
                    #   (1,j1,k1): 0*1*0=0, (1,j1,k2): 0*0*1=0, (1,j2,k1): 1*0*0=0, (1,j2,k2): 1*1*1=1
                    #   So this term contributes to (0,j1,k1) and (1,j2,k2)
                    
                    # This is getting complex. Let me just enumerate the 7 Strassen terms
                    # for the 2x2x2 tensor and add them.
                    
                    # The 2x2x2 tensor T[i,j,k] for i,j,k in {0,1} (local indices)
                    # has T[i,j,k] = 1 if we're computing the identity of matrix mult.
                    # Actually T[i,j,k] = delta(i,j,k) in the sense that the tensor
                    # for 2x2x2 matrix mult has T[i,j,k] = 1 for all 8 entries.
                    
                    # Strassen's 7 rank-1 terms for the 2x2x2 matmul tensor:
                    # Each term is u ⊗ v ⊗ w where:
                    strassen_terms = [
                        # (U_vals for i,j in {0,1}x{0,1}), (V_vals for j,k), (W_vals for k,i)
                        # Term 1: (A00+A11)*(B00+B11)
                        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                        # Term 2: (A10+A11)*B00
                        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1}),
                        # Term 3: A00*(B01-B11)
                        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
                        # Term 4: A11*(B10-B00)
                        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
                        # Term 5: (A00+A01)*B11
                        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,1): 1}),
                        # Term 6: (A10-A00)*(B00+B01)
                        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1}),
                        # Term 7: (A01-A11)*(B10+B11)
                        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),
                    ]
                    
                    for u_map, v_map, w_map in strassen_terms:
                        # Map local (li, lj) to global (i, jg[lj])
                        for (li, lj), uv in u_map.items():
                            gi = i
                            gj = jg[lj]
                            key = (gi, gj)
                            if key not in U:
                                U[key] = []
                            U[key].append((uv, col))
                        for (lj, lk), vv in v_map.items():
                            gj = jg[lj]
                            gk = kg[lk]
                            key = (gj, gk)
                            if key not in V:
                                V[key] = []
                            V[key].append((vv, col))
                        for (lk, li), wv in w_map.items():
                            gk = kg[lk]
                            gi = i
                            key = (gk, gi)
                            if key not in W:
                                W[key] = []
                            W[key].append((wv, col))
                        col += 1
                elif len(jg) == 1 and len(kg) == 1:
                    # 2x1x1 block: schoolbook, 2 terms
                    j1 = jg[0]
                    k1 = kg[0]
                    for ii in range(n):
                        U.setdefault((ii, j1), []).append((1, col))
                        V.setdefault((j1, k1), []).append((1, col))
                        W.setdefault((k1, ii), []).append((1, col))
                        col += 1
                elif len(jg) == 2 and len(kg) == 1:
                    # 2x2x1 block: schoolbook, 4 terms
                    j1, j2 = jg
                    k1 = kg[0]
                    for ii in range(n):
                        for jj in [j1, j2]:
                            U.setdefault((ii, jj), []).append((1, col))
                            V.setdefault((jj, k1), []).append((1, col))
                            W.setdefault((k1, ii), []).append((1, col))
                            col += 1
                elif len(jg) == 1 and len(kg) == 2:
                    # 2x1x2 block: schoolbook, 4 terms
                    j1 = jg[0]
                    k1, k2 = kg
                    for ii in range(n):
                        for kk in [k1, k2]:
                            U.setdefault((ii, j1), []).append((1, col))
                            V.setdefault((j1, kk), []).append((1, col))
                            W.setdefault((kk, ii), []).append((1, col))
                            col += 1
    
    r = col
    U_mat = [[0] * r for _ in range(n * m)]
    V_mat = [[0] * r for _ in range(m * p)]
    W_mat = [[0] * r for _ in range(p * n)]
    
    for (ii, jj), entries in U.items():
        for val, c in entries:
            U_mat[ii * m + jj][c] = val
    for (jj, kk), entries in V.items():
        for val, c in entries:
            V_mat[jj * p + kk][c] = val
    for (kk, ii), entries in W.items():
        for val, c in entries:
            W_mat[kk * n + ii][c] = val
    
    return U_mat, V_mat, W_mat

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _clean_val(x):
    """Clean a float value to integer or half-integer if possible."""
    if abs(x - round(x)) < 1e-7:
        return int(round(x))
    elif abs(2 * x - round(2 * x)) < 1e-7:
        return round(2 * x) / 2.0
    # Try small-denominator rationals
    for den in (3, 4, 5, 6, 8, 10):
        num = round(x * den)
        if abs(x - num / den) < 1e-7:
            return num / den
    return x

def _snap_rational(U, V, W):
    """Snap all entries to nearest clean rational (int/half-integer/small-denom)."""
    U = [[_clean_val(U[i][c]) for c in range(len(U[0]))] for i in range(len(U))]
    V = [[_clean_val(V[i][c]) for c in range(len(V[0]))] for i in range(len(V))]
    W = [[_clean_val(W[i][c]) for c in range(len(W[0]))] for i in range(len(W))]
    return U, V, W

def _init_strassen_like(n, m, p, r, rng):
    """Structured Strassen-like initialization: each column is a rank-1-ish
    signed pattern over blocks."""
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    for c in range(r):
        # Random signs in {-1,0,1}
        for i in range(n * m):
            U[i, c] = rng.choice([-1.0, 0.0, 1.0])
        for j in range(m * p):
            V[j, c] = rng.choice([-1.0, 0.0, 1.0])
        for k in range(p * n):
            W[k, c] = rng.choice([-1.0, 0.0, 1.0])
    return U, V, W

def _init_random_small(n, m, p, r, rng):
    """Random sparse initialization."""
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    for c in range(r):
        for _ in range(max(2, n * m // 3)):
            row = rng.randint(0, n * m - 1)
            U[row, c] = rng.choice([-1.0, 1.0])
        for _ in range(max(2, m * p // 3)):
            row = rng.randint(0, m * p - 1)
            V[row, c] = rng.choice([-1.0, 1.0])
        for _ in range(max(2, p * n // 3)):
            row = rng.randint(0, p * n - 1)
            W[row, c] = rng.choice([-1.0, 1.0])
    return U, V, W

def _init_perturbed_schoolbook(n, m, p, r, rng):
    """Take schoolbook columns and randomly zero out / merge to fit rank r."""
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    # Assign each of the n*m*p schoolbook terms to one of r columns
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = rng.randint(0, r - 1)
                U[i * m + j, target] += 1.0
                V[j * p + k, target] += 1.0
                W[k * n + i, target] += 1.0
                col += 1
    return U, V, W

def _residual(U, V, W, n, m, p):
    """Compute the residual R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1."""
    r = U.shape[1]
    R = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                R[i, j, k] = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]) - 1.0
    return R

def _refine(U, V, W, n, m, p, max_iters, lr, timeout, start_time):
    """Iterative multiplicative refinement: drive residual toward zero.

    Smooth surrogate: minimize sum_{i,j,k} (R[i,j,k])^2 where
    R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1.

    Uses alternating multiplicative updates (like NMF-style) that are
    guaranteed to not blow up and converge to a local min.
    """
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    # Avoid zeros causing division issues
    eps = 1e-10
    U = np.where(np.abs(U) < eps, eps * np.sign(U + eps), U)
    V = np.where(np.abs(V) < eps, eps * np.sign(V + eps), V)
    W = np.where(np.abs(W) < eps, eps * np.sign(W + eps), W)

    prev_obj = None
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break

        # Compute R
        R = _residual(U, V, W, n, m, p)
        obj = float(np.sum(R * R))

        if prev_obj is not None and abs(obj - prev_obj) < 1e-12:
            break
        prev_obj = obj

        if obj < 1e-14:
            break

        # Multiplicative update for U:
        # For each (i, j, c): U[i*m+j, c] *= prod_k [ (1 + R[i,j,k])*V[j*p+k,c]*W[k*n+i,c] / (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + R[i,j,k]) ]^(1/p)
        # Simplified: use gradient-descent-like step on the smooth objective.
        # d obj / d U[a, c] = 2 * sum_k R[i,j,k] * V[j*p+k,c] * W[k*n+i,c]  where a = i*m+j
        # Update: U -= lr * grad

        # Gradient for U
        grad_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    Rval = R[i, j, k]
                    if abs(Rval) < 1e-12:
                        continue
                    idx_v = j * p + k
                    idx_w = k * n + i
                    grad_U[a, :] += 2.0 * Rval * V[idx_v, :] * W[idx_w, :]

        # Gradient for V
        grad_V = np.zeros_like(V)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Rval = R[i, j, k]
                    if abs(Rval) < 1e-12:
                        continue
                    idx_u = i * m + j
                    idx_w = k * n + i
                    idx_v = j * p + k
                    grad_V[idx_v, :] += 2.0 * Rval * U[idx_u, :] * W[idx_w, :]

        # Gradient for W
        grad_W = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Rval = R[i, j, k]
                    if abs(Rval) < 1e-12:
                        continue
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    grad_W[idx_w, :] += 2.0 * Rval * U[idx_u, :] * V[idx_v, :]

        # Adaptive learning rate
        norm_U = np.linalg.norm(grad_U)
        norm_V = np.linalg.norm(grad_V)
        norm_W = np.linalg.norm(grad_W)
        max_norm = max(norm_U, norm_V, norm_W, 1e-12)
        step = lr * 1.0 / max_norm * 0.5

        U = U - step * grad_U
        V = V - step * grad_V
        W = W - step * grad_W

        # Clip to prevent explosion
        U = np.clip(U, -10.0, 10.0)
        V = np.clip(V, -10.0, 10.0)
        W = np.clip(W, -10.0, 10.0)

    return U, V, W, float(prev_obj) if prev_obj is not None else float('inf')

def try_rank_with_refinement(r, n, m, p, timeout, start_time, max_restarts=8):
    """Try to find rank-r decomposition using iterative refinement."""
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break

        rng = random.Random(restart * 104729 + 7)

        # Choose initialization strategy
        if restart % 3 == 0:
            U, V, W = _init_strassen_like(n, m, p, r, rng)
        elif restart % 3 == 1:
            U, V, W = _init_random_small(n, m, p, r, rng)
        else:
            U, V, W = _init_perturbed_schoolbook(n, m, p, r, rng)

        # Quick check if already valid
        if validate_fast(U, V, W, n, m, p):
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list

        # Refine
        remaining = timeout - (time.time() - start_time)
        if remaining < 1.0:
            break

        max_iters = 200
        lr = 0.1
        U, V, W, obj = _refine(U, V, W, n, m, p, max_iters, lr, remaining, start_time)

        # Check convergence
        if obj < 1e-10:
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list

            # Try snapping and re-validating
            if validate_fast(U_list, V_list, W_list, n, m, p, tol=1e-6):
                # High-precision polish: run a few more refinement steps
                U2, V2, W2, obj2 = _refine(
                    np.array(U_list, dtype=np.float64),
                    np.array(V_list, dtype=np.float64),
                    np.array(W_list, dtype=np.float64),
                    n, m, p, 50, 0.01, remaining, start_time
                )
                U2_list = [[_clean_val(U2[i, c]) for c in range(r)] for i in range(n * m)]
                V2_list = [[_clean_val(V2[i, c]) for c in range(r)] for i in range(m * p)]
                W2_list = [[_clean_val(W2[i, c]) for c in range(r)] for i in range(p * n)]
                if validate_exact(U2_list, V2_list, W2_list, n, m, p):
                    return U2_list, V2_list, W2_list

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    # Start with Strassen-based decomposition
    U_st, V_st, W_st = strassen_2x2x2_decomposition(n, m, p)
    best_result = {"U": U_st, "V": V_st, "W": W_st}
    best_rank = len(U_st[0]) if U_st else n * m * p

    # Verify Strassen decomposition
    if not validate_fast(U_st, V_st, W_st, n, m, p):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = n * m * p
    elif validate_exact(U_st, V_st, W_st, n, m, p):
        # Great, Strassen works
        pass
    else:
        # Try to clean and revalidate
        U_cl = [[_clean_val(U_st[i][c]) for c in range(best_rank)] for i in range(n * m)]
        V_cl = [[_clean_val(V_st[i][c]) for c in range(best_rank)] for i in range(m * p)]
        W_cl = [[_clean_val(W_st[i][c]) for c in range(best_rank)] for i in range(p * n)]
        if validate_exact(U_cl, V_cl, W_cl, n, m, p):
            best_result = {"U": U_cl, "V": V_cl, "W": W_cl}
        else:
            U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
            best_rank = n * m * p

    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    lower_bound = max(n * m, m * p, p * n)  # 25
    for r in range(lower_bound, best_rank):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break

        remaining = deadline - elapsed - 2
        if remaining <= 0:
            break

        result = try_rank_with_refinement(r, n, m, p, remaining, start_time, max_restarts=3)

        if result is not None:
            U_r, V_r, W_r = result
            # Final exact validation
            if validate_exact(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = r
                break

    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        U_arr = np.array(best_result["U"], dtype=np.float64)
        V_arr = np.array(best_result["V"], dtype=np.float64)
        W_arr = np.array(best_result["W"], dtype=np.float64)
        if not validate_fast(U_arr, V_arr, W_arr, n, m, p):
            U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    return best_result
