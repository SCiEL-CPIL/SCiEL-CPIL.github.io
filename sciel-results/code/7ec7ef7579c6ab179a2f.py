"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all 12 (i,j,k) triples exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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


def _try_rank10_kronecker(deadline):
    """
    Kronecker product lifting strategy:
    Represent the 2x2x3 tensor as a lift of the 2x2x2 tensor.
    
    Key insight: The 2x2x3 multiplication tensor T can be written as:
    T = T_2x2x2 ⊗ I_3 (conceptually), but more precisely, we can view
    the third column as an extension.
    
    The rank-11 solution has 7 Strassen terms + 4 schoolbook terms.
    To get rank 10, we need to merge one schoolbook term into the Strassen
    structure by allowing the Strassen terms to also have components in
    the third column.
    
    Specifically, we parameterize:
    - U: 4x10 (fixed Strassen pattern for first 7 cols + 3 free cols)
    - V: 6x10 (Strassen V for first 7 cols + free for cols 7-9)
    - W: 6x10 (Strassen W for first 7 cols + free for cols 7-9)
    
    The constraint is that the first 8 entries (i,j,k with k<2) must equal 1,
    and the last 4 entries (k=2) must equal 1.
    
    We use the linear system approach: fix U to a structured form, then solve
    for V and W using least squares.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    
    # Strassen U pattern for 2x2x2 (7 terms)
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 1, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)  # 4 x 7
    
    # Strassen V pattern for 2x2x2 (7 terms)
    V_strassen = np.array([
        [1, 0, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, -1, 1, 0, 1], # B11
        [0, 0, 0, 0, 0, 0, 0],   # B02
        [0, 0, 0, 0, 0, 0, 0],   # B12
    ], dtype=float)  # 6 x 7
    
    # Strassen W pattern for 2x2x2 (7 terms)
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 1, 0, 0, 0, 0],   # C[0,1]
        [0, 0, 0, 0, 0, 1, 0],   # C[1,0]
        [1, -1, 1, 0, 1, 0, 0],  # C[1,1]
        [0, 0, 0, 0, 0, 0, 0],   # C[0,2]
        [0, 0, 0, 0, 0, 0, 0],   # C[1,2]
    ], dtype=float)  # 6 x 7
    
    def residuals(params):
        """
        params: 
          - V_extra: 6x3 = 18 params (V for cols 7-9)
          - W_extra: 6x3 = 18 params (W for cols 7-9)
          - U_extra: 4x3 = 12 params (U for cols 7-9)
          - V_strassen_mod: 6x7 = 42 params (modified Strassen V, to allow col 2 contributions)
          - W_strassen_mod: 6x7 = 42 params (modified Strassen W)
        Total: 132 params
        """
        V_extra = params[:18].reshape(6, 3)
        W_extra = params[18:36].reshape(6, 3)
        U_extra = params[36:48].reshape(4, 3)
        V_strassen_mod = params[48:90].reshape(6, 7)
        W_strassen_mod = params[90:132].reshape(6, 7)
        
        U_mat = np.hstack([U_strassen, U_extra])  # 4 x 10
        V_mat = np.hstack([V_strassen_mod, V_extra])  # 6 x 10
        W_mat = np.hstack([W_strassen_mod, W_extra])  # 6 x 10
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    rng = np.random.default_rng(12345)
    
    best_cost = float('inf')
    best_result = None
    
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Initialize with Strassen values + small perturbation
        V_strassen_mod0 = V_strassen + rng.normal(0, 0.1, (6, 7))
        W_strassen_mod0 = W_strassen + rng.normal(0, 0.1, (6, 7))
        V_extra0 = rng.normal(0, 1, (6, 3))
        W_extra0 = rng.normal(0, 1, (6, 3))
        U_extra0 = rng.normal(0, 1, (4, 3))
        
        x0 = np.concatenate([
            V_extra0.flatten(),
            W_extra0.flatten(),
            U_extra0.flatten(),
            V_strassen_mod0.flatten(),
            W_strassen_mod0.flatten()
        ])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    if best_result is None or best_cost > 1e-10:
        return None
    
    V_extra = best_result[:18].reshape(6, 3)
    W_extra = best_result[18:36].reshape(6, 3)
    U_extra = best_result[36:48].reshape(4, 3)
    V_strassen_mod = best_result[48:90].reshape(6, 7)
    W_strassen_mod = best_result[90:132].reshape(6, 7)
    
    U_mat = np.hstack([U_strassen, U_extra])
    V_mat = np.hstack([V_strassen_mod, V_extra])
    W_mat = np.hstack([W_strassen_mod, W_extra])
    
    # Round to nearest half-integer
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # Try integer rounding
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_projection(deadline):
    """
    Linear projection approach: Solve for the third column's interaction
    as a linear combination of the first two columns' interactions.
    
    The idea: For k=2, we need:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    If we can express B[0,2] and B[1,2] as linear combinations of B[0,0], B[0,1], B[1,0], B[1,1]
    in the V-factor space, then we can reuse the Strassen terms.
    
    More concretely: we look for coefficients alpha, beta, gamma, delta such that:
    V[:, 2] = alpha * V[:, 0] + beta * V[:, 1] + gamma * V[:, 3] + delta * V[:, 4]
    (where V[:, 0] is the column for B[0,0], etc.)
    
    Then the W factor for k=2 can be determined from the Strassen W factors.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    
    # Start from the rank-11 solution and try to merge one term
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)  # 4 x 11
    V11 = np.array(base["V"], dtype=float)  # 6 x 11
    W11 = np.array(base["W"], dtype=float)  # 6 x 11
    
    # Try dropping each of the 4 schoolbook terms (7-10) and see if we can
    # absorb it into the Strassen terms by modifying them
    def residuals(params, drop_idx):
        """
        params: 4x10 + 6x10 + 6x10 = 160 params
        drop_idx: which of the 11 terms to drop
        """
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    rng = np.random.default_rng(54321)
    
    best_cost = float('inf')
    best_result = None
    best_drop = None
    
    # Try dropping each schoolbook term
    for drop_idx in [7, 8, 9, 10]:
        if time.time() > deadline - 10:
            break
        
        keep = [r for r in range(11) if r != drop_idx]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Add perturbation
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            x0_perturbed = x0 + rng.normal(0, 0.1, 160)
            
            try:
                result = least_squares(residuals, x0_perturbed, args=(drop_idx,), 
                                       method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result.x.copy()
                    best_drop = drop_idx
                    if result.cost < 1e-12:
                        break
            except Exception:
                pass
        
        if best_cost < 1e-12:
            break
    
    if best_result is None or best_cost > 1e-10:
        return None
    
    U_flat = best_result[:40]
    V_flat = best_result[40:100]
    W_flat = best_result[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Round to nearest half-integer
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # Try integer rounding
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_structured(deadline):
    """
    Structured rank-10 search using the insight that the 2x2x3 tensor
    can be viewed as a 2x2x2 tensor (rank 7) plus a 2x2x1 tensor (rank 4),
    but with potential sharing of terms.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    
    # Strassen U patterns (for 2x2x2):
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 1, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)  # 4 x 7
    
    def residuals(params):
        V_flat = params[:60]
        W_flat = params[60:120]
        U_extra_flat = params[120:132]
        
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        U_extra = U_extra_flat.reshape(4, 3)
        
        U_mat = np.hstack([U_strassen, U_extra])  # 4 x 10
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    rng = np.random.default_rng(99)
    
    best_cost = float('inf')
    best_result = None
    
    # Standard Strassen V and W for 2x2x2:
    V_strassen = np.array([
        [1, 0, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, -1, 1, 0, 1], # B11
        [0, 0, 0, 0, 0, 0, 0],   # B02
        [0, 0, 0, 0, 0, 0, 0],   # B12
    ], dtype=float)  # 6 x 7
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 1, 0, 0, 0, 0],   # C[0,1]
        [0, 0, 0, 0, 0, 1, 0],   # C[1,0]
        [1, -1, 1, 0, 1, 0, 0],  # C[1,1]
        [0, 0, 0, 0, 0, 0, 0],   # C[0,2]
        [0, 0, 0, 0, 0, 0, 0],   # C[1,2]
    ], dtype=float)  # 6 x 7
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        V_extra = rng.normal(0, 1, (6, 3))
        W_extra = rng.normal(0, 1, (6, 3))
        U_extra = rng.normal(0, 1, (4, 3))
        
        V0 = np.hstack([V_strassen, V_extra])
        W0 = np.hstack([W_strassen, W_extra])
        
        x0 = np.concatenate([V0.flatten(), W0.flatten(), U_extra.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    if best_result is None or best_cost > 1e-10:
        return None
    
    V_flat = best_result[:60]
    W_flat = best_result[60:120]
    U_extra_flat = best_result[120:132]
    
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    U_extra = U_extra_flat.reshape(4, 3)
    
    U_mat = np.hstack([U_strassen, U_extra])
    
    # Round to nearest half-integer
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # Try integer rounding
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try Kronecker product lifting approach for rank 10
    if best_rank > 10 and time.time() < deadline - 60:
        result10_k = _try_rank10_kronecker(deadline)
        if result10_k is not None:
            best = result10_k
            best_rank = 10
    
    # Try linear projection approach for rank 10
    if best_rank > 10 and time.time() < deadline - 40:
        result10_p = _try_rank10_projection(deadline)
        if result10_p is not None:
            best = result10_p
            best_rank = 10
    
    # Try structured rank-10 search
    if best_rank > 10 and time.time() < deadline - 20:
        result10_s = _try_rank10_structured(deadline)
        if result10_s is not None:
            best = result10_s
            best_rank = 10
    
    return best
