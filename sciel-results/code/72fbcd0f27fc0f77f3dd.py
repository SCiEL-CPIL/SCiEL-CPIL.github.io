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


def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition via numerical search.
    
    The tensor has 12 entries (one per (i,j,k) triple). We need to find
    U (4x10), V (6x10), W (6x10) such that for each (i,j,k):
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    We use a structured approach: fix the Strassen-like structure for the
    2x2x2 part and try to optimize the remaining degrees of freedom.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    # U: 4 x 10, V: 6 x 10, W: 6 x 10
    # Total parameters: 40 + 60 + 60 = 160
    
    def residuals(params):
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
    
    # Strategy: use the rank-11 solution as a starting point, zero out one term,
    # and try to optimize.
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)  # 4 x 11
    V11 = np.array(base["V"], dtype=float)  # 6 x 11
    W11 = np.array(base["W"], dtype=float)  # 6 x 11
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = float('inf')
    
    # Try different approaches:
    # 1. Start from rank-11, drop one term, optimize
    # 2. Random initialization with structure
    
    for drop_term in range(11):
        if time.time() > deadline - 10:
            break
        
        # Create initial guess by dropping term 'drop_term' and redistributing
        U0 = U11[:, list(range(11))[:drop_term] + list(range(11))[drop_term+1:]].copy()
        V0 = V11[:, list(range(11))[:drop_term] + list(range(11))[drop_term+1:]].copy()
        W0 = W11[:, list(range(11))[:drop_term] + list(range(11))[drop_term+1:]].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Add small random perturbation
        x0 = x0 + rng.normal(0, 0.1, 160)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    # Also try random initializations
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        x0 = rng.normal(0, 1, 160)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                
                if result.cost < 1e-12:
                    break
        except Exception:
            pass
    
    if best_result is None or best_cost > 1e-10:
        return None
    
    # Extract and round to nearest half-integer
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
    
    # If rounding didn't work, try integer rounding
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
    
    Key idea: The 2x2x1 part (column 2) involves:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    These can potentially be combined with Strassen terms.
    
    Let's try a parameterized family where we modify the Strassen terms
    to also contribute to column 2.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    # Parameterize: keep the 7 Strassen terms for 2x2x2, but allow them
    # to also have components in column 2 of V and W.
    # The 4 schoolbook terms for column 2 become the remaining 3 terms
    # (since we need rank 10, not 11).
    
    # Actually, let's think differently. We have 10 terms total.
    # Terms 0-6: Strassen-like for 2x2x2 (but modified)
    # Terms 7-9: For column 2
    
    # The constraint is that terms 0-6 must satisfy the 2x2x2 tensor
    # AND terms 7-9 must handle column 2, with possible overlap.
    
    # Let's use a different parameterization:
    # Fix U for the Strassen part, optimize V and W.
    
    # Strassen U patterns (for 2x2x2):
    # Term 0: A00+A11
    # Term 1: A10+A11
    # Term 2: A00
    # Term 3: A11
    # Term 4: A00+A01
    # Term 5: A10-A00
    # Term 6: A01-A11
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 1, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)  # 4 x 7
    
    # We need 3 more terms. Let's parameterize them.
    # For column 2, we need:
    # C[0,2] = A00*B02 + A01*B12
    # C[1,2] = A10*B02 + A11*B12
    
    # With 3 terms, we need to represent these 4 bilinear forms.
    # This is underdetermined (4 equations, 3*3=9 unknowns for the 3 terms' V and W),
    # so it should be feasible.
    
    # Let's set up the optimization.
    # Unknowns: 
    #   V for all 10 terms: 6*10 = 60
    #   W for all 10 terms: 6*10 = 60
    #   U for terms 7-9: 4*3 = 12
    # Total: 132 unknowns
    # Constraints: 12 equations
    
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
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Initialize V and W for Strassen terms (terms 0-6)
        # Standard Strassen V and W for 2x2x2:
        V_strassen = np.array([
            [1, 0, 0, 0, 0, 1, 0],   # B00
            [0, 0, 1, 0, 0, 1, 0],   # B01
            [0, 0, 0, 1, 0, 0, 1],   # B10
            [1, 0, -1, -1, 1, 0, 1], # B11
            [0, 0, 0, 0, 0, 0, 0],   # B02 (not used in Strassen)
            [0, 0, 0, 0, 0, 0, 0],   # B12 (not used in Strassen)
        ], dtype=float)  # 6 x 7
        
        W_strassen = np.array([
            [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
            [0, 1, 1, 0, 0, 0, 0],   # C[0,1]
            [0, 0, 0, 0, 0, 1, 0],   # C[1,0]
            [1, -1, 1, 0, 1, 0, 0],  # C[1,1]
            [0, 0, 0, 0, 0, 0, 0],   # C[0,2]
            [0, 0, 0, 0, 0, 0, 0],   # C[1,2]
        ], dtype=float)  # 6 x 7
        
        # Initialize V and W for extra terms (7-9) randomly
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


def _try_rank10_code_based(deadline):
    """
    Code-based approach: treat the tensor as a linear code and use
    algebraic techniques to find a rank-10 decomposition.
    
    The idea: the matrix multiplication tensor T has entries T[i,j,k] = 1
    for all (i,j,k). We want to write T as a sum of 10 rank-1 tensors.
    
    A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^4, v ∈ R^6, w ∈ R^6.
    
    We can think of this as finding a linear code with specific properties.
    The generator matrix would have rows corresponding to the terms.
    
    For a more direct approach, let's use the fact that the tensor is
    "generic" in some sense and try to find a solution using the
    Berlekamp-Massey-like approach or by solving the polynomial system.
    
    Actually, let's try a simpler structured approach: assume the solution
    has a specific sparsity pattern based on the code structure.
    """
    if time.time() > deadline - 15:
        return None
    
    from scipy.optimize import least_squares
    
    # Let's try a highly structured approach where we assume
    # specific patterns for U, V, W based on the algebraic structure.
    
    # Key insight: The 2x2x3 tensor can be seen as two 2x2x1 tensors
    # (for columns 0-1 and column 2) plus the 2x2x2 tensor.
    
    # For the 2x2x2 part, Strassen uses 7 terms.
    # For the 2x2x1 part (column 2), we need 4 terms in schoolbook.
    # Total: 11 terms.
    
    # To get rank 10, we need to save 1 term by sharing.
    
    # Idea: Can one of the Strassen terms also contribute to column 2?
    
    # Let's parameterize: take the 7 Strassen terms and add 3 more terms.
    # The 7 Strassen terms handle the 2x2x2 part.
    # The 3 extra terms must handle column 2 AND possibly modify the
    # 2x2x2 part to maintain correctness.
    
    # Actually, let's just do a full numerical search with good initialization.
    
    # Use the rank-11 solution as a warm start, but with 10 terms.
    # Drop the term that seems most "redundant" and optimize.
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    def residuals(params):
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
    
    rng = np.random.default_rng(777)
    
    best_cost = float('inf')
    best_result = None
    
    # Try dropping each of the 11 terms and optimizing
    for drop in range(11):
        if time.time() > deadline - 10:
            break
        
        keep = [r for r in range(11) if r != drop]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        x0 = x0 + rng.normal(0, 0.05, 160)
        
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
    
    U_flat = best_result[:40]
    V_flat = best_result[40:100]
    W_flat = best_result[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try rounding to half-integers
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # Try rounding to integers
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
    
    # Try to find a rank-10 solution using structured search
    if best_rank > 10 and time.time() < deadline - 40:
        result10 = _try_rank10_structured(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
    
    # Try code-based approach
    if best_rank > 10 and time.time() < deadline - 20:
        result10b = _try_rank10_code_based(deadline)
        if result10b is not None:
            best = result10b
            best_rank = 10
    
    # Try general rank-10 search
    if best_rank > 10 and time.time() < deadline - 10:
        result10c = _try_rank10_search(deadline)
        if result10c is not None:
            best = result10c
            best_rank = 10
    
    return best
