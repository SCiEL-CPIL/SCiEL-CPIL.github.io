"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    rng = np.random.default_rng(42)
    for _ in range(50):
        A = rng.integers(-10, 11, size=(n, m)).astype(float)
        B = rng.integers(-10, 11, size=(m, p)).astype(float)
        C = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = np.array(U)[:, r]
            v_vec = np.array(V)[:, r]
            w_vec = np.array(W)[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    C_decomp[i, k] += s
        if np.max(np.abs(C - C_decomp)) > tol:
            return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known best construction from inspiration 2.
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

    # Term 0: (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> M7
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
    
    Strategy: Start from the rank-11 solution and try to eliminate one term
    by finding a linear dependency or by solving a constrained optimization.
    
    For rank 10, we need 10 terms. The 2x2x2 part has rank 7 (Strassen is optimal
    for 2x2x2). The 2x2x1 part (A times a column vector) has tensor rank 4 
    (schoolbook is optimal for 2x2x1? Let's check: T[i,j,k] with k fixed, 
    it's just a 2x2 matrix times a scalar... actually 2x2x1 means C[i] = sum_j A[i,j]*v[j],
    which is a 2x2x1 tensor. Its rank is 2 (not 4) because it's just two independent
    bilinear forms: C[0] = A[0,0]*v[0] + A[0,1]*v[1] and C[1] = A[1,0]*v[0] + A[1,1]*v[1].
    
    Wait, that's not right. The tensor for 2x2x1 is T[i,j,k] where k is fixed (say k=2).
    T[i,j,2] = 1 for all i,j. The rank of this tensor... it's a 2x2x1 tensor which is
    just a 2x2 matrix. The tensor rank of a matrix is its matrix rank, which is 2.
    
    So actually, the 2x2x1 part can be done in rank 2, not 4!
    
    Let me reconsider. For A (2x2) times v (2x1), we get C (2x1).
    C[0] = A[0,0]*v[0] + A[0,1]*v[1]
    C[1] = A[1,0]*v[0] + A[1,1]*v[1]
    
    This can be written as:
    C[0] = (A[0,0]+A[0,1])*(v[0]+v[1]) - A[0,1]*v[0] - A[0,0]*v[1]... no that's not helpful.
    
    Actually, the simplest: use 2 terms:
    Term 1: A[0,0]*v[0] contributes to C[0]
    Term 2: A[0,1]*v[1] contributes to C[0]
    Term 3: A[1,0]*v[0] contributes to C[1]
    Term 4: A[1,1]*v[1] contributes to C[1]
    
    That's 4 terms. Can we do it in 2?
    C[0] = sum_j A[0,j] v[j]
    C[1] = sum_j A[1,j] v[j]
    
    If we use rank 2:
    M1 = (alpha00*A[0,0] + alpha10*A[1,0] + ...) * (beta00*v[0] + beta10*v[1] + ...)
    M2 = ...
    
    We need:
    C[0] = M1_w0 * M1 + M2_w0 * M2
    C[1] = M1_w1 * M1 + M2_w1 * M2
    
    where M1, M2 are bilinear forms in A and v.
    
    For example:
    M1 = (A[0,0]+A[1,0]) * v[0]
    M2 = (A[0,1]+A[1,1]) * v[1]
    
    Then:
    C[0] = ? We need coefficients w0_1, w0_2 such that:
    w0_1 * (A[0,0]+A[1,0])*v[0] + w0_2 * (A[0,1]+A[1,1])*v[1] = A[0,0]*v[0] + A[0,1]*v[1]
    
    This requires w0_1=1, w0_2=1, but then we also get A[1,0]*v[0] + A[1,1]*v[1] in C[0], which is wrong.
    
    So rank 2 doesn't work for 2x2x1. The rank is 4.
    
    Therefore, the total rank is at least 7 + 4 = 11 if we can't share terms.
    But maybe we CAN share terms between the Strassen part and the schoolbook part?
    
    Let me try a numerical search for rank 10.
    """
    if time.time() > deadline - 30:
        return None
    
    # Try to find a rank-10 solution by solving a system of equations.
    # We have 12 equations (one per (i,j,k) triple) and 30 unknowns (10*3).
    # Actually, the unknowns are U (4*10=40), V (6*10=60), W (6*10=60) = 160 unknowns.
    # But we only have 12 equations. This is underdetermined, so we should be able to find a solution.
    # The challenge is that the equations are nonlinear (products of U, V, W).
    
    # Let me try a simple approach: fix some variables and solve for others.
    # Or use scipy's least_squares.
    
    from scipy.optimize import least_squares
    
    # Parameters: flatten U, V, W
    # U: 4x10, V: 6x10, W: 6x10
    # Total: 40 + 60 + 60 = 160 parameters
    
    # But we have gauge freedom: we can scale U_r by alpha, V_r by 1/alpha, W_r by 1.
    # And we can permute the terms.
    
    # Let me try a different approach: start from the rank-11 solution and try to
    # merge two terms into one.
    
    # Actually, let me just try a random initialization and see if least_squares can converge.
    
    rng = np.random.default_rng(123)
    
    def residuals(params):
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        res = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res.append(s - 1.0)
        return res
    
    # Initialize with random values
    x0 = rng.normal(0, 1, 160)
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        x0 = rng.normal(0, 1, 160)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.cost < 1e-12:
                U_flat = result.x[:40]
                V_flat = result.x[40:100]
                W_flat = result.x[100:160]
                
                U = U_flat.reshape(4, 10).tolist()
                V = V_flat.reshape(6, 10).tolist()
                W = W_flat.reshape(6, 10).tolist()
                
                # Round to nearest half-integer
                U = [[round(u * 2) / 2 for u in row] for row in U]
                V = [[round(v * 2) / 2 for v in row] for row in V]
                W = [[round(w * 2) / 2 for w in row] for row in W]
                
                if _verify(U, V, W, 2, 2, 3):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return None


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # First, try the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        # Fallback to schoolbook
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution
    if best_rank > 10 and time.time() < deadline - 30:
        result10 = _try_rank10_search(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
    
    return best
