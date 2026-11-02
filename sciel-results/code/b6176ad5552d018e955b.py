"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Verify using exact rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(str(U[i * m + j][r])) * Fraction(str(V[j * p + k][r])) * Fraction(str(W[k * n + i][r]))
                if s != 1:
                    return False
    return True

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
    return U, V, W

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 by 2x2 = 2x2.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00 -> c10(+1), c11(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11) -> c01(+1), c11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00) -> c00(+1), c10(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (a00+a01)b11 -> c00(-1), c01(+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c11(+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11) -> c00(+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks.
    For 3x4x6: 
    n=3: rows 0,1 (2x2 block) + row 2 (1x1)
    m=4: cols 0,1 (2x2 block) + cols 2,3 (2x2 block)  
    p=6: output cols 0,1,2 (2x2+1) + 3,4,5 (2x2+1)
    
    Total rank: 6*7 + 6*4 = 66
    """
    n, m, p = 3, 4, 6
    
    total_rank = 6 * 7 + 6 * 4  # 66
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    def embed_2x2x2(a_row0, a_col0, b_row0, c_row0, c_col0, start_col):
        """
        A submatrix: rows a_row0..a_row0+1, cols a_col0..a_col0+1 (2x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: rows c_row0..c_row0+1, cols c_col0..c_col0+1 (2x2)
        """
        sU, sV, sW = _strassen_2x2_2x2()
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = a_row0 + li
                    gj = a_col0 + lj
                    U[gi * m + gj][start_col + r] += sU[li * 2 + lj][r]
                    
                    for lk in range(2):
                        gj_b = b_row0 + lj
                        gk = c_col0 + lk
                        V[gj_b * p + gk][start_col + r] += sV[lj * 2 + lk][r]
                    
                    for lk in range(2):
                        gi_c = c_row0 + li
                        gk = c_col0 + lk
                        W[gk * n + gi_c][start_col + r] += sW[lk * 2 + li][r]
        
        return start_col + 7
    
    def embed_1x2x2(a_row, a_col0, b_row0, c_row, c_col0, start_col):
        """
        A submatrix: row a_row, cols a_col0..a_col0+1 (1x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: row c_row, cols c_col0..c_col0+1 (1x2)
        
        Schoolbook: 4 terms
        """
        for l1 in range(2):
            for l2 in range(2):
                gi = a_row
                gj = a_col0 + l1
                U[gi * m + gj][start_col] = 1
                
                gj_b = b_row0 + l1
                gk = c_col0 + l2
                V[gj_b * p + gk][start_col] = 1
                
                gi_c = c_row
                gk = c_col0 + l2
                W[gk * n + gi_c][start_col] = 1
                
                start_col += 1
        
        return start_col
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    col = embed_2x2x2(0, 0, 0, 0, 0, col)
    col = embed_2x2x2(0, 2, 2, 0, 0, col)
    
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    col = embed_2x2x2(0, 0, 0, 0, 2, col)
    col = embed_2x2x2(0, 2, 2, 0, 2, col)
    
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    col = embed_2x2x2(0, 0, 0, 0, 4, col)
    col = embed_2x2x2(0, 2, 2, 0, 4, col)
    
    # C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    col = embed_1x2x2(2, 0, 0, 2, 0, col)
    col = embed_1x2x2(2, 2, 2, 2, 0, col)
    
    # C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    col = embed_1x2x2(2, 0, 0, 2, 2, col)
    col = embed_1x2x2(2, 2, 2, 2, 2, col)
    
    # C[2, 4:6] = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6]
    col = embed_1x2x2(2, 0, 0, 2, 4, col)
    col = embed_1x2x2(2, 2, 2, 2, 4, col)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    return U, V, W

def _try_rank_am(n, m, p, rank, max_iter=500, tol=1e-10, seed=0, verbose=False):
    """
    Try to find a decomposition of given rank using alternating minimization.
    Uses vectorized operations for speed.
    """
    if time.time() > _DEADLINE - 3.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    U_idx = [(i * m + j) for i in range(n) for j in range(m)]
    V_idx = [(j * p + k) for j in range(m) for k in range(p)]
    W_idx = [(k * n + i) for k in range(p) for i in range(n)]
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3.0:
            return None
        
        # Update U: for each (i,j), solve least squares
        for idx, (i, j) in enumerate(zip(*np.unravel_index(U_idx, (n, m)))):
            # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[idx, :] = x
            except:
                pass
        
        # Update V: for each (j,k), solve least squares
        for idx, (j, k) in enumerate(zip(*np.unravel_index(V_idx, (m, p)))):
            # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
            B = np.zeros((n, rank))
            for i in range(n):
                B[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V[idx, :] = x
            except:
                pass
        
        # Update W: for each (k,i), solve least squares
        for idx, (k, i) in enumerate(zip(*np.unravel_index(W_idx, (p, n)))):
            # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
            C = np.zeros((m, rank))
            for j in range(m):
                C[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W[idx, :] = x
            except:
                pass
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err = max(err, abs(s - 1.0))
            if verbose:
                print(f"Rank {rank}, iter {it}, err {err:.2e}")
            if err < tol:
                # Round to half-integers
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                for row in U_list:
                    for idx2 in range(len(row)):
                        v2 = row[idx2] * 2
                        if abs(v2 - round(v2)) < 1e-6:
                            row[idx2] = round(v2) / 2.0
                for row in V_list:
                    for idx2 in range(len(row)):
                        v2 = row[idx2] * 2
                        if abs(v2 - round(v2)) < 1e-6:
                            row[idx2] = round(v2) / 2.0
                for row in W_list:
                    for idx2 in range(len(row)):
                        v2 = row[idx2] * 2
                        if abs(v2 - round(v2)) < 1e-6:
                            row[idx2] = round(v2) / 2.0
                return U_list, V_list, W_list
    
    return None

def _try_rank_gauss_newton(n, m, p, rank, max_iter=200, tol=1e-12, seed=0):
    """
    Try to find a decomposition using Gauss-Newton / Levenberg-Marquardt.
    This is more robust than plain alternating minimization.
    """
    if time.time() > _DEADLINE - 3.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Stack all parameters
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    n_params = len(params)
    
    # Residual: for each (i,j,k), residual = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    def residual(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
        W_p = params[2*n*m*rank:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_p[i*m+j, :], V_p[j*p+k, :] * W_p[k*n+i, :])
                    res[idx] = 1.0 - s
                    idx += 1
        return res
    
    def jacobian(params):
        """Compute Jacobian using finite differences (slow but simple)."""
        res0 = residual(params)
        n_res = len(res0)
        J = np.zeros((n_res, n_params))
        eps = 1e-7
        for j in range(n_params):
            params_pert = params.copy()
            params_pert[j] += eps
            res_pert = residual(params_pert)
            J[:, j] = (res_pert - res0) / eps
        return J
    
    # Levenberg-Marquardt
    lam = 1e-3
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3.0:
            return None
        
        res = residual(params)
        err = np.max(np.abs(res))
        if err < tol:
            break
        
        J = jacobian(params)
        JtJ = J.T @ J
        Jtr = J.T @ res
        
        # Solve (JtJ + lam*I) delta = -Jtr
        I = np.eye(n_params)
        A = JtJ + lam * I
        try:
            delta = np.linalg.solve(A, -Jtr)
        except:
            delta = -Jtr * 0.01
        
        # Try step
        params_new = params + delta
        res_new = residual(params_new)
        err_new = np.max(np.abs(res_new))
        
        if err_new < err:
            params = params_new
            lam = max(lam * 0.1, 1e-12)
        else:
            lam = min(lam * 10.0, 1e6)
        
        if err < tol:
            break
    
    err = np.max(np.abs(residual(params)))
    if err < 1e-8:
        U_list = params[:n*m*rank].reshape(n*m, rank).tolist()
        V_list = params[n*m*rank:2*n*m*rank].reshape(m*p, rank).tolist()
        W_list = params[2*n*m*rank:].reshape(p*n, rank).tolist()
        for row in U_list:
            for idx2 in range(len(row)):
                v2 = row[idx2] * 2
                if abs(v2 - round(v2)) < 1e-6:
                    row[idx2] = round(v2) / 2.0
        for row in V_list:
            for idx2 in range(len(row)):
                v2 = row[idx2] * 2
                if abs(v2 - round(v2)) < 1e-6:
                    row[idx2] = round(v2) / 2.0
        for row in W_list:
            for idx2 in range(len(row)):
                v2 = row[idx2] * 2
                if abs(v2 - round(v2)) < 1e-6:
                    row[idx2] = round(v2) / 2.0
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First try the structured Strassen-based decomposition (rank 66)
    try:
        U, V, W = _construct_strassen_based(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find even lower rank via alternating minimization
    for rank in range(55, 66):
        if time.time() > _DEADLINE - 10.0:
            break
        for seed in range(10):
            if time.time() > _DEADLINE - 5.0:
                break
            result = _try_rank_am(n, m, p, rank, max_iter=300, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Try Gauss-Newton for a few ranks
    for rank in range(58, 66):
        if time.time() > _DEADLINE - 5.0:
            break
        for seed in range(3):
            if time.time() > _DEADLINE - 3.0:
                break
            result = _try_rank_gauss_newton(n, m, p, rank, max_iter=50, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
