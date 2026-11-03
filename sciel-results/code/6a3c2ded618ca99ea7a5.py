"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Strategy:
  1. Use the "fix two, solve for third" trick: fix U and V, solve for W via
     least-squares (linear system), then snap W to half-integers and verify.
     Symmetrize by also fixing (U,W) -> solve V, and (V,W) -> solve U.
  2. Use Adam-based continuous polishing with a final snap-and-verify step.
  3. Fall back to schoolbook rank-36.

All evaluations are bounded by an explicit deadline well inside the 120 s wall.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 105.0   # 15 s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
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


def _evaluate(U, V, W, n, m, p):
    """Compute max |sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1| over all (i,j,k)."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _snap_half(mat):
    """Snap a numpy array to nearest half-integer."""
    return np.round(mat * 2.0) / 2.0


def _try_fix_two_solve_third(n, m, p, rank, deadline, seed=0):
    """Fix two factor matrices, solve for the third via least squares, snap, verify.
    
    Cycles through which pair to fix:
      - fix U, V -> solve W
      - fix U, W -> solve V
      - fix V, W -> solve U
    """
    rng = np.random.RandomState(seed)
    pairs = [
        ('U', 'V', 'W'),
        ('U', 'W', 'V'),
        ('V', 'W', 'U'),
    ]
    
    for attempt in range(200):
        if time.time() > deadline - 1.0:
            break
        
        # Pick a random pair to fix
        pair = pairs[attempt % len(pairs)]
        fixed_a, fixed_b, solve_c = pair
        
        # Initialize the two fixed matrices with small half-integers
        # Shape depends on which matrix
        if fixed_a == 'U':
            shape_a = (n * m, rank)
        elif fixed_a == 'V':
            shape_a = (m * p, rank)
        else:
            shape_a = (p * n, rank)
        
        if fixed_b == 'U':
            shape_b = (n * m, rank)
        elif fixed_b == 'V':
            shape_b = (m * p, rank)
        else:
            shape_b = (p * n, rank)
        
        # Random initialization: mix of sparse and dense patterns
        if attempt % 4 == 0:
            # Sparse: each column has 1-3 nonzeros
            A = np.zeros(shape_a)
            B = np.zeros(shape_b)
            for r in range(rank):
                for _ in range(rng.randint(1, 3)):
                    row = rng.randint(0, shape_a[0])
                    A[row, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
                for _ in range(rng.randint(1, 3)):
                    row = rng.randint(0, shape_b[0])
                    B[row, r] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        elif attempt % 4 == 1:
            # Structured: cyclic assignment
            A = np.zeros(shape_a)
            B = np.zeros(shape_b)
            for r in range(rank):
                row_a = r % shape_a[0]
                A[row_a, r] = 1.0
                row_b = (r * 7 + attempt) % shape_b[0]
                B[row_b, r] = 1.0
        else:
            # Dense small random
            A = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=shape_a)
            B = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=shape_b)
        
        # Build the linear system to solve for the third matrix
        # We need to construct the matrix M such that M @ W_vec = ones
        # where W_vec is the flattened unknown matrix
        
        # Number of equations = n*m*p = 36
        n_eq = n * m * p
        
        # Determine shape of unknown
        if solve_c == 'W':
            shape_c = (p * n, rank)
        elif solve_c == 'V':
            shape_c = (m * p, rank)
        else:
            shape_c = (n * m, rank)
        
        n_unknowns = shape_c[0] * rank
        
        # Build M (n_eq x n_unknowns)
        M = np.zeros((n_eq, n_unknowns))
        rhs = np.ones(n_eq)
        
        # Map from (i,j,k) to equation index
        # For each equation (i,j,k):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # The unknown is W[k*n+i, r], so for each r, M[equation_idx, (k*n+i)*rank + r] = U[i*m+j,r] * V[j*p+k,r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    
                    if solve_c == 'W':
                        # Fixed: A=U, B=V, Solve: W
                        for r in range(rank):
                            col_idx = (k * n + i) * rank + r
                            M[eq_idx, col_idx] = A[i * m + j, r] * B[j * p + k, r]
                    elif solve_c == 'V':
                        # Fixed: A=U, B=W, Solve: V
                        for r in range(rank):
                            col_idx = (j * p + k) * rank + r
                            M[eq_idx, col_idx] = A[i * m + j, r] * B[k * n + i, r]
                    else:
                        # Fixed: A=V, B=W, Solve: U
                        for r in range(rank):
                            col_idx = (i * m + j) * rank + r
                            M[eq_idx, col_idx] = A[j * p + k, r] * B[k * n + i, r]
        
        # Solve least squares
        try:
            W_vec, residuals, rank_ls, sv = np.linalg.lstsq(M, rhs, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Reshape and snap
        C = _snap_half(W_vec.reshape(shape_c))
        
        # Now verify all three
        if solve_c == 'W':
            U_mat, V_mat, W_mat = A, B, C
        elif solve_c == 'V':
            U_mat, V_mat, W_mat = A, C, B
        else:
            U_mat, V_mat, W_mat = C, A, B
        
        err = _evaluate(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
        if err < 1e-9:
            return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None


def _smooth_polish(U, V, W, n, m, p, deadline, tol=1e-9, max_iters=300):
    """Adam optimizer to minimize smooth surrogate, then snap and verify."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    def _obj_and_grad(Ua, Va, Wa):
        loss = 0.0
        gU = np.zeros_like(Ua)
        gV = np.zeros_like(Va)
        gW = np.zeros_like(Wa)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
                    d = s - 1.0
                    loss += d * d
                    gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
                    gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
                    gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])
        return loss, gU, gV, gW
    
    b1, b2, eps = 0.9, 0.999, 1e-8
    mU = np.zeros_like(Ua); vU = np.zeros_like(Ua)
    mV = np.zeros_like(Va); vV = np.zeros_like(Va)
    mW = np.zeros_like(Wa); vW = np.zeros_like(Wa)
    t = 0
    lr = 0.03
    
    cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
    best_loss = cur_loss
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        t += 1
        loss, gU, gV, gW = _obj_and_grad(Ua, Va, Wa)
        
        mU = b1 * mU + (1 - b1) * gU
        vU = b2 * vU + (1 - b2) * gU ** 2
        mU_hat = mU / (1 - b1 ** t)
        vU_hat = vU / (1 - b2 ** t)
        
        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * gV ** 2
        mV_hat = mV / (1 - b1 ** t)
        vV_hat = vV / (1 - b2 ** t)
        
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW ** 2
        mW_hat = mW / (1 - b1 ** t)
        vW_hat = vW / (1 - b2 ** t)
        
        Ua -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        Va -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        Wa -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        if cur_loss < 1e-14:
            break
    
    Ua = _snap_half(best_U)
    Va = _snap_half(best_V)
    Wa = _snap_half(best_W)
    
    U, V, W = Ua.tolist(), Va.tolist(), Wa.tolist()
    err = _evaluate(U, V, W, n, m, p)
    if err < tol:
        return U, V, W
    return None


def _random_init(n, m, p, rank, seed):
    """Generate a random initialization for U, V, W with half-integer entries."""
    rng = np.random.RandomState(seed)
    U = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, rank))
    V = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, rank))
    W = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, rank))
    return U.tolist(), V.tolist(), W.tolist()


def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions."""
    best_result = None
    
    for rank in range(14, 36):
        if time.time() > deadline - 5.0:
            break
        
        # Phase 1: Fix-two-solve-third
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break
            result = _try_fix_two_solve_third(n, m, p, rank, deadline, seed=restart)
            if result is not None:
                best_result = result
                return result
        
        # Phase 2: Smooth polish from random init
        for restart in range(15):
            if time.time() > deadline - 5.0:
                break
            U, V, W = _random_init(n, m, p, rank, seed=1000 + restart)
            result = _smooth_polish(U, V, W, n, m, p, deadline)
            if result is not None:
                best_result = result
                return result
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.
    
    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (9, rank)
    V: (12, rank)
    W: (12, rank)
    
    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS
    
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-36
    return _schoolbook()
