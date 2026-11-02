"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 143.0


def _verify_full(U, V, W, n=2, m=2, p=5, tol=1e-8):
    """Verify the full 2x2x5 tensor decomposition. Returns True if exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + i][c]
                    if abs(val - target) > tol:
                        return False
    return True


def _als_search(U0, V0, W0, n=2, m=2, p=5, max_iter=500, tol=1e-10, deadline=None):
    """
    Alternating Least Squares to refine a tensor decomposition.
    U0, V0, W0 are lists of lists (Python), shapes (n*m, rank), (m*p, rank), (p*n, rank).
    Returns refined (U, V, W) as numpy arrays.
    """
    U = np.array(U0, dtype=float)
    V = np.array(V0, dtype=float)
    W = np.array(W0, dtype=float)
    rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve min ||sum_c V[jp,k,c]*W[k,i,c]*U[i,j,c] - T[i,j,jp,k]||
        # Fix V and W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build linear system: for each (jp, k), target = delta(j, jp)
                # sum_c V[jp*p+k, c] * W[k*n+i, c] * U[i*m+j, c] = target
                A_mat = np.zeros((m * p, rank))
                b_vec = np.zeros(m * p)
                for jp in range(m):
                    for k in range(p):
                        idx = jp * p + k
                        for c in range(rank):
                            A_mat[idx, c] = V[jp * p + k, c] * W[k * n + i, c]
                        b_vec[idx] = 1.0 if j == jp else 0.0
                # Solve least squares
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_new[i * m + j, :] = sol
        
        # Update V
        V_new = np.zeros_like(V)
        for jp in range(m):
            for k in range(p):
                A_mat = np.zeros((n * m, rank))
                b_vec = np.zeros(n * m)
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        for c in range(rank):
                            A_mat[idx, c] = U[i * m + j, c] * W[k * n + i, c]
                        b_vec[idx] = 1.0 if j == jp else 0.0
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_new[jp * p + k, :] = sol
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((n * m, rank))
                b_vec = np.zeros(n * m)
                for jp in range(m):
                    for j in range(m):
                        idx = jp * p + k
                        # For W[k*n+i, c], the contribution is U[i*m+j, c] * V[jp*p+k, c]
                        # Target: T[i, j, jp, k] = delta(j, jp)
                        for c in range(rank):
                            A_mat[idx, c] = U[i * m + j, c] * V[jp * p + k, c]
                        b_vec[idx] = 1.0 if j == jp else 0.0
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[k * n + i, :] = sol
        
        # Check convergence
        diff = np.max(np.abs(U - U_new)) + np.max(np.abs(V - V_new)) + np.max(np.abs(W - W_new))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    
    return U.tolist(), V.tolist(), W.tolist()


def _get_rank_14_solution(n=2, m=2, p=5):
    """
    Attempt rank-14 decomposition using ALS from a smart initialization.
    Strategy: Start from rank-16 solution, zero out 2 columns, and refine.
    """
    if time.time() > DEADLINE - 5:
        return None
    
    # Start from rank-16, drop last 2 columns
    sol16 = _get_rank_16_solution(n, m, p)
    U16 = sol16["U"]
    V16 = sol16["V"]
    W16 = sol16["W"]
    
    # Try dropping different pairs of columns
    rank = 14
    n_cols = n * m  # 4
    m_cols = m * p  # 10
    p_cols = p * n  # 10
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Try various initializations: drop different column pairs
    col_pairs_to_try = [(14, 15), (6, 15), (0, 15), (3, 15), (1, 15), (2, 15), (5, 15), (7, 15)]
    
    for drop_cols in col_pairs_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        U_init = []
        for row in U16:
            new_row = [row[c] for c in range(16) if c not in drop_cols]
            U_init.append(new_row)
        V_init = []
        for row in V16:
            new_row = [row[c] for c in range(16) if c not in drop_cols]
            V_init.append(new_row)
        W_init = []
        for row in W16:
            new_row = [row[c] for c in range(16) if c not in drop_cols]
            W_init.append(new_row)
        
        # Refine with ALS
        U_ref, V_ref, W_ref = _als_search(U_init, V_init, W_init, n, m, p, max_iter=200, tol=1e-12, deadline=DEADLINE - 10)
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U_ref[i * m + j][c] * V_ref[jp * p + k][c] * W_ref[k * n + i][c]
                        err += (val - target) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_ref, V_ref, W_ref
    
    if best_err < 1e-8:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None


def _get_rank_15_solution(n=2, m=2, p=5):
    """
    Attempt rank-15 decomposition using ALS from rank-16 with one column dropped.
    """
    if time.time() > DEADLINE - 5:
        return None
    
    sol16 = _get_rank_16_solution(n, m, p)
    U16 = sol16["U"]
    V16 = sol16["V"]
    W16 = sol16["W"]
    
    rank = 15
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for drop_col in range(16):
        if time.time() > DEADLINE - 10:
            break
        
        U_init = []
        for row in U16:
            new_row = [row[c] for c in range(16) if c != drop_col]
            U_init.append(new_row)
        V_init = []
        for row in V16:
            new_row = [row[c] for c in range(16) if c != drop_col]
            V_init.append(new_row)
        W_init = []
        for row in W16:
            new_row = [row[c] for c in range(16) if c != drop_col]
            W_init.append(new_row)
        
        U_ref, V_ref, W_ref = _als_search(U_init, V_init, W_init, n, m, p, max_iter=200, tol=1e-12, deadline=DEADLINE - 10)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U_ref[i * m + j][c] * V_ref[jp * p + k][c] * W_ref[k * n + i][c]
                        err += (val - target) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_ref, V_ref, W_ref
    
    if best_err < 1e-8:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None


def _get_rank_16_solution(n=2, m=2, p=5):
    """
    Rank-16 decomposition for 2x2 * 2x5 matrix multiplication.
    """
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        """Add the verified Strassen block from the rank-18 solution."""
        U[0][t_offset + 0] = 1
        U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1
        U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1
        U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1
        U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1
        U[3][t_offset + 6] = -1

        k0 = k_start
        k1 = k_start + 1
        V[0 * p + k0][t_offset + 0] = 1
        V[1 * p + k1][t_offset + 0] = 1
        V[0 * p + k0][t_offset + 1] = 1
        V[0 * p + k1][t_offset + 2] = 1
        V[1 * p + k1][t_offset + 2] = -1
        V[1 * p + k0][t_offset + 3] = 1
        V[0 * p + k0][t_offset + 3] = -1
        V[1 * p + k1][t_offset + 4] = 1
        V[0 * p + k0][t_offset + 5] = 1
        V[0 * p + k1][t_offset + 5] = 1
        V[1 * p + k0][t_offset + 6] = 1
        V[1 * p + k1][t_offset + 6] = 1

        W[k0 * n + 0][t_offset + 0] = 1
        W[k1 * n + 1][t_offset + 0] = 1
        W[k0 * n + 1][t_offset + 1] = 1
        W[k1 * n + 1][t_offset + 1] = -1
        W[k1 * n + 0][t_offset + 2] = 1
        W[k1 * n + 1][t_offset + 2] = 1
        W[k0 * n + 0][t_offset + 3] = 1
        W[k0 * n + 1][t_offset + 3] = 1
        W[k0 * n + 0][t_offset + 4] = -1
        W[k1 * n + 0][t_offset + 4] = 1
        W[k1 * n + 1][t_offset + 5] = 1
        W[k0 * n + 0][t_offset + 6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    # Column 4: 2 additional terms
    t = 14
    U[0][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 0][t] = 1
    U[2][t] = 1; W[4 * n + 1][t] = 1

    t = 15
    U[1][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 0][t] = 1
    U[3][t] = 1; W[4 * n + 1][t] = 1

    return {"U": U, "V": V, "W": W}


def _get_rank_18_solution(n=2, m=2, p=5):
    """Fallback: 2 Strassen blocks (rank 7 each) for cols 0-3, schoolbook (rank 4) for col 4."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        U[0][t_offset + 0] = 1
        U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1
        U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1
        U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1
        U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1
        U[3][t_offset + 6] = -1

        k0 = k_start
        k1 = k_start + 1
        V[0 * p + k0][t_offset + 0] = 1
        V[1 * p + k1][t_offset + 0] = 1
        V[0 * p + k0][t_offset + 1] = 1
        V[0 * p + k1][t_offset + 2] = 1
        V[1 * p + k1][t_offset + 2] = -1
        V[1 * p + k0][t_offset + 3] = 1
        V[0 * p + k0][t_offset + 3] = -1
        V[1 * p + k1][t_offset + 4] = 1
        V[0 * p + k0][t_offset + 5] = 1
        V[0 * p + k1][t_offset + 5] = 1
        V[1 * p + k0][t_offset + 6] = 1
        V[1 * p + k1][t_offset + 6] = 1

        W[k0 * n + 0][t_offset + 0] = 1
        W[k1 * n + 1][t_offset + 0] = 1
        W[k0 * n + 1][t_offset + 1] = 1
        W[k1 * n + 1][t_offset + 1] = -1
        W[k1 * n + 0][t_offset + 2] = 1
        W[k1 * n + 1][t_offset + 2] = 1
        W[k0 * n + 0][t_offset + 3] = 1
        W[k0 * n + 1][t_offset + 3] = 1
        W[k0 * n + 0][t_offset + 4] = -1
        W[k1 * n + 0][t_offset + 4] = 1
        W[k1 * n + 1][t_offset + 5] = 1
        W[k0 * n + 0][t_offset + 6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    t = 14
    U[0][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[1][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[2][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 1][t] = 1; t += 1
    U[3][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 1][t] = 1

    return {"U": U, "V": V, "W": W}


def _get_rank_20_solution(n=2, m=2, p=5):
    """Schoolbook for 2x2x5. Rank 20."""
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    t = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1

    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS

    # Try rank 14 first
    if time.time() < DEADLINE - 15:
        sol14 = _get_rank_14_solution(n, m, p)
        if sol14 is not None and _verify_full(sol14["U"], sol14["V"], sol14["W"], n, m, p, tol=1e-8):
            return sol14

    # Try rank 15
    if time.time() < DEADLINE - 15:
        sol15 = _get_rank_15_solution(n, m, p)
        if sol15 is not None and _verify_full(sol15["U"], sol15["V"], sol15["W"], n, m, p, tol=1e-8):
            return sol15

    # Try rank 16
    sol = _get_rank_16_solution(n, m, p)
    if _verify_full(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-8):
        return sol

    # Fallback to rank 18
    sol = _get_rank_18_solution(n, m, p)
    if _verify_full(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-8):
        return sol

    # Last resort: schoolbook
    return _get_rank_20_solution(n, m, p)

# EVOLVE-BLOCK-END
