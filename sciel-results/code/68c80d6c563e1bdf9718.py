"""Rewritten program: structured search for low-rank exact tensor decomposition of 3x3x5 matmul."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds


def _verify_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j, c]) * Fraction(V[j*p+k, c]) * Fraction(W[k*n+i, c])
                if s != Fraction(1):
                    return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-10):
    """Fast float verification."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank_optimization(rank, n, m, p, seed=42, max_iter=200):
    """
    Try to find a decomposition at the given rank using coordinate descent / Gauss-Newton.
    The tensor equation is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    """
    np.random.seed(seed)
    random_seed = seed
    
    # Initialize with small random values
    U = np.random.randn(n*m, rank) * 0.3
    V = np.random.randn(m*p, rank) * 0.3
    W = np.random.randn(p*n, rank) * 0.3
    
    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # For efficiency, precompute which (i*m+j), (j*p+k), (k*n+i) indices each triple uses
    idx_U = [i*m+j for i, j, k in triples]
    idx_V = [j*p+k for i, j, k in triples]
    idx_W = [k*n+i for i, j, k in triples]
    
    def compute_residuals():
        """Compute residual vector: for each triple, sum_c U*V*W - 1"""
        res = np.zeros(num_triples)
        for c in range(rank):
            uc = U[idx_U, c]  # (num_triples,)
            vc = V[idx_V, c]
            wc = W[idx_W, c]
            res += uc * vc * wc
        res -= 1.0
        return res
    
    # Gauss-Newton style: at each step, fix two matrices and solve for the third
    # This is a linear least squares problem when two are fixed
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Fix U and V, solve for W: for each (i,j,k), sum_c W[k*n+i,c] * (U[i*m+j,c]*V[j*p+k,c]) = 1
    # This is a linear system in W.
    
    def solve_for_W(U_mat, V_mat):
        """Solve linear system for W given U and V."""
        # For each (i,j,k): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
        # Group by (k*n+i) index: for each w_idx = k*n+i, we have equations for all (i,j,k) with that w_idx
        # w_idx = k*n+i, so for fixed w_idx, i = w_idx % n, k = w_idx // n
        # Equations: for j in range(m): sum_c W[w_idx,c] * U[i*m+j,c] * V[j*p+k,c] = 1
        # So for each w_idx, we have m equations in rank unknowns (W[w_idx, :])
        
        W_mat = np.zeros((p*n, rank))
        for w_idx in range(p*n):
            k = w_idx // n
            i = w_idx % n
            # Build A matrix: m x rank, A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            A = np.zeros((m, rank))
            b = np.ones(m)
            for j in range(m):
                A[j, :] = U_mat[i*m+j, :] * V_mat[j*p+k, :]
            # Solve A @ W[w_idx,:] = b (least squares if rank > m)
            if rank <= m:
                # Underdetermined or square: use pinv
                W_mat[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            else:
                # Overdetermined: least squares
                W_mat[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return W_mat
    
    def solve_for_U(U_mat, V_mat, W_mat):
        """Solve linear system for U given V and W."""
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (i*m+j) index: u_idx = i*m+j, i = u_idx // m, j = u_idx % m
        # Equations: for k in range(p): sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # So for each u_idx, we have p equations in rank unknowns
        
        U_mat_new = np.zeros((n*m, rank))
        for u_idx in range(n*m):
            i = u_idx // m
            j = u_idx % m
            A = np.zeros((p, rank))
            b = np.ones(p)
            for k in range(p):
                A[k, :] = V_mat[j*p+k, :] * W_mat[k*n+i, :]
            U_mat_new[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return U_mat_new
    
    def solve_for_V(U_mat, V_mat, W_mat):
        """Solve linear system for V given U and W."""
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (j*p+k) index: v_idx = j*p+k, j = v_idx // p, k = v_idx % p
        # Equations: for i in range(n): sum_c V[v_idx,c] * U[i*m+j,c] * W[k*n+i,c] = 1
        # So for each v_idx, we have n equations in rank unknowns
        
        V_mat_new = np.zeros((m*p, rank))
        for v_idx in range(m*p):
            j = v_idx // p
            k = v_idx % p
            A = np.zeros((n, rank))
            b = np.ones(n)
            for i in range(n):
                A[i, :] = U_mat[i*m+j, :] * W_mat[k*n+i, :]
            V_mat_new[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return V_mat_new
    
    # Alternating least squares
    for iteration in range(max_iter):
        # Solve for W
        W = solve_for_W(U, V)
        # Solve for U
        U = solve_for_U(U, V, W)
        # Solve for V
        V = solve_for_V(U, V, W)
        
        # Check convergence
        res = compute_residuals()
        max_res = np.max(np.abs(res))
        if max_res < 1e-12:
            break
    
    # Final check
    res = compute_residuals()
    if np.max(np.abs(res)) < 1e-8:
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_float(U_r, V_r, W, n, m, p, tol=1e-6):
            return U_r, V_r, W_r
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if _verify_float(U_i, V_i, W_i, n, m, p, tol=1e-6):
            return U_i, V_i, W_i
        
        # Return the unrounded solution (should be close to rational)
        return U, V, W
    
    return None


def _polish_solution(U, V, W, n, m, p):
    """
    Try to polish a near-solution to exact integer/half-integer entries.
    Uses the fact that if the solution is close to half-integers, we can snap.
    """
    # Try snapping to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    if _verify_float(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
        return U_snap, V_snap, W_snap
    
    # Try integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify_float(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return U_int, V_int, W_int
    
    # Try to find a nearby half-integer solution via local search
    # For each element, try flipping to neighboring half-integer
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for _ in range(50):
        # Compute current error
        res = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += best_U[i*m+j, c] * best_V[j*p+k, c] * best_W[k*n+i, c]
                    res[i*m*p + j*p + k] = s - 1.0
        curr_err = np.sum(res**2)
        
        if curr_err < best_err:
            best_err = curr_err
        
        if curr_err < 1e-20:
            break
        
        # Try random local moves
        improved = False
        for _ in range(200):
            # Pick a random matrix and element
            mat = np.random.randint(3)
            if mat == 0:
                row = np.random.randint(n*m)
                col = np.random.randint(U.shape[1])
                old_val = best_U[row, col]
                # Try neighboring half-integers
                for delta in [-0.5, -0.25, 0.25, 0.5]:
                    new_val = np.round((old_val + delta) * 2) / 2
                    if new_val == old_val:
                        continue
                    best_U[row, col] = new_val
                    # Quick check
                    res_check = 0.0
                    for i in range(n):
                        for j in range(m):
                            if i*m+j == row:
                                for k in range(p):
                                    s = 0.0
                                    for c in range(U.shape[1]):
                                        s += best_U[i*m+j, c] * best_V[j*p+k, c] * best_W[k*n+i, c]
                                    res_check += (s - 1.0)**2
                    if res_check < curr_err:
                        curr_err = res_check
                        improved = True
                    else:
                        best_U[row, col] = old_val
            elif mat == 1:
                row = np.random.randint(m*p)
                col = np.random.randint(V.shape[1])
                old_val = best_V[row, col]
                for delta in [-0.5, -0.25, 0.25, 0.5]:
                    new_val = np.round((old_val + delta) * 2) / 2
                    if new_val == old_val:
                        continue
                    best_V[row, col] = new_val
                    res_check = 0.0
                    for i in range(n):
                        for j in range(m):
                            if j*p+k == row:
                                for k in range(p):
                                    s = 0.0
                                    for c in range(U.shape[1]):
                                        s += best_U[i*m+j, c] * best_V[j*p+k, c] * best_W[k*n+i, c]
                                    res_check += (s - 1.0)**2
                    if res_check < curr_err:
                        curr_err = res_check
                        improved = True
                    else:
                        best_V[row, col] = old_val
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(W.shape[1])
                old_val = best_W[row, col]
                for delta in [-0.5, -0.25, 0.25, 0.5]:
                    new_val = np.round((old_val + delta) * 2) / 2
                    if new_val == old_val:
                        continue
                    best_W[row, col] = new_val
                    res_check = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if k*n+i == row:
                                    s = 0.0
                                    for c in range(U.shape[1]):
                                        s += best_U[i*m+j, c] * best_V[j*p+k, c] * best_W[k*n+i, c]
                                    res_check += (s - 1.0)**2
                    if res_check < curr_err:
                        curr_err = res_check
                        improved = True
                    else:
                        best_W[row, col] = old_val
        
        if not improved:
            break
    
    # Final verification
    if _verify_float(best_U, best_V, best_W, n, m, p, tol=1e-9):
        return best_U, best_V, best_W
    
    return None


def _structured_search(rank, n, m, p, seed=42, time_limit=30.0):
    """
    Structured search leveraging block decomposition.
    For 3x3x5, we can view the 5 columns as blocks.
    """
    start_time = time.time()
    
    # Try multiple random initializations with alternating least squares
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_rank_optimization(rank, n, m, p, seed=seed + attempt * 1000, max_iter=100)
        if result is not None:
            U, V, W = result
            # Polish
            polished = _polish_solution(U, V, W, n, m, p)
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify_float(U_p, V_p, W_p, n, m, p, tol=1e-9):
                    return U_p, V_p, W_p
    
    return None


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 15.0  # 15s safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 45
    
    # Known: for 3x3x3, Strassen gives rank 7 (vs schoolbook 9)
    # For 3x3x5, we expect something less than 45
    # Lower bound considerations: 
    #   - Trivial: n*m*p = 45
    #   - For 3x3x3, min rank is 7 (Strassen)
    #   - For 3x3x5, a reasonable target is around 30-35
    #   - Let's search from low to high
    
    # Search ranks from low to high
    # Start with a reasonable lower bound
    # The tensor has n*m*p = 45 entries, each must be 1
    # Rank r means we have r rank-1 tensors
    # Each rank-1 tensor contributes to all 45 entries
    # So we need r such that the system is solvable
    
    # Try a range of ranks
    # For 3x3x3, rank 7 works (Strassen)
    # For 3x3x5, let's try from 20 upward
    min_rank_to_try = 20
    max_rank_to_try = best_rank - 1  # 44
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = _structured_search(rank, n, m, p, seed=42, 
                                    time_limit=min(10.0, max(2.0, deadline - time.time())))
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            # Found a decomposition at this rank; try to go lower
            break
    
    return best_result
# EVOLVE-BLOCK-END
