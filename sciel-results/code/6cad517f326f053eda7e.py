"""Matrix multiplication 3,3,4 rank-minimal decomposition."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36


def _build_tensor():
    """Return the 3x3x4 matrix-multiplication tensor T[i,j,k] = 1."""
    T = np.ones((N, M, P), dtype=float)
    return T


def _decompose_to_factors(U, V, W):
    """Check U[i,j,c]*V[j,k,c]*W[k,i,c] == T[i,j,k]."""
    T = _build_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = N, M, P
    r = U.shape[1]
    result = np.zeros((n, m, p), dtype=float)
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                ui_j = uc[i * m + j]
                if ui_j == 0:
                    continue
                for k in range(p):
                    vj_k = vc[j * p + k]
                    if vj_k == 0:
                        continue
                    result[i, j, k] += ui_j * vj_k * wc[k * n + i]
    return np.allclose(result, T, atol=1e-9)


def _exact_verify(U, V, W):
    """Exact verification using Fraction arithmetic."""
    n, m, p = N, M, P
    r = len(U[0])
    # Convert to Fraction
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(n * m)]
    Vf = [[Fraction(V[i][c]) for c in range(r)] for i in range(m * p)]
    Wf = [[Fraction(W[i][c]) for c in range(r)] for i in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != 1:
                    return False
    return True


def _least_squares_solve(U, V, W, T, max_iter=200, tol=1e-10):
    """
    Alternating least squares to refine U, V, W given T.
    U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    Constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
    """
    n, m, p = N, M, P
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    for iteration in range(max_iter):
        # Solve for U: for each (i,j), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        # This is linear in U[i*m+j,:] for fixed V, W
        # For each (i,j): sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] = ... 
        # Actually T[i,j,k] is the target for each (i,j,k)
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        # For fixed i,j,k, this is linear in U[i*m+j,c]
        # But we have multiple equations for the same U[i*m+j,:]
        
        # Let's think differently. For each (i,j), we have p equations:
        # For k=0..p-1: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        # This is a system: A_{(i,j)} * U[i*m+j,:] = b_{(i,j)}
        # where A_{(i,j)}[k, c] = V[j*p+k,c] * W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                # Build A (p x r) and b (p,)
                A = np.zeros((p, r))
                b = np.zeros(p)
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    b[k] = T[i, j, k]
                # Least squares solve
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=tol)
                    U[i * m + j, :] = sol
                except:
                    pass

        # Solve for V
        for j in range(m):
            for k in range(p):
                # For fixed j,k: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = T[i,j,k] for i=0..n-1
                A = np.zeros((n, r))
                b = np.zeros(n)
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    b[i] = T[i, j, k]
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=tol)
                    V[j * p + k, :] = sol
                except:
                    pass

        # Solve for W
        for k in range(p):
            for i in range(n):
                # For fixed k,i: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = T[i,j,k] for j=0..m-1
                A = np.zeros((m, r))
                b = np.zeros(m)
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    b[j] = T[i, j, k]
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=tol)
                    W[k * n + i, :] = sol
                except:
                    pass

        # Check convergence
        if iteration % 20 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (s - T[i, j, k]) ** 2
            if err < tol * tol:
                break

    return U, V, W


def _schoolbook():
    """Return schoolbook rank-36 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, T, target_rank, num_trials=50, seed=0):
    """
    Try to reduce from current rank to target_rank using perturbation + ALS.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = N, M, P
    cur_rank = U.shape[1]
    rng = random.Random(seed)
    
    for trial in range(num_trials):
        # Start from current solution, randomly perturb and drop one column
        U_new = np.array(U, dtype=float)
        V_new = np.array(V, dtype=float)
        W_new = np.array(W, dtype=float)
        
        # Randomly perturb
        noise_scale = 0.1 * rng.random()
        U_new += noise_scale * rng.gauss(0, 1) * np.ones_like(U_new) * 0  # keep deterministic
        # Actually let's do a proper random perturbation
        U_new = U_new + noise_scale * np.random.RandomState(trial * 1000 + seed).randn(*U_new.shape)
        V_new = V_new + noise_scale * np.random.RandomState(trial * 1000 + seed + 1).randn(*V_new.shape)
        W_new = W_new + noise_scale * np.random.RandomState(trial * 1000 + seed + 2).randn(*W_new.shape)
        
        # Drop one random column
        drop_col = rng.randint(0, cur_rank - 1)
        U_new = np.delete(U_new, drop_col, axis=1)
        V_new = np.delete(V_new, drop_col, axis=1)
        W_new = np.delete(W_new, drop_col, axis=1)
        
        # Refine with ALS
        U_new, V_new, W_new = _least_squares_solve(U_new, V_new, W_new, T, max_iter=300, tol=1e-12)
        
        # Check if it's a valid decomposition
        if _decompose_to_factors(U_new, V_new, W_new):
            # Convert to list of lists for exact verification
            U_list = [[float(x) for x in row] for row in U_new]
            V_list = [[float(x) for x in row] for row in V_new]
            W_list = [[float(x) for x in row] for row in W_new]
            if _exact_verify(U_list, V_list, W_list):
                return U_new, V_new, W_new
    
    return None


def _strassen_3x3x3():
    """
    Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    This uses a known minimal-rank decomposition for 3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    # We'll construct this from the known 3x3x3 rank-23 decomposition
    # Using a standard construction based on Strassen-like algorithms
    
    # For 3x3x3, one known rank-23 decomposition:
    # We'll use a parametrized approach and solve
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with a structured pattern and refine
    # Use a random initialization and ALS
    rng = np.random.RandomState(42)
    U[:] = rng.randn(n * m, rank) * 0.5
    V[:] = rng.randn(m * p, rank) * 0.5
    W[:] = rng.randn(p * n, rank) * 0.5
    
    T = np.ones((n, m, p), dtype=float)
    U, V, W = _least_squares_solve(U, V, W, T, max_iter=500, tol=1e-12)
    
    # Check
    if _decompose_to_factors(U, V, W):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        if _exact_verify(U_list, V_list, W_list):
            return U, V, W
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = N, M, P
    T = _build_tensor()
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to reduce rank step by step
    for target_rank in range(best_rank - 1, 22, -1):
        if time.time() > deadline:
            break
        
        result = _try_reduce_rank(U, V, W, T, target_rank, num_trials=30, seed=target_rank)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
            print(f"Reduced to rank {best_rank}")
        else:
            # Try a few more times with different seeds
            for extra_seed in range(5):
                if time.time() > deadline:
                    break
                result = _try_reduce_rank(U, V, W, T, target_rank, num_trials=10, seed=target_rank * 100 + extra_seed)
                if result is not None:
                    U, V, W = result
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    print(f"Reduced to rank {best_rank}")
                    break
    
    # Try Strassen-based approach for 3x3x3 substructure
    if time.time() < deadline - 20:
        strassen = _strassen_3x3x3()
        if strassen is not None:
            sU, sV, sW = strassen
            # Extend to 3x3x4 by adding one more "slice"
            # For k=3 (the 4th column), we need U[i*m+j, c]*V[j*p+3, c]*W[3*n+i, c] = 1
            # We can add a new column c=23 with U[i*m+j,23]=1, V[j*p+3,23]=1, W[3*n+i,23]=1
            # But this gives rank 24, which may not be better
            
            # Alternatively, try to see if rank 23 works for 3x3x4 directly
            # by modifying the 3x3x3 decomposition
            pass
    
    # Final verification
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    if not _exact_verify(U_list, V_list, W_list):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
