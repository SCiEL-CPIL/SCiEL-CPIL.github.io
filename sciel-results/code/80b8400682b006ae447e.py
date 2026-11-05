"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_3x3_init(n=3, m=3, p=6):
    """
    Build an initial decomposition using Strassen-like structure for the 3x3 part.
    This gives a rank-21 base (7 for 2x2 + 3 for the rest) times 2 for the 6 columns = 42.
    Actually, let's think more carefully.
    
    For 3x3 * 3x6:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can split the 3 columns of A into 2+1 and the 3 rows of B into 2+1.
    The 2x2 part can use Strassen (rank 7).
    The cross terms (2x1, 1x2, 1x1) need additional terms.
    
    Actually, a simpler approach: just use the standard rank-19 or rank-20 algorithms.
    But I don't have the coefficients memorized. Let me try a different approach.
    
    For now, let's just use schoolbook but with a better initialization for ALS.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook as base
    base = _schoolbook(n, m, p)
    
    # Try to compress using SVD on the tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape and SVD
    T_mat = T.reshape(nm, mp * pn)
    U_svd, S_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # Keep top 42 components (21*2, based on Strassen-like intuition)
    r = 42
    U_svd = U_svd[:, :r]
    S_vals = S_vals[:r]
    Vt_svd = Vt_svd[:r, :]
    
    # Reconstruct as a rank-r approximation
    # U_new[a, c] = U_svd[a, c] * S_vals[c]
    # V_new[b, c] * W_new[cp, c] = Vt_svd[c, b*pn+cp]
    # This is tricky. Let's just return schoolbook for now.
    return base


def _try_als_decomposition(n=3, m=3, p=6, deadline=None):
    """
    Try to find a low-rank decomposition using ALS with various initializations.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 36 upward (since 18 is too low, 36 is a reasonable target)
    for r in [36, 38, 40, 42]:
        if time.time() > deadline - 15:
            break
        
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart * 1000 + r)
            
            # Sparse random initialization
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for col in range(r):
                n_u = np.random.randint(2, 5)
                n_v = np.random.randint(2, 5)
                n_w = np.random.randint(2, 5)
                u_idx = np.random.choice(nm, n_u, replace=False)
                v_idx = np.random.choice(mp, n_v, replace=False)
                w_idx = np.random.choice(pn, n_w, replace=False)
                U[u_idx, col] = np.random.choice([-1, 1], n_u)
                V[v_idx, col] = np.random.choice([-1, 1], n_v)
                W[w_idx, col] = np.random.choice([-1, 1], n_w)
            
            # ALS iterations
            max_iter = 100
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                new_U = np.zeros_like(U)
                for a in range(nm):
                    rows = []
                    i = a // m
                    j = a % m
                    for k in range(p):
                        b_idx = j * p + k
                        c_idx = k * n + i
                        rows.append(V[b_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_U[a, :] = U[a, :]
                U = new_U
                
                # Update V
                new_V = np.zeros_like(V)
                for b in range(mp):
                    rows = []
                    j = b // p
                    k = b % p
                    for i in range(n):
                        a_idx = i * m + j
                        c_idx = k * n + i
                        rows.append(U[a_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_V[b, :] = V[b, :]
                V = new_V
                
                # Update W
                new_W = np.zeros_like(W)
                for c in range(pn):
                    rows = []
                    k = c // n
                    i = c % n
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        rows.append(U[a_idx, :] * V[b_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_W[c, :] = W[c, :]
                W = new_W
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                # Check off-support (sampled for speed)
                if it % 10 == 9:
                    for _ in range(100):
                        a = np.random.randint(0, nm)
                        b = np.random.randint(0, mp)
                        c = np.random.randint(0, pn)
                        i_a, j_a = a // m, a % m
                        j_b, k_b = b // p, b % p
                        k_c, i_c = c // n, c % n
                        if not (j_a == j_b and k_b == k_c and i_c == i_a):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += val ** 2
                
                if err < 1e-6:
                    # Verify exactly
                    exact_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                exact_err += (val - 1.0) ** 2
                    for a in range(nm):
                        i_a, j_a = a // m, a % m
                        for b in range(mp):
                            j_b, k_b = b // p, b % p
                            for c in range(pn):
                                k_c, i_c = c // n, c % n
                                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                                    exact_err += val ** 2
                    
                    if exact_err < 1e-6:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
        
        if best_result is not None:
            break
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to find better
        better = _try_als_decomposition(n, m, p, deadline)
        if better is not None:
            rank_better = len(better["U"][0]) if better["U"] else 0
            rank_current = len(result["U"][0]) if result["U"] else 0
            if rank_better < rank_current:
                result = better
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
