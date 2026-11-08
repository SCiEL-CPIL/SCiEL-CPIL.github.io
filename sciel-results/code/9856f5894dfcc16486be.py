import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, T_target):
    """
    Fast vectorized verification.
    """
    T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_calc, T_target, atol=1e-9)

def _construct_tensor(n, m, p):
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    return T_target

def _greedy_rank1_residual(T, U_list, V_list, W_list, I, J, K):
    """
    Compute the residual tensor after subtracting the current rank-1 components.
    T is the target tensor (I, J, K).
    U_list, V_list, W_list are lists of vectors already extracted.
    Returns the residual tensor.
    """
    residual = T.copy()
    for u, v, w in zip(U_list, V_list, W_list):
        residual -= np.einsum('i,j,k->ijk', u, v, w)
    return residual

def _project_orthogonal(vec, basis):
    """
    Project vec onto the orthogonal complement of the span of basis vectors.
    basis is a list of 1D arrays.
    """
    if len(basis) == 0:
        return vec.copy()
    
    # Build the projection matrix
    B = np.array(basis)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-15)
    
    # Orthonormalize using Gram-Schmidt
    Q = []
    for b in B:
        q = b.copy()
        for q_prev in Q:
            q -= np.dot(q, q_prev) * q_prev
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            Q.append(q / norm)
    
    if len(Q) == 0:
        return vec.copy()
    
    Q = np.array(Q)
    proj = Q.T @ vec
    return vec - Q @ proj

def _find_best_rank1(T, I, J, K, max_candidates=200):
    """
    Find the best rank-1 approximation of tensor T using a greedy approach.
    Returns (u, v, w) unit vectors that maximize ||T - u⊗v⊗w|| reduction.
    """
    best_norm = np.inf
    best_u, best_v, best_w = None, None, None
    
    # Try random directions
    np.random.seed(42)
    for _ in range(max_candidates):
        u = np.random.randn(I)
        v = np.random.randn(J)
        w = np.random.randn(K)
        
        # Normalize
        u = u / (np.linalg.norm(u) + 1e-15)
        v = v / (np.linalg.norm(v) + 1e-15)
        w = w / (np.linalg.norm(w) + 1e-15)
        
        # Compute residual norm after subtracting u⊗v⊗w
        T_approx = np.einsum('i,j,k->ijk', u, v, w)
        residual = T - T_approx
        norm = np.linalg.norm(residual)
        
        if norm < best_norm:
            best_norm = norm
            best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
    
    # Also try structured directions based on the tensor's nonzero entries
    nonzero_indices = np.argwhere(T != 0)
    if len(nonzero_indices) > 0:
        for idx in nonzero_indices[:50]:
            i, j, k = idx
            u = np.zeros(I); u[i] = 1.0
            v = np.zeros(J); v[j] = 1.0
            w = np.zeros(K); w[k] = 1.0
            
            T_approx = np.einsum('i,j,k->ijk', u, v, w)
            residual = T - T_approx
            norm = np.linalg.norm(residual)
            
            if norm < best_norm:
                best_norm = norm
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
    
    return best_u, best_v, best_w, best_norm

def _als_decompose(T, rank, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    T is shape (I, J, K).
    Returns U (I, rank), V (J, rank), W (K, rank).
    """
    I, J, K = T.shape
    # Initialize with random values
    U = np.random.randn(I, rank)
    V = np.random.randn(J, rank)
    W = np.random.randn(K, rank)
    
    prev_err = np.inf
    for _ in range(max_iter):
        # Solve for U
        VW = np.einsum('jr,kr->jkr', V, W).reshape(J*K, rank)
        T_flat = T.reshape(I, J*K)
        try:
            U = np.linalg.lstsq(VW, T_flat.T, rcond=None)[0].T
        except:
            pass
            
        # Solve for V
        UW = np.einsum('ir,kr->ikr', U, W).reshape(I*K, rank)
        T_trans = np.transpose(T, (1, 0, 2)).reshape(J, I*K)
        try:
            V = np.linalg.lstsq(UW, T_trans.T, rcond=None)[0].T
        except:
            pass
            
        # Solve for W
        UV = np.einsum('ir,jr->ijr', U, V).reshape(I*J, rank)
        T_trans = np.transpose(T, (2, 0, 1)).reshape(K, I*J)
        try:
            W = np.linalg.lstsq(UV, T_trans.T, rcond=None)[0].T
        except:
            pass
        
        # Calculate error
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_calc - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _round_to_half_int(arr):
    # Round to nearest 0.5
    return np.round(arr * 2) / 2

def _greedy_decompose(T, target_rank, time_budget=30.0):
    """
    Greedy rank-reduction algorithm with orthogonal projection.
    T is the target tensor (I, J, K).
    Returns U (I, target_rank), V (J, target_rank), W (K, target_rank) or None if not found.
    """
    I, J, K = T.shape
    start_time = time.time()
    
    U_list = []
    V_list = []
    W_list = []
    residual = T.copy()
    
    for r in range(target_rank):
        if time.time() - start_time > time_budget:
            return None
        
        # Find best rank-1 component of the residual
        u, v, w, norm = _find_best_rank1(residual, I, J, K, max_candidates=100)
        
        if u is None:
            return None
        
        # Check if the component is significant
        if norm >= np.linalg.norm(residual) - 1e-10:
            # No improvement, stop
            break
        
        # Add to lists
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Update residual
        residual -= np.einsum('i,j,k->ijk', u, v, w)
        
        # If residual is essentially zero, we're done
        if np.linalg.norm(residual) < 1e-10:
            break
    
    if len(U_list) == 0:
        return None
    
    # Build the matrices
    U_mat = np.column_stack(U_list)
    V_mat = np.column_stack(V_list)
    W_mat = np.column_stack(W_list)
    
    # Verify the decomposition
    T_calc = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
    if np.allclose(T_calc, T, atol=1e-8):
        return U_mat, V_mat, W_mat
    
    return None

def _try_exact_half_int(U, V, W, T_target, n, m, p):
    """
    Try to find an exact half-integer decomposition by rounding and verifying.
    """
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    
    if _verify(U_r, V_r, W_r, n, m, p, T_target):
        return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    T_target = _construct_tensor(n, m, p)
    
    # Try ranks from 29 down to 15
    ranks_to_try = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Strategy 1: Greedy decomposition with orthogonal projection
        if time.time() < deadline - 20:
            result = _greedy_decompose(T_target, rank, time_budget=15.0)
            if result is not None:
                U, V, W = result
                exact = _try_exact_half_int(U, V, W, T_target, n, m, p)
                if exact is not None:
                    U_r, V_r, W_r = exact
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        # Strategy 2: ALS with multiple restarts
        if not found and time.time() < deadline - 10:
            max_restarts = 30
            for _ in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                    
                try:
                    U, V, W = _als_decompose(T_target, rank, max_iter=150, tol=1e-12)
                    exact = _try_exact_half_int(U, V, W, T_target, n, m, p)
                    if exact is not None:
                        U_r, V_r, W_r = exact
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                        break
                except Exception:
                    continue
        
        if found:
            break
    
    return best

# EVOLVE-BLOCK-END
