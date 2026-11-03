import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s

def _time_ok():
    return time.time() - _START_TIME < TIME_LIMIT

def _build_target_tensor():
    """Build the target matrix multiplication tensor T[i, j, k] for A(n x m) * B(m x p) = C(n x p).
    U index: i*n + j (n*m entries)
    V index: j*p + k (m*p entries)
    W index: k*n + i (p*n entries)
    T[a_idx, b_idx, w_idx] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (k,i) entry of C.
    Actually T is indexed by (a_idx, b_idx, w_idx) where:
    a_idx = i*m + j  (A is n x m)
    b_idx = j*p + k  (B is m x p)
    w_idx = k*n + i  (C is n x p, stored as W[k*n+i])
    T[a_idx, b_idx, w_idx] = 1 iff j == j (they share the middle index) which is always true for valid a,b.
    Wait, the tensor is: T[i,j,k] = 1 for the multiplication.
    The factorization is: sum_r U[a_idx, r] * V[b_idx, r] * W[w_idx, r] = T[a_idx, b_idx, w_idx]
    T[a_idx, b_idx, w_idx] = 1 if a_idx and b_idx share the same j, and w_idx is consistent.
    a_idx = i*m + j, b_idx = j'*p + k, w_idx = k'*n + i'
    T = 1 iff j == j', k == k', i == i'
    """
    n, m, p = DIMENSIONS
    # T is a 3D tensor of size (n*m) x (m*p) x (p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                w_idx = k * n + i
                T[a_idx, b_idx, w_idx] = 1.0
    return T

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Compute sum_r U[:,r] * V[:,r] * W[:,r] as a tensor
    # Using einsum: U[a,r] * V[b,r] * W[w,r] -> T[a,b,w]
    result = np.einsum('ar,br,wr->abw', U, V, W, optimize=True)
    return np.allclose(result, T, atol=1e-9, rtol=1e-9)

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_indices, v_indices, w_indices) where indices are into the 2x2=4 elements.
    u: (i, j) -> index i*2+j
    v: (j, k) -> index j*2+k
    w: (k, i) -> index k*2+i
    """
    terms = []
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 -= M1, C21 += M1, C22 += M1
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices (w): 0=(0,0)->k=0,i=0; 1=(0,1)->k=0,i=1; 2=(1,0)->k=1,i=0; 3=(1,1)->k=1,i=1
    
    # M1: U: (0,1),(3,1); V: (0,1),(3,1); W: (0,1),(1,-1),(2,1),(3,1)
    terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,-1),(2,1),(3,1)]))
    # M2: U: (2,1),(3,1); V: (0,1); W: (2,1),(3,-1)
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]))
    # M3: U: (0,1); V: (1,1),(3,-1); W: (1,1),(3,1)
    terms.append(([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]))
    # M4: U: (3,1); V: (2,1),(0,-1); W: (0,1),(2,1)
    terms.append(([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]))
    # M5: U: (0,1),(1,1); V: (3,1); W: (0,-1),(1,1)
    terms.append(([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]))
    # M6: U: (2,1),(0,-1); V: (0,1),(1,1); W: (3,1)
    terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]))
    # M7: U: (1,1),(3,-1); V: (2,1),(3,1); W: (0,1)
    terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
    
    return terms

def _construct_strassen_2x4x6():
    """Construct rank 42 decomposition using 6 blocks of 2x2x2 Strassen."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Define 6 blocks
    # A is 2x4, B is 4x6
    # We split A into left (cols 0-1) and right (cols 2-3)
    # We split B into top (rows 0-1) and bottom (rows 2-3)
    # And into col blocks: 0-1, 2-3, 4-5
    
    blocks = [
        # (a_row_range, a_col_range, b_row_range, b_col_range)
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx, col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx, col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx, col] += coeff
            
            col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_alternating_least_squares(target_tensor, rank, max_iters=500, tol=1e-8, seed=42):
    """Try to find a rank decomposition using ALS.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    rng = np.random.RandomState(seed)
    
    for it in range(max_iters):
        if not _time_ok():
            return None, False
        
        # Update W: for each (a, b), W[:, ?] such that sum_r U[a,r] V[b,r] W[w,r] = T[a,b,w]
        # For each r: sum_w T[a,b,w] * U[a,r] * V[b,r] = sum_w W[w,r] * (U[a,r] * V[b,r]) ... no
        # The equation is: sum_r U[a,r] V[b,r] W[w,r] = T[a,b,w]
        # This is not separable in the usual ALS way for 3-tensors with this structure.
        # Actually, for fixed U and V, for each w:
        # T[:, :, w] = sum_r (U[:, r] * V[:, r]) * W[w, r]
        # So for each w, we have a linear system: T[:, :, w] = M_w @ W[w, :]
        # where M_w[a, b, r] = U[a, r] * V[b, r]
        
        # Reshape T to (n*m, m*p, p*n)
        # For each w in range(p*n):
        #   T_abw = T[:, :, w]  # shape (n*m, m*p)
        #   We want W[w, :] (shape (rank,)) such that:
        #   sum_r U[:, r] * V[:, r] * W[w, r] = T_abw
        #   Let M be matrix of shape (n*m * m*p, rank) where M[a*m*p + b, r] = U[a, r] * V[b, r]
        #   Then M @ W[w, :] = T_abw.ravel()
        
        T_flat = target_tensor.reshape(-1, p * n)  # shape (n*m*m*p, p*n)
        
        for w in range(p * n):
            if not _time_ok():
                return None, False
            # Build M: shape (n*m*m*p, rank)
            # M[idx, r] = U[a, r] * V[b, r] where idx = a * (m*p) + b
            # We can build this efficiently:
            # U_rep = np.tile(U, (1, 1, m*p)) ... too big
            # U_expanded = U[:, :, np.newaxis] * V[np.newaxis, :, :] -> shape (n*m, rank, m*p)
            # Reshape to (n*m * m*p, rank)
            U_exp = U[:, :, np.newaxis]  # (n*m, rank, 1)
            V_exp = V[np.newaxis, :, :]  # (1, m*p, rank)
            M = (U_exp * V_exp).reshape(n * m * m * p, rank)
            
            # Solve least squares: M @ W[w, :] = T_flat[:, w]
            try:
                W[w, :], residuals, rank_solve, sv = np.linalg.lstsq(M, T_flat[:, w], rcond=None)
            except:
                pass
        
        # Similarly update U and V
        # For each a: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
        # V_W_expanded[r, b, w] = V[b, r] * W[w, r] -> shape (rank, m*p, p*n)
        # T[a, b, w] = sum_r U[a, r] * V_W_expanded[r, b, w]
        # For each a: T[a].ravel() = M_a @ U[a, :]
        # M_a shape: (m*p * p*n, rank), M_a[b * p*n + w, r] = V[b, r] * W[w, r]
        
        T_reshaped = target_tensor.reshape(n * m, m * p, p * n)
        
        for a in range(n * m):
            if not _time_ok():
                return None, False
            VW = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(rank, m * p * p * n)
            M_a = VW.T  # shape (m*p*p*n, rank)
            try:
                U[a, :], residuals, rank_solve, sv = np.linalg.lstsq(M_a, T_reshaped[a].ravel(), rcond=None)
            except:
                pass
        
        # Update V
        # For each b: T[:, b, :] = sum_r V[b, r] * (U[:, r] * W[:, r])
        # U_W_expanded[r, a, w] = U[a, r] * W[w, r] -> shape (rank, n*m, p*n)
        # T[a, b, w] = sum_r V[b, r] * U_W_expanded[r, a, w]
        # For each b: T[:, b, :].ravel() = M_b @ V[b, :]
        # M_b shape: (n*m * p*n, rank), M_b[a * p*n + w, r] = U[a, r] * W[w, r]
        
        for b in range(m * p):
            if not _time_ok():
                return None, False
            UW = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(rank, n * m * p * n)
            M_b = UW.T  # shape (n*m*p*n, rank)
            try:
                V[b, :], residuals, rank_solve, sv = np.linalg.lstsq(M_b, T_reshaped[:, b, :].ravel(), rcond=None)
            except:
                pass
        
        # Check convergence
        result = np.einsum('ar,br,wr->abw', U, V, W, optimize=True)
        error = np.linalg.norm(result - target_tensor) / (np.linalg.norm(target_tensor) + 1e-15)
        if error < tol:
            return (U.tolist(), V.tolist(), W.tolist()), True
    
    return None, False

def _try_rank_reduction(starting_rank, target_rank, max_attempts=10):
    """Try to find a decomposition of target_rank starting from a valid higher-rank one."""
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # First get a valid starting decomposition
    U_start, V_start, W_start = _construct_strassen_2x4x6()
    if not _verify(U_start, V_start, W_start):
        return None
    
    for attempt in range(max_attempts):
        if not _time_ok():
            return None
        
        # Randomly select target_rank columns from the starting decomposition
        # and try to fit the remaining part with a low-rank completion
        indices = list(range(starting_rank))
        random.shuffle(indices)
        kept = indices[:target_rank]
        
        U_kept = np.array(U_start)[:, kept]
        V_kept = np.array(V_start)[:, kept]
        W_kept = np.array(W_start)[:, kept]
        
        # Compute residual
        residual = T - np.einsum('ar,br,wr->abw', U_kept, V_kept, W_kept, optimize=True)
        
        # Try to decompose the residual with rank (starting_rank - target_rank) extra terms
        extra_rank = starting_rank - target_rank
        if extra_rank <= 0:
            if _verify(U_kept.tolist(), V_kept.tolist(), W_kept.tolist()):
                return (U_kept.tolist(), V_kept.tolist(), W_kept.tolist())
            continue
        
        # This approach is flawed - we can't just drop terms.
        # Instead, let's try ALS directly for the target rank.
        break
    
    # Try ALS directly
    for attempt in range(max_attempts):
        if not _time_ok():
            return None
        result, success = _try_alternating_least_squares(
            T, target_rank, max_iters=200, tol=1e-8, seed=42 + attempt
        )
        if success and result:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
    
    return None

def _snap_to_half_integer(M):
    """Snap values to nearest half-integer."""
    return np.round(M * 2) / 2

def _try_polish(U, V, W):
    """Try to polish a solution to have exact half-integer entries."""
    n, m, p = DIMENSIONS
    U_s = _snap_to_half_integer(np.array(U, dtype=float))
    V_s = _snap_to_half_integer(np.array(V, dtype=float))
    W_s = _snap_to_half_integer(np.array(W, dtype=float))
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist()):
        return (U_s.tolist(), V_s.tolist(), W_s.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known valid rank 42 solution
    U, V, W = _construct_strassen_2x4x6()
    best_rank = 42
    best_result = (U, V, W)
    
    if not _verify(best_result[0], best_result[1], best_result[2]):
        # Fallback: schoolbook rank 48
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    w_idx = k * n + i
                    U[a_idx, idx] = 1.0
                    V[b_idx, idx] = 1.0
                    W[w_idx, idx] = 1.0
                    idx += 1
        best_result = (U.tolist(), V.tolist(), W.tolist())
        best_rank = 48
        if _verify(best_result[0], best_result[1], best_result[2]):
            pass
    
    # Try to improve with ALS at lower ranks
    for target_rank in [40, 38, 36, 34, 32, 30, 28]:
        if not _time_ok():
            break
        if target_rank >= best_rank:
            continue
        
        result = _try_rank_reduction(best_rank, target_rank, max_attempts=5)
        if result:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r):
                # Try to polish
                polished = _try_polish(U_r, V_r, W_r)
                if polished:
                    best_result = polished
                else:
                    best_result = (U_r, V_r, W_r)
                best_rank = target_rank
                break  # Found a better rank, stop trying lower ones
    
    # Final check
    U_final, V_final, W_final = best_result
    if not _verify(U_final, V_final, W_final):
        # Use the safe rank 42
        U_final, V_final, W_final = _construct_strassen_2x4x6()
        if not _verify(U_final, V_final, W_final):
            # Last resort: schoolbook
            U_final = np.zeros((n * m, n * m * p), dtype=float)
            V_final = np.zeros((m * p, n * m * p), dtype=float)
            W_final = np.zeros((p * n, n * m * p), dtype=float)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_final[i * m + j, idx] = 1.0
                        V_final[j * p + k, idx] = 1.0
                        W_final[k * n + i, idx] = 1.0
                        idx += 1
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
