import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_rec = np.zeros(T.shape)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=tol)

def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded)) if arr.size > 0 else 0.0
    return rounded, max_err

def _try_round_and_verify(U, V, W, T):
    """Attempt to round U, V, W to half-integers and verify the decomposition."""
    U_r, e_u = _round_to_halfint(U)
    V_r, e_v = _round_to_halfint(V)
    W_r, e_w = _round_to_halfint(W)
    
    # If rounding error is small, check if it's an exact solution
    if max(e_u, e_v, e_w) < 1e-3:
        if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-6):
            return U_r, V_r, W_r
    return None

def _als_decompose(T, n, m, p, rank, max_iters=50, tol=1e-10, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    """
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    prev_loss = np.inf
    for it in range(max_iters):
        # Solve for U
        # T_flat shape: (n*m, m*p*p*n)
        # X shape: (rank, m*p*p*n)
        X = np.zeros((rank, m * p * p * n))
        for r in range(rank):
            X[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        T_flat = T.reshape(n * m, m * p * p * n)
        U_new, _, _, _ = np.linalg.lstsq(X.T, T_flat.T, rcond=None)
        U = U_new.T
        
        # Solve for V
        Y = np.zeros((rank, n * m * p * n))
        for r in range(rank):
            Y[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        T_stacked = np.zeros((m * p, n * m * p * n))
        for v in range(m * p):
            T_stacked[v, :] = T[:, v, :].flatten()
        V_new, _, _, _ = np.linalg.lstsq(Y.T, T_stacked.T, rcond=None)
        V = V_new.T
        
        # Solve for W
        Z = np.zeros((rank, n * m * m * p))
        for r in range(rank):
            Z[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        T_stacked_w = np.zeros((p * n, n * m * m * p))
        for w in range(p * n):
            T_stacked_w[w, :] = T[:, :, w].flatten()
        W_new, _, _, _ = np.linalg.lstsq(Z.T, T_stacked_w.T, rcond=None)
        W = W_new.T
        
        # Calculate loss
        T_rec = np.zeros(T.shape)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        loss = np.sum((T_rec - T) ** 2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
    return U, V, W, loss

def _schoolbook_3x3x2():
    """
    Returns U, V, W for M_{3,3} x M_{3,2} schoolbook decomposition.
    U shape: (9, 18)
    V shape: (6, 18)
    W shape: (6, 18)
    """
    n, m, p = 3, 3, 2
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Full dimensions
    N_full, M_full, P_full = n, m, p
    
    # Sub-problem 1: 3x3x3
    n1, m1, p1 = 3, 3, 3
    T1 = _build_target_tensor(n1, m1, p1)
    
    # Sub-problem 2: 3x3x2
    n2, m2, p2 = 3, 3, 2
    T2 = _build_target_tensor(n2, m2, p2)
    
    # Get schoolbook for sub-problem 2 (Rank 18)
    U2, V2, W2 = _schoolbook_3x3x2()
    
    best_U1, best_V1, best_W1, best_rank1 = None, None, None, 27
    
    # Try to find a better decomposition for 3x3x3
    # Known optimal rank is 23. Try 23, 24, 25, 26.
    for target_rank1 in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
            
        found = False
        n_restarts = 10 if target_rank1 <= 24 else 5
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 5000 + restart * 17 + target_rank1 * 11
            try:
                U1, V1, W1, loss = _als_decompose(T1, n1, m1, p1, target_rank1, max_iters=60, seed=seed)
                
                if loss < 1e-4:
                    res = _try_round_and_verify(U1, V1, W1, T1)
                    if res is not None:
                        U1_r, V1_r, W1_r = res
                        best_U1, best_V1, best_W1 = U1_r, V1_r, W1_r
                        best_rank1 = target_rank1
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # If we found a rank < 27, we can stop searching for higher ranks for this sub-problem
            # because we want the minimal rank.
            # However, if we found 23, we are done. If we found 24, maybe 23 is possible but we missed it?
            # Given time constraints, we accept the first found valid lower rank.
            break

    # If no better rank found, use schoolbook for 3x3x3 (Rank 27)
    if best_U1 is None:
        U1, V1, W1 = _schoolbook_3x3x2() # Wait, this is 3x3x2. Need 3x3x3 schoolbook.
        # Generate schoolbook for 3x3x3
        n1, m1, p1 = 3, 3, 3
        rank1 = n1 * m1 * p1
        U1 = np.zeros((n1 * m1, rank1))
        V1 = np.zeros((m1 * p1, rank1))
        W1 = np.zeros((p1 * n1, rank1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i * m1 + j, col] = 1
                    V1[j * p1 + k, col] = 1
                    W1[k * n1 + i, col] = 1
                    col += 1
        best_rank1 = rank1

    # Combine U1, V1, W1 with U2, V2, W2
    rank1 = best_rank1
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    # Full dimensions
    dim_U = n * m  # 9
    dim_V = m * p  # 15
    dim_W = p * n  # 15
    
    U_full = np.zeros((dim_U, total_rank))
    V_full = np.zeros((dim_V, total_rank))
    W_full = np.zeros((dim_W, total_rank))
    
    # Fill part 1 (3x3x3)
    # U1 is (9, rank1). V1 is (9, rank1). W1 is (9, rank1).
    U_full[:, :rank1] = best_U1
    V_full[:9, :rank1] = best_V1  # V1 corresponds to B1 (first 9 entries of B)
    W_full[:9, :rank1] = best_W1  # W1 corresponds to C1 (first 9 entries of C)
    
    # Fill part 2 (3x3x2)
    # U2 is (9, rank2). V2 is (6, rank2). W2 is (6, rank2).
    U_full[:, rank1:] = U2
    V_full[9:, rank1:] = V2  # V2 corresponds to B2 (last 6 entries of B)
    W_full[9:, rank1:] = W2  # W2 corresponds to C2 (last 6 entries of C)
    
    # Verify the full decomposition
    T_full = _build_target_tensor(n, m, p)
    if not _verify_decomposition(U_full, V_full, W_full, T_full, tol=1e-5):
        # Fallback to pure schoolbook if combination failed (should not happen)
        U_full = np.zeros((dim_U, n*m*p))
        V_full = np.zeros((dim_V, n*m*p))
        W_full = np.zeros((dim_W, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i*m+j, col] = 1
                    V_full[j*p+k, col] = 1
                    W_full[k*n+i, col] = 1
                    col += 1
        total_rank = n*m*p

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
