import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    return time.time() > deadline

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            U_row = U_arr[:, r].reshape(n, m)
            term_U = (U_row * A).sum(axis=1)
            
            V_col = V_arr[:, r].reshape(m, p)
            term_V = (V_col * B).sum(axis=0)
            
            W_mat = W_arr[:, r].reshape(p, n)
            C_computed += W_mat.T * np.outer(term_U, term_V)
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _als_decomposition(n, m, p, target_rank, max_iters=200, tol=1e-10, seed=0):
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    reg = 1e-8 * np.eye(target_rank)
    
    # Precompute strides/indices?
    # V is (m*p, R), W is (p*n, R), U is (n*m, R)
    
    for iteration in range(max_iters):
        # Update U
        # M[i, j, r, k] = V[j, k, r] * W[k, i, r]
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        # einsum: V(j,k,r) * W(k,i,r) -> M(i,j,r,k)
        M_U = np.einsum('jkr,kir->ijrk', V_reshaped, W_reshaped)
        
        for i in range(n):
            for j in range(m):
                # M_U[i, j] is (R, P)
                A_mat = M_U[i, j]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        # N[j, k, r, i] = U[i, j, r] * W[k, i, r]
        U_reshaped = U.reshape(n, m, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        # einsum: U(i,j,r) * W(k,i,r) -> N(j,k,r,i)
        M_V = np.einsum('ijr,kir->jkri', U_reshaped, W_reshaped)
        
        for j in range(m):
            for k in range(p):
                # M_V[j, k] is (R, N)
                A_mat = M_V[j, k]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        # O[k, i, r, j] = U[i, j, r] * V[j, k, r]
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        # einsum: U(i,j,r) * V(j,k,r) -> O(k,i,r,j)
        M_W = np.einsum('ijr,jkr->kirj', U_reshaped, V_reshaped)
        
        for k in range(p):
            for i in range(n):
                # M_W[k, i] is (R, M)
                A_mat = M_W[k, i]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        error = 0.0
        # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Compute tensor contraction
        # U(i,j,r) V(j,k,r) W(k,i,r) -> T(i,j,k)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        error = np.sum((1.0 - T) ** 2)
        
        if error < tol:
            return U.tolist(), V.tolist(), W.tolist(), True
    
    return U.tolist(), V.tolist(), W.tolist(), False

def _polish_decomposition(U_list, V_list, W_list, n, m, p, max_iters=50, tol=1e-14):
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    target_rank = U.shape[1]
    
    reg = 1e-12 * np.eye(target_rank)
    
    for iteration in range(max_iters):
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        M_U = np.einsum('jkr,kir->ijrk', V_reshaped, W_reshaped)
        for i in range(n):
            for j in range(m):
                A_mat = M_U[i, j]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        U_reshaped = U.reshape(n, m, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        M_V = np.einsum('ijr,kir->jkri', U_reshaped, W_reshaped)
        for j in range(m):
            for k in range(p):
                A_mat = M_V[j, k]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        M_W = np.einsum('ijr,jkr->kirj', U_reshaped, V_reshaped)
        for k in range(p):
            for i in range(n):
                A_mat = M_W[k, i]
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        error = np.sum((1.0 - T) ** 2)
        
        if error < tol:
            return U.tolist(), V.tolist(), W.tolist(), True
    
    return U.tolist(), V.tolist(), W.tolist(), False

def _round_to_half_integer(U, V, W):
    # Round to nearest 0.5
    U_r = np.round(np.array(U) * 2) / 2
    V_r = np.round(np.array(V) * 2) / 2
    W_r = np.round(np.array(W) * 2) / 2
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 115
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start from 42, go up to 59
    for target_rank in range(42, 60):
        if _check_time(deadline - 5):
            break
        
        found = False
        for seed in range(20):
            if _check_time(deadline - 2):
                break
            
            U_list, V_list, W_list, converged = _als_decomposition(
                n, m, p, target_rank, max_iters=100, tol=1e-8, seed=seed
            )
            
            if converged:
                # Polish
                U_polished, V_polished, W_polished, polished_ok = _polish_decomposition(
                    U_list, V_list, W_list, n, m, p, max_iters=20, tol=1e-12
                )
                
                # Try rounding
                U_rounded, V_rounded, W_rounded = _round_to_half_integer(U_polished, V_polished, W_polished)
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = target_rank
                    found = True
                    break
                elif _verify_decomposition(U_polished, V_polished, W_polished, n, m, p, tol=1e-6):
                    # If float works, maybe rounding didn't, but task requires int/half-int.
                    # If we can't round, we might fail the "integer/half-integer" constraint.
                    # But if the float is very close to half-int, rounding should work.
                    # If it doesn't, we skip.
                    pass
        
        if found:
            break
    
    return best
