import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 115.0

    # Try to find a lower rank decomposition using Greedy + ALS
    target_ranks = [25, 28, 30, 32, 35, 40]
    
    best_result = None
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        result = search_rank_greedy(n, m, p, target_rank, deadline)
        if result is not None:
            return result

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank_greedy(n, m, p, rank, deadline):
    """
    Greedy construction of rank-1 components followed by ALS refinement.
    """
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    Residual = T.copy()
    
    # Greedy Phase
    for r in range(rank):
        if time.time() > deadline - 20:
            break
            
        # 1. Initialize V, W randomly or from previous step?
        # For the first step, use random. For subsequent, keep previous V, W and update?
        # Standard greedy: Fix V, W, find best U. Then Fix U, W, find best V. Then Fix U, V, find best W.
        # Repeat this cycle a few times for each r?
        
        # Initialize V, W for this rank step if not present
        if r == 0:
            V[:, r] = np.random.randn(m * p)
            W[:, r] = np.random.randn(p * n)
        # Else, V[:, r] and W[:, r] are currently 0. We need to initialize them.
        # Better: Initialize all V, W randomly at start?
        # No, greedy builds it up.
        # Let's initialize the new column r with random values.
        else:
            V[:, r] = np.random.randn(m * p)
            W[:, r] = np.random.randn(p * n)
            
        # Optimize U given V, W
        # U[i*m+j, r] = argmin sum_k (Residual[i,j,k] - U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])^2
        # Note: Only the r-th column of V and W is non-zero in this step?
        # No, the residual is reduced by previous steps.
        # The term is sum_r' U[i,j, r'] V[j,k, r'] W[k,i, r'].
        # For the new column r, we assume U[:, r] is the only unknown in this column?
        # Yes, we are adding a rank-1 component.
        # So we want to find u_r, v_r, w_r such that u_r v_r w_r approximates Residual.
        
        # Let's do 3 iterations of alternating optimization for this single column.
        for _ in range(3):
            # Update U
            # For each i, j:
            # u_ij = sum_k Residual[i,j,k] v_jk w_ki / sum_k (v_jk w_ki)^2
            # Note: v_jk is V[j*p+k, r], w_ki is W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    num = 0
                    den = 0
                    for k in range(p):
                        v_val = V[j * p + k, r]
                        w_val = W[k * n + i, r]
                        num += Residual[i, j, k] * v_val * w_val
                        den += (v_val * w_val) ** 2
                    if den > 1e-10:
                        U[i * m + j, r] = num / den
                    else:
                        U[i * m + j, r] = 0
            
            # Update V
            # For each j, k:
            # v_jk = sum_i Residual[i,j,k] u_ij w_ki / sum_i (u_ij w_ki)^2
            for j in range(m):
                for k in range(p):
                    num = 0
                    den = 0
                    for i in range(n):
                        u_val = U[i * m + j, r]
                        w_val = W[k * n + i, r]
                        num += Residual[i, j, k] * u_val * w_val
                        den += (u_val * w_val) ** 2
                    if den > 1e-10:
                        V[j * p + k, r] = num / den
                    else:
                        V[j * p + k, r] = 0
                        
            # Update W
            # For each k, i:
            # w_ki = sum_j Residual[i,j,k] u_ij v_jk / sum_j (u_ij v_jk)^2
            for k in range(p):
                for i in range(n):
                    num = 0
                    den = 0
                    for j in range(m):
                        u_val = U[i * m + j, r]
                        v_val = V[j * p + k, r]
                        num += Residual[i, j, k] * u_val * v_val
                        den += (u_val * v_val) ** 2
                    if den > 1e-10:
                        W[k * n + i, r] = num / den
                    else:
                        W[k * n + i, r] = 0
        
        # Update Residual
        # Residual = Residual - U[:, r] outer V[:, r] outer W[:, r]
        # T[i,j,k] -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        U3 = U[:, r].reshape(n, m)
        V3 = V[:, r].reshape(m, p)
        W3 = W[:, r].reshape(p, n)
        Residual -= np.einsum('ij,jk,ki->ijk', U3, V3, W3)
        
    # After greedy phase, we have U, V, W.
    # Run ALS to refine
    U, V, W = als_solve(U, V, W, n, m, p, rank, deadline)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    error = compute_error(U_snap, V_snap, W_snap, n, m, p)
    
    if error < 1e-10:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
    return None

def als_solve(U, V, W, n, m, p, rank, deadline, max_iters=20):
    """
    Alternating Least Squares.
    """
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = V3[j] * W3[:, i]
                b = np.ones(p)
                try:
                    U3[i, j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                B = U3[:, j] * W3[k, :]
                b = np.ones(n)
                try:
                    V3[j, k] = np.linalg.lstsq(B, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                C = U3[i, :] * V3[:, k]
                b = np.ones(m)
                try:
                    W3[k, i] = np.linalg.lstsq(C, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
    U = U3.reshape(n * m, rank)
    V = V3.reshape(m * p, rank)
    W = W3.reshape(p * n, rank)
    
    return U, V, W

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    error = np.sum((T - 1.0)**2)
    return error
