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
    deadline = start_time + 110.0

    # Try to find a lower rank decomposition
    # Known lower bounds and potential ranks for 3x3x5
    # Schoolbook 45.
    # If Rank(3,3,3)=23 and Rank(3,3,2)=14, sum=37.
    # Try ranks from 37 upwards.
    target_ranks = [37, 38, 39, 40, 41, 42, 43, 44]
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for each rank
        max_restarts = 5
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            try:
                result = search_rank_cd(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_cd(n, m, p, rank, deadline):
    """
    Uses Cyclic Coordinate Descent to minimize the Frobenius norm of the tensor decomposition error.
    """
    # Random initialization with small values
    # Using half-integers or small integers helps snapping later
    U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
    V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
    W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5
    
    # Reshape for easier indexing
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # W3: (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    max_sweeps = 5000
    sweep = 0
    
    while sweep < max_sweeps:
        if time.time() > deadline - 5:
            break
            
        # --- Update U ---
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    # Compute optimal U[i, j, r]
                    # Error term: sum_k (U[i,j,r]*V[j,k,r]*W[k,i,r] + rest - 1)^2
                    # Let a_k = V[j,k,r]*W[k,i,r]
                    # Let b_k = sum_{r'!=r} U[i,j,r']*V[j,k,r']*W[k,i,r'] - 1
                    # We need sum(a_k^2) and sum(a_k*b_k)
                    
                    sum_a2 = 0.0
                    sum_ab = 0.0
                    
                    # Precompute current T for this i,j slice? 
                    # It's faster to just compute the "rest" part directly.
                    # rest_k = sum_{r'!=r} U3[i,j,r'] * V3[j,k,r'] * W3[k,i,r']
                    
                    for k in range(p):
                        a_k = V3[j, k, r] * W3[k, i, r]
                        sum_a2 += a_k * a_k
                        
                        # Compute rest
                        rest = 0.0
                        for rp in range(rank):
                            if rp == r:
                                continue
                            rest += U3[i, j, rp] * V3[j, k, rp] * W3[k, i, rp]
                        
                        b_k = rest - 1.0
                        sum_ab += a_k * b_k
                        
                    if sum_a2 > 1e-12:
                        new_val = -sum_ab / sum_a2
                        U3[i, j, r] = new_val
                    # else leave as is (or 0)
                        
        # --- Update V ---
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    sum_a2 = 0.0
                    sum_ab = 0.0
                    for i in range(n):
                        a_i = U3[i, j, r] * W3[k, i, r]
                        sum_a2 += a_i * a_i
                        
                        rest = 0.0
                        for rp in range(rank):
                            if rp == r:
                                continue
                            rest += U3[i, j, rp] * V3[j, k, rp] * W3[k, i, rp]
                        
                        b_i = rest - 1.0
                        sum_ab += a_i * b_i
                        
                    if sum_a2 > 1e-12:
                        new_val = -sum_ab / sum_a2
                        V3[j, k, r] = new_val
                        
        # --- Update W ---
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    sum_a2 = 0.0
                    sum_ab = 0.0
                    for j in range(m):
                        a_j = U3[i, j, r] * V3[j, k, r]
                        sum_a2 += a_j * a_j
                        
                        rest = 0.0
                        for rp in range(rank):
                            if rp == r:
                                continue
                            rest += U3[i, j, rp] * V3[j, k, rp] * W3[k, i, rp]
                        
                        b_j = rest - 1.0
                        sum_ab += a_j * b_j
                        
                    if sum_a2 > 1e-12:
                        new_val = -sum_ab / sum_a2
                        W3[k, i, r] = new_val
                        
        # Check convergence / error every few sweeps
        if sweep % 10 == 0:
            error = compute_error(U, V, W, n, m, p)
            if error < 1e-8:
                # Try to snap
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
                if snap_error < 1e-8:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
        sweep += 1
        
    # Final check
    error = compute_error(U, V, W, n, m, p)
    if error < 1e-8:
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        if snap_error < 1e-8:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    return None

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error
