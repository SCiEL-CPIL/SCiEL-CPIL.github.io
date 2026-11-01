"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 baseline with rank 27 search."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate_fast(U_np, V_np, W_np, rank):
        """Fast validation using numpy vectorization."""
        # Check diagonal entries: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i*m+j]
                    v_val = V_np[j*p+k]
                    w_val = W_np[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries should be 0
        # Only check entries where the tensor should be 0
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b_idx in range(m*p):
                j2 = b_idx // p
                k1 = b_idx % p
                # Skip if this is a diagonal entry (i1==i2, j1==j2, k1==k2)
                # We'll check all and skip diagonals
                for c_idx in range(p*n):
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    u_val = U_np[a]
                    v_val = V_np[b_idx]
                    w_val = W_np[c_idx]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s) > 1e-9:
                        return False
        return True

    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        return validate_fast(U_np, V_np, W_np, rank)

    # Build baseline rank-28
    best_result = build_strassen_28()
    best_rank = 28
    
    target_rank = 27
    found = False
    
    U28 = np.array(best_result["U"], dtype=np.float64)
    V28 = np.array(best_result["V"], dtype=np.float64)
    W28 = np.array(best_result["W"], dtype=np.float64)
    
    # Precompute the target tensor T as a 3D array
    # T[i,j,k] = 1 for all i,j,k (standard matrix multiplication)
    # For validation, we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
    
    # Strategy: Try each possible column removal from the 28-column Strassen decomposition
    # and use alternating minimization to find a rank-27 solution
    
    for remove_col in range(28):
        if time.time() > deadline:
            break
        
        keep_cols = [c for c in range(28) if c != remove_col]
        
        U = U28[:, keep_cols].copy()
        V = V28[:, keep_cols].copy()
        W = W28[:, keep_cols].copy()
        
        # Try multiple perturbations for each removed column
        for pert_scale in [0.0, 0.1, 0.3, 0.5]:
            if time.time() > deadline:
                break
            
            if pert_scale > 0:
                np.random.seed(777 + remove_col * 1000 + int(pert_scale * 10))
                U += np.random.randn(n*m, target_rank) * pert_scale
                V += np.random.randn(m*p, target_rank) * pert_scale
                W += np.random.randn(p*n, target_rank) * pert_scale
            
            # Alternating minimization with more steps
            for step in range(30):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve least squares
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    M = np.zeros((m*p*p*n, target_rank))
                    rhs = np.zeros(m*p*p*n)
                    for b in range(m*p):
                        for c_idx in range(p*n):
                            M[b*p*n + c_idx, :] = V[b, :] * W[c_idx, :]
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i == i2 and j == j2 and k1 == k2:
                                rhs[b*p*n + c_idx] = 1.0
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    M = np.zeros((n*m*p*n, target_rank))
                    rhs = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        for c_idx in range(p*n):
                            M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                            i1 = a // m
                            j1 = a % m
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == (b // p) and k1 == k2:
                                rhs[a*p*n + c_idx] = 1.0
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                # Update W
                for c_idx in range(p*n):
                    M = np.zeros((n*m*m*p, target_rank))
                    rhs = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        for b in range(m*p):
                            M[a*m*p + b, :] = U[a, :] * V[b, :]
                            i1 = a // m
                            j1 = a % m
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*m*p + b] = 1.0
                    try:
                        W[c_idx, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            # Final polish: do a few more iterations with tighter tolerance
            for polish_step in range(10):
                if time.time() > deadline:
                    break
                
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    M = np.zeros((m*p*p*n, target_rank))
                    rhs = np.zeros(m*p*p*n)
                    for b in range(m*p):
                        for c_idx in range(p*n):
                            M[b*p*n + c_idx, :] = V[b, :] * W[c_idx, :]
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i == i2 and j == j2 and k1 == k2:
                                rhs[b*p*n + c_idx] = 1.0
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=1e-12)
                    except:
                        pass
                
                for b in range(m*p):
                    M = np.zeros((n*m*p*n, target_rank))
                    rhs = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        for c_idx in range(p*n):
                            M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                            i1 = a // m
                            j1 = a % m
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == (b // p) and k1 == k2:
                                rhs[a*p*n + c_idx] = 1.0
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=1e-12)
                    except:
                        pass
                
                for c_idx in range(p*n):
                    M = np.zeros((n*m*m*p, target_rank))
                    rhs = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        for b in range(m*p):
                            M[a*m*p + b, :] = U[a, :] * V[b, :]
                            i1 = a // m
                            j1 = a % m
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*m*p + b] = 1.0
                    try:
                        W[c_idx, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=1e-12)
                    except:
                        pass
            
            # Check if we found a valid solution
            if validate_fast(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            break
    
    # If not found via alternating minimization, try a more aggressive random search
    if not found and time.time() < deadline - 10:
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + attempt * 7919)
            
            # Initialize from Strassen-28 with random column removed and larger perturbation
            remove_col = np.random.randint(0, 28)
            keep_cols = [c for c in range(28) if c != remove_col]
            
            U = U28[:, keep_cols].copy()
            V = V28[:, keep_cols].copy()
            W = W28[:, keep_cols].copy()
            
            U += np.random.randn(n*m, target_rank) * 0.8
            V += np.random.randn(m*p, target_rank) * 0.8
            W += np.random.randn(p*n, target_rank) * 0.8
            
            # Quick alternating minimization
            for step in range(15):
                if time.time() > deadline:
                    break
                
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    M = np.zeros((m*p*p*n, target_rank))
                    rhs = np.zeros(m*p*p*n)
                    for b in range(m*p):
                        for c_idx in range(p*n):
                            M[b*p*n + c_idx, :] = V[b, :] * W[c_idx, :]
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i == i2 and j == j2 and k1 == k2:
                                rhs[b*p*n + c_idx] = 1.0
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                for b in range(m*p):
                    M = np.zeros((n*m*p*n, target_rank))
                    rhs = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        for c_idx in range(p*n):
                            M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                            i1 = a // m
                            j1 = a % m
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == (b // p) and k1 == k2:
                                rhs[a*p*n + c_idx] = 1.0
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
                
                for c_idx in range(p*n):
                    M = np.zeros((n*m*m*p, target_rank))
                    rhs = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        for b in range(m*p):
                            M[a*m*p + b, :] = U[a, :] * V[b, :]
                            i1 = a // m
                            j1 = a % m
                            j2 = b // p
                            k1 = b % p
                            i2 = c_idx // n
                            k2 = c_idx % n
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*m*p + b] = 1.0
                    try:
                        W[c_idx, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        pass
            
            if validate_fast(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result

# EVOLVE-BLOCK-END
