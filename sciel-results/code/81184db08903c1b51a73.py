import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify_full(U, V, W, n=3, m=3, p=6):
    """Verify full 3x3x6 decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]): return False
    if V.shape != (m*p, V.shape[1]): return False
    if W.shape != (p*n, W.shape[1]): return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]: return False
    
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Check on-diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-5:
                    return False
                    
    # Check off-diagonal
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-5:
                        return False
    return True

def _verify_333(U, V, W):
    """Verify 3x3x3 decomposition. U,V,W are 9xr."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Target tensor entries for 3x3x3
    # a = i*3+j, b = j*3+k, c = k*3+i
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*3 + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-5:
                    return False
    # Off-diagonal
    for a in range(9):
        i_a = a // 3
        j_a = a % 3
        for b in range(9):
            j_b = b // 3
            k_b = b % 3
            for c in range(9):
                k_c = c // 3
                i_c = c % 3
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-5:
                        return False
    return True

def _find_333_decomposition(r, max_iters=500, seed=0):
    """Try to find a rank r decomposition of 3x3x3 multiplication."""
    np.random.seed(seed)
    U = np.random.randn(9, r)
    V = np.random.randn(9, r)
    W = np.random.randn(9, r)
    
    # Precompute T
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*3 + i
                T[a, b, c] = 1.0
                
    # ALS: Alternate fixing two matrices and solving for the third.
    # T[a,b,c] = sum_alpha U[a,alpha] V[b,alpha] W[c,alpha]
    # Fix U, V. Solve for W.
    # For each alpha, W[:, alpha] is a vector of length 9.
    # The term is U[:, alpha] V[:, alpha] W[:, alpha] (outer product? No, sum over alpha).
    # T[a,b,c] = sum_alpha (U[a,alpha] V[b,alpha]) W[c,alpha]
    # Let M_alpha[a,b] = U[a,alpha] V[b,alpha].
    # T[a,b,c] = sum_alpha M_alpha[a,b] W[c,alpha]
    # This is a linear system for W[:, alpha] if we fix other W's?
    # Actually, for fixed alpha, we want to minimize sum_{a,b,c} (T[a,b,c] - sum_beta M_beta[a,b] W[c,beta])^2
    # This is complex.
    
    # Simpler: Gradient Descent.
    # Error E = sum (T - sum U V W)^2
    # dE/dU[a,alpha] = -2 sum_{b,c} (T[a,b,c] - approx[a,b,c]) V[b,alpha] W[c,alpha]
    
    for it in range(max_iters):
        # Compute approx
        # approx[a,b,c] = sum_alpha U[a,alpha] V[b,alpha] W[c,alpha]
        # Use einsum for speed
        approx = np.einsum('ia,jb,kc->abc', U, V, W)
        error = np.sum((T - approx) ** 2)
        
        if error < 1e-10:
            return U, V, W
            
        # Compute gradients
        # dE/dU = -2 * einsum('abc,jb,kc->ia', T-approx, V, W)
        # dE/dV = -2 * einsum('abc,ia,kc->jb', T-approx, U, W)
        # dE/dW = -2 * einsum('abc,ia,jb->kc', T-approx, U, V)
        
        diff = T - approx
        gU = -2 * np.einsum('abc,jb,kc->ia', diff, V, W)
        gV = -2 * np.einsum('abc,ia,kc->jb', diff, U, W)
        gW = -2 * np.einsum('abc,ia,jb->kc', diff, U, V)
        
        # Simple gradient step with adaptive learning rate
        lr = 0.01
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Baseline: Schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find better 3x3x3 decomposition
    # If we find rank r for 3x3x3, we get rank 2r for 3x3x6.
    # Schoolbook 3x3x3 is rank 27. 2*27 = 54.
    # We want r < 27.
    
    for r in range(20, 27):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(50):
            if time.time() > deadline - 10:
                break
                
            res = _find_333_decomposition(r, max_iters=200, seed=seed)
            if res is not None:
                U_333, V_333, W_333 = res
                # Verify
                if _verify_333(U_333, V_333, W_333):
                    # Embed into 3x3x6
                    # U_333 is 9xr. V_333 is 9xr. W_333 is 9xr.
                    # 3x3x6: U is 9x2r. V is 18x2r. W is 18x2r.
                    # U_full = [U_333, U_333] (concatenate columns)
                    # V_full: V_333 corresponds to first 3 cols of B.
                    # B has 6 cols. Indices 0-5 (col 0), 6-11 (col 1), 12-17 (col 2) for first 3?
                    # No, B index b = j*p + k. p=6.
                    # j=0, k=0..5 -> b=0..5
                    # j=1, k=0..5 -> b=6..11
                    # j=2, k=0..5 -> b=12..17
                    # In 3x3x3, b = j*3 + k. k=0..2.
                    # So V_333[b_333, :] maps to V_full[b_333, :] for the first 3 k's?
                    # We need to map k=0,1,2 in 333 to k=0,1,2 in 336?
                    # And k=3,4,5?
                    # The 333 tensor computes C[:, 0:3] from B[:, 0:3].
                    # The second copy computes C[:, 3:6] from B[:, 3:6].
                    # So V_full for b (j, k) where k in 0..2 uses V_333(j, k).
                    # V_full for b (j, k) where k in 3..5 uses V_333(j, k-3).
                    
                    U_full = np.hstack([U_333, U_333])
                    V_full = np.zeros((18, 2*r))
                    W_full = np.zeros((18, 2*r))
                    
                    for j in range(3):
                        for k in range(3):
                            b_333 = j*3 + k
                            b_336_1 = j*6 + k
                            b_336_2 = j*6 + (k+3)
                            
                            V_full[b_336_1, :r] = V_333[b_333, :]
                            V_full[b_336_2, r:] = V_333[b_333, :]
                            
                            c_333 = k*3 + i # Wait, W index is c = k*n + i.
                            # W_333[c_333, :]
                            # W_full[c_336, :]
                            # c_336 = k*3 + i.
                            # For first copy (k 0..2), c_336 is k*3+i.
                            # For second copy (k 3..5), c_336 is k*3+i.
                            # But W_333 was computed for k=0..2.
                            # So W_full[c_336, :r] = W_333[c_333, :] for k in 0..2.
                            # W_full[c_336, r:] = W_333[c_333, :] for k in 3..5 (using k-3).
                            
                    for i in range(3):
                        for k in range(3):
                            c_333 = k*3 + i
                            c_336_1 = k*3 + i
                            c_336_2 = (k+3)*3 + i
                            
                            W_full[c_336_1, :r] = W_333[c_333, :]
                            W_full[c_336_2, r:] = W_333[c_333, :]
                            
                    # Verify full
                    if _verify_full(U_full, V_full, W_full):
                        best_result = {
                            "U": U_full.tolist(),
                            "V": V_full.tolist(),
                            "W": W_full.tolist()
                        }
                        best_rank = 2*r
                        found = True
                        break
        if found:
            break
            
    return best_result
