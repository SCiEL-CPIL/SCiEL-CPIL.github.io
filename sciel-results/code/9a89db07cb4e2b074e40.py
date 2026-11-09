import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS

    def verify(U, V, W, rank):
        """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def verify_np(U_arr, V_arr, W_arr, rank):
        """Vectorized verification using numpy."""
        # U_arr: (9, rank), V_arr: (9, rank), W_arr: (9, rank)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this as: for each c, outer product of U[:,c], V[:,c], W[:,c]
        # and sum over c, then check entries.
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Let's build T as a 3x3x3 array.
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U_arr[:, c]  # 9 values: U[i*m+j, c]
            vc = V_arr[:, c]  # 9 values: V[j*p+k, c]
            wc = W_arr[:, c]  # 9 values: W[k*n+i, c]
            # Reshape to (n,m), (m,p), (p,n)
            u_mat = uc.reshape(n, m)
            v_mat = vc.reshape(m, p)
            w_mat = wc.reshape(p, n)
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i, j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += u_val * v_mat[j, k] * w_mat[k, i]
        return np.all(T == 1)

    def to_np(U, V, W):
        return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)

    # Fallback: schoolbook rank 27
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb

    # Strategy: search for rank 23, 24, 25, 26 with improved local search
    # Key insight: use numpy for fast verification and smarter initialization
    
    target_ranks = [23, 24, 25, 26]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 0
        max_restarts = 50000
        
        while not found and num_restarts < max_restarts and time.time() < deadline - 5:
            num_restarts += 1
            
            # Initialize with random sparse {-1, 0, 1} matrices
            # Use varying densities
            density = 0.2 + 0.15 * (num_restarts % 5) / 5.0
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            for r in range(n*m):
                for c in range(target_rank):
                    if random.random() < density:
                        U[r][c] = 1 if random.random() < 0.5 else -1
            for r in range(m*p):
                for c in range(target_rank):
                    if random.random() < density:
                        V[r][c] = 1 if random.random() < 0.5 else -1
            for r in range(p*n):
                for c in range(target_rank):
                    if random.random() < density:
                        W[r][c] = 1 if random.random() < 0.5 else -1
            
            # Quick check with numpy
            U_np, V_np, W_np = to_np(U, V, W)
            if verify_np(U_np, V_np, W_np, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
            
            # Local search: try to fix violations by modifying entries
            if num_restarts % 10 == 0 and time.time() < deadline - 5:
                # Compute current errors
                U_np, V_np, W_np = to_np(U, V, W)
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    uc = U_np[:, c].reshape(n, m)
                    vc = V_np[:, c].reshape(m, p)
                    wc = W_np[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            u_val = uc[i, j]
                            if u_val == 0:
                                continue
                            for k in range(p):
                                T[i, j, k] += u_val * vc[j, k] * wc[k, i]
                
                errors = T - 1
                max_err = np.max(np.abs(errors))
                
                if max_err == 0:
                    # Already verified (shouldn't happen if verify_np said False)
                    continue
                
                # Find the worst error location
                worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                wi, wj, wk = worst_idx
                
                # Try to fix by modifying entries in U[wi*m+wj], V[wj*p+wk], W[wk*n+wi]
                fixed = False
                for c in range(target_rank):
                    if fixed:
                        break
                    
                    # Try modifying U[wi*m+wj][c]
                    row_u = wi * m + wj
                    for new_val in [-1, 0, 1]:
                        if new_val == U[row_u][c]:
                            continue
                        old_val = U[row_u][c]
                        U[row_u][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        U[row_u][c] = old_val
                    
                    if fixed:
                        break
                    
                    # Try modifying V[wj*p+wk][c]
                    row_v = wj * p + wk
                    for new_val in [-1, 0, 1]:
                        if new_val == V[row_v][c]:
                            continue
                        old_val = V[row_v][c]
                        V[row_v][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        V[row_v][c] = old_val
                    
                    if fixed:
                        break
                    
                    # Try modifying W[wk*n+wi][c]
                    row_w = wk * n + wi
                    for new_val in [-1, 0, 1]:
                        if new_val == W[row_w][c]:
                            continue
                        old_val = W[row_w][c]
                        W[row_w][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        W[row_w][c] = old_val
                
                if found:
                    break
        
        if found:
            break
    
    # If we haven't found a better rank, try a more systematic approach for rank 24
    if best_rank > 24 and time.time() < deadline - 15:
        target_rank = 24
        found = False
        num_restarts = 0
        max_restarts = 30000
        
        while not found and num_restarts < max_restarts and time.time() < deadline - 5:
            num_restarts += 1
            
            # Use a structured initialization: start with schoolbook-like structure
            # and then perturb
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            # First 23 columns: random
            for c in range(23):
                for r in range(n*m):
                    if random.random() < 0.3:
                        U[r][c] = 1 if random.random() < 0.5 else -1
                for r in range(m*p):
                    if random.random() < 0.3:
                        V[r][c] = 1 if random.random() < 0.5 else -1
                for r in range(p*n):
                    if random.random() < 0.3:
                        W[r][c] = 1 if random.random() < 0.5 else -1
            
            # Last column: try to be a "correction" column
            for r in range(n*m):
                U[r][23] = 1 if random.random() < 0.5 else -1
            for r in range(m*p):
                V[r][23] = 1 if random.random() < 0.5 else -1
            for r in range(p*n):
                W[r][23] = 1 if random.random() < 0.5 else -1
            
            U_np, V_np, W_np = to_np(U, V, W)
            if verify_np(U_np, V_np, W_np, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
            
            # Local search
            if num_restarts % 5 == 0 and time.time() < deadline - 5:
                U_np, V_np, W_np = to_np(U, V, W)
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    uc = U_np[:, c].reshape(n, m)
                    vc = V_np[:, c].reshape(m, p)
                    wc = W_np[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            u_val = uc[i, j]
                            if u_val == 0:
                                continue
                            for k in range(p):
                                T[i, j, k] += u_val * vc[j, k] * wc[k, i]
                
                errors = T - 1
                max_err = np.max(np.abs(errors))
                
                if max_err == 0:
                    continue
                
                worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                wi, wj, wk = worst_idx
                
                fixed = False
                for c in range(target_rank):
                    if fixed:
                        break
                    
                    row_u = wi * m + wj
                    for new_val in [-1, 0, 1]:
                        if new_val == U[row_u][c]:
                            continue
                        old_val = U[row_u][c]
                        U[row_u][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        U[row_u][c] = old_val
                    
                    if fixed:
                        break
                    
                    row_v = wj * p + wk
                    for new_val in [-1, 0, 1]:
                        if new_val == V[row_v][c]:
                            continue
                        old_val = V[row_v][c]
                        V[row_v][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        V[row_v][c] = old_val
                    
                    if fixed:
                        break
                    
                    row_w = wk * n + wi
                    for new_val in [-1, 0, 1]:
                        if new_val == W[row_w][c]:
                            continue
                        old_val = W[row_w][c]
                        W[row_w][c] = new_val
                        U_np, V_np, W_np = to_np(U, V, W)
                        if verify_np(U_np, V_np, W_np, target_rank):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = target_rank
                            found = True
                            fixed = True
                            break
                        W[row_w][c] = old_val
                
                if found:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
