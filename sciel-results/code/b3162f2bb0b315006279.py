import time
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        # Fallback to schoolbook if Strassen construction fails (should not happen)
        U_school = np.zeros((nm, 32), dtype=float)
        V_school = np.zeros((mp, 32), dtype=float)
        W_school = np.zeros((pn, 32), dtype=float)
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    idx = i * m * p + l * p + k
                    U_school[a, idx] = 1.0
                    V_school[b, idx] = 1.0
                    W_school[c, idx] = 1.0
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}

    # Try to find a lower rank decomposition by searching for linear dependencies
    # in the Strassen decomposition.
    
    # The rank-28 decomposition has 28 rank-1 terms. Let's see if we can express
    # some of these as linear combinations of others.
    
    # Actually, let's try a different approach: use the structure of the problem
    # to find a better decomposition.
    
    # For 2x4 * 4x4, we can use the following decomposition:
    # C[i,k] = sum_l A[i,l] * B[l,k]
    
    # Let's try to find a rank-27 decomposition by solving a linear system.
    # We'll assume that the rank is 27 and try to find U, V, W.
    
    # This is a hard problem, so let's try a heuristic approach.
    
    # One idea: in the Strassen decomposition, some of the 7 terms per block
    # might be combinable if the A or B entries are the same.
    
    # Let's check: in blocks 1 and 3, A_L is the same. In blocks 2 and 4, A_R is the same.
    # But the B entries are different, so the M terms are different.
    
    # However, we might be able to find a different decomposition that exploits
    # the shared A entries more efficiently.
    
    # Let's try a direct search: assume rank r and try to find U, V, W.
    # We'll use a greedy approach: pick rank-1 tensors that best approximate
    # the remaining tensor.
    
    if time.time() < deadline - 30:
        # Try to find a rank-27 decomposition
        best_rank = 28
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        # Try ranks from 27 down to 24
        for target_rank in range(27, 23, -1):
            if time.time() > deadline - 20:
                break
                
            # Try to find a rank-target_rank decomposition
            # We'll use a randomized approach: initialize U, V, W randomly,
            # then use alternating least squares to minimize the error.
            
            np.random.seed(42)
            U_rand = np.random.randn(nm, target_rank) * 0.1
            V_rand = np.random.randn(mp, target_rank) * 0.1
            W_rand = np.random.randn(pn, target_rank) * 0.1
            
            # Use ALS to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
            for iteration in range(100):
                if time.time() > deadline - 15:
                    break
                    
                # Update U
                for a in range(nm):
                    # T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r]
                    # For fixed a, this is a linear system in U[a,:]
                    # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                    # Let M[b,c,r] = V[b,r] * W[c,r]
                    # Then T[a] = U[a,:] @ M (reshaped)
                    M = V_rand @ W_rand.T  # shape (mp, pn)
                    # T[a] is (mp, pn), U[a] is (target_rank,)
                    # We need: T[a] = sum_r U[a,r] * M[:, :, r]
                    # This is: T[a].flatten() = (M.reshape(mp*pn, target_rank)).T @ U[a]
                    M_flat = M.reshape(mp * pn, target_rank)
                    # But M is (mp, pn), and we need M[b,c,r] = V[b,r]*W[c,r]
                    # So M_flat has shape (mp*pn, target_rank)
                    # T[a].flatten() has shape (mp*pn,)
                    # We want to solve: T[a].flatten() = M_flat.T @ U[a]
                    # This is an underdetermined system, so use least squares
                    U_rand[a, :] = np.linalg.lstsq(M_flat.T, T[a, :, :].flatten(), rcond=None)[0]
                
                # Update V
                for b in range(mp):
                    M = U_rand @ W_rand.T  # shape (nm, pn)
                    # T[:,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                    # Let N[a,c,r] = U[a,r] * W[c,r]
                    # Then T[:,b,:] = V[b,:] @ N (reshaped)
                    N = U_rand @ W_rand.T  # shape (nm, pn)
                    N_flat = N.reshape(nm * pn, target_rank)
                    V_rand[b, :] = np.linalg.lstsq(N_flat.T, T[:, b, :].flatten(), rcond=None)[0]
                
                # Update W
                for c in range(pn):
                    M = U_rand @ V_rand.T  # shape (nm, mp)
                    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                    # Let P[a,b,r] = U[a,r] * V[b,r]
                    # Then T[:,:,c] = W[c,:] @ P (reshaped)
                    P = U_rand @ V_rand.T  # shape (nm, mp)
                    P_flat = P.reshape(nm * mp, target_rank)
                    W_rand[c, :] = np.linalg.lstsq(P_flat.T, T[:, :, c].flatten(), rcond=None)[0]
                
                # Check convergence
                err = verify_decomp(U_rand, V_rand, W_rand, target_rank)
                if err < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    break
            
            if best_rank < 28:
                break
        
        if best_rank < 28:
            return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}
