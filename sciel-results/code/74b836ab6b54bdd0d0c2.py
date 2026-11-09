import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Target: 2x3x6
    # Strategy: Partition the 3x6 matrix B into two 3x3 blocks B1 and B2.
    # C = A * B = A * B1 + A * B2.
    # This is equivalent to two independent 2x3x3 multiplications.
    # If we can find a rank-9 decomposition for 2x3x3, total rank is 18.
    # Schoolbook for 2x3x6 is 2*3*6 = 36.
    # Schoolbook for 2x3x3 is 2*3*3 = 18.
    
    # Let's try to find a rank-9 decomposition for 2x3x3.
    # Tensor T of shape (6, 9, 6) for 2x3x3.
    # Indices:
    # A: (i, j) -> a = i*3 + j (0..5)
    # B: (j, k) -> b = j*3 + k (0..8)
    # C: (i, k) -> c = i*3 + k (0..5)
    
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = i * 3 + k
                T_block[a, b, c] = 1.0

    # We want U (6xR), V (9xR), W (6xR) such that sum_r U[:,r] V[:,r] W[:,r] = T_block
    # Let's try R=9.
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Try a few random initializations with ALS
    # This might be too slow or unreliable.
    # Let's check if there's a known rank-9 decomposition.
    # Actually, the rank of 2x3x3 is known to be 9?
    # Wait, the rank of 2x2x2 is 7.
    # Rank of 2x3x3?
    # Lower bound: max(2*3, 3*3, 2*3) = 9.
    # So rank 9 is the lower bound.
    # If we can find it, it's optimal.
    
    # Let's attempt a simple search for rank 9.
    # This is a hard non-convex problem.
    # Given the time limit, a robust search might fail.
    # However, the directive says "Exploit the structural equivalence... using the known rank-9 decomposition".
    # This implies rank 9 exists.
    
    # I will implement a simple ALS solver for rank 9.
    # If it fails to converge to a small error, I will fallback to rank 18 (schoolbook for 2x3x3) or 36 (schoolbook for 2x3x6).
    
    rank_target = 9
    max_iters = 50
    num_restarts = 5
    
    found = False
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        U = np.random.randn(6, rank_target)
        V = np.random.randn(9, rank_target)
        W = np.random.randn(6, rank_target)
        
        # Normalize to prevent scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Solve for U
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V, W, this is linear in U.
            # T[:, :, c] = sum_r U[:, r] (V[:, r] * W[c, r])
            # Let M be a matrix of shape (6, 6*9) ? No.
            # Let's reshape.
            # T_flat is not quite right.
            # For each c, T[:, :, c] is a 6x9 matrix.
            # We want to express T[:, :, c] as sum_r U[:, r] * (V[:, r] * W[c, r])
            # This is a sum of rank-1 matrices in the (a,b) space?
            # No, U[:, r] is 6x1, V[:, r] is 9x1.
            # So U[:, r] * V[:, r].T is 6x9.
            # T[:, :, c] = sum_r W[c, r] * (U[:, r] V[:, r].T)
            
            # We can solve for U[:, r] by least squares.
            # Let's vectorize the (a,b) dimensions.
            # T_ab_c = vec(T[:, :, c]) is length 54.
            # The term is W[c, r] * vec(U[:, r] V[:, r].T) = W[c, r] * (U[:, r] \otimes V[:, r])
            # Let Z be a matrix of shape (54, 9) where Z[:, r] = U[:, r] \otimes V[:, r].
            # Then T_ab_c = Z * W[c, :].T ?
            # No, sum_r W[c, r] Z[:, r] = Z @ W[c, :]
            # So T_ab_c = Z @ W[c, :]
            # This is a linear system for Z?
            # But Z depends on U and V.
            
            # Alternative ALS step for U:
            # Fix V, W.
            # We want to minimize || T - sum_r U[:,r] V[:,r].T W[c,r] ||^2
            # This is equivalent to, for each c, minimizing || T[:,:,c] - sum_r W[c,r] (U[:,r] V[:,r].T) ||_F^2
            # This is a linear least squares problem for the entries of U.
            # Let's stack the equations.
            # There are 6*9*6 = 324 equations.
            # Unknowns are 6*9 = 54 entries of U.
            
            # Construct the system A_u * u_vec = t_vec
            # t_vec is T.flatten() (length 324)
            # A_u is a 324 x 54 matrix.
            
            # Index mapping:
            # t_index = c * 54 + b * 6 + a
            # u_index = r * 6 + a
            # Coefficient for U[a, r] in equation (a, b, c) is V[b, r] * W[c, r].
            
            A_u = np.zeros((324, 54))
            t_vec = T_block.flatten()
            
            for a in range(6):
                for b in range(9):
                    for c in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            u_idx = r * 6 + a
                            A_u[t_idx, u_idx] = V[b, r] * W[c, r]
                            
            # Solve A_u * u_vec = t_vec
            try:
                u_vec, _, _, _ = np.linalg.lstsq(A_u, t_vec, rcond=None)
            except:
                break
            U = u_vec.reshape(rank_target, 6).T
            
            # Similarly for V
            # Coefficient for V[b, r] is U[a, r] * W[c, r].
            A_v = np.zeros((324, 54)) # 9*9 = 81 unknowns? No, 9*9=81.
            # Wait, V is 9x9. So 81 unknowns.
            A_v = np.zeros((324, 81))
            for a in range(6):
                for b in range(9):
                    for c in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            v_idx = r * 9 + b
                            A_v[t_idx, v_idx] = U[a, r] * W[c, r]
                            
            try:
                v_vec, _, _, _ = np.linalg.lstsq(A_v, t_vec, rcond=None)
            except:
                break
            V = v_vec.reshape(rank_target, 9).T
            
            # Similarly for W
            # Coefficient for W[c, r] is U[a, r] * V[b, r].
            A_w = np.zeros((324, 54)) # 6*9 = 54 unknowns.
            for a in range(6):
                for b in range(9):
                    for c in range(6):
                        t_idx = c * 54 + b * 6 + a
                        for r in range(rank_target):
                            w_idx = r * 6 + c
                            A_w[t_idx, w_idx] = U[a, r] * V[b, r]
                            
            try:
                w_vec, _, _, _ = np.linalg.lstsq(A_w, t_vec, rcond=None)
            except:
                break
            W = w_vec.reshape(rank_target, 6).T
            
            # Check error
            # Reconstruct T
            T_rec = np.zeros((6, 9, 6))
            for r in range(rank_target):
                T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_rec - T_block)
            
            if err < 1e-5:
                found = True
                break
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        if found:
            break

    # If we found a rank-9 decomposition for the block, we can construct the rank-18 solution for 2x3x6.
    # U_total (12x18), V_total (18x18), W_total (12x18)
    # Block 1 (cols 0-8 of B): uses U, V1, W
    # Block 2 (cols 9-17 of B): uses U, V2, W
    # V1 is the first 9 columns of V (indices 0-8 of B)
    # V2 is the last 9 columns of V (indices 9-17 of B)
    # Wait, V for the block is 9x9.
    # For the full problem, V is 18x18.
    # The first 9 rows of V_full correspond to B1, the next 9 to B2.
    # So V_full[0:9, 0:9] = V_block
    #    V_full[9:18, 9:18] = V_block
    #    V_full[0:9, 9:18] = 0
    #    V_full[9:18, 0:9] = 0
    
    # U_full is 12x18.
    # U_full[0:6, 0:9] = U_block
    # U_full[0:6, 9:18] = U_block
    # U_full[6:12, :] = 0 ?
    # No, U corresponds to A (2x3=6).
    # So U_full is 6x18?
    # Wait, the interface says U has shape (n*m, rank) = (6, rank).
    # V has shape (m*p, rank) = (18, rank).
    # W has shape (p*n, rank) = (12, rank).
    
    # My block decomposition:
    # A is 2x3.
    # B1 is 3x3, B2 is 3x3.
    # C1 is 2x3, C2 is 2x3.
    # T_block is for A*B1 -> C1.
    # U_block (6x9), V_block (9x9), W_block (6x9).
    
    # Full T:
    # U_full (6x18):
    #   Cols 0-8: U_block
    #   Cols 9-17: U_block
    # V_full (18x18):
    #   Rows 0-8 (B1), Cols 0-8: V_block
    #   Rows 9-17 (B2), Cols 9-17: V_block
    #   Else 0
    # W_full (12x18):
    #   Rows 0-5 (C1), Cols 0-8: W_block
    #   Rows 6-11 (C2), Cols 9-17: W_block
    #   Else 0
    
    if found and best_err < 1e-5:
        U_full = np.zeros((6, 18))
        V_full = np.zeros((18, 18))
        W_full = np.zeros((12, 18))
        
        U_full[:, 0:9] = best_U
        U_full[:, 9:18] = best_U
        
        V_full[0:9, 0:9] = best_V
        V_full[9:18, 9:18] = best_V
        
        W_full[0:6, 0:9] = best_W
        W_full[6:12, 9:18] = best_W
        
        # Convert to lists of lists with rounded values if close to integers/half-integers
        # The problem allows integer/half-integer entries.
        # If the solution is not exactly integer/half-integer, it might be invalid?
        # "exactly decompose" usually implies the values must be such that the sum is exactly 1.
        # If the solver finds a real solution, it might not be rational.
        # However, if a rank-9 decomposition exists, it likely has rational entries.
        # Let's check if we can round to half-integers.
        
        def round_half(x):
            return np.round(x * 2) / 2
            
        U_r = round_half(U_full)
        V_r = round_half(V_full)
        W_r = round_half(W_full)
        
        # Verify the rounded solution
        T_full = np.zeros((6, 18, 12))
        for a in range(6):
            for b in range(18):
                for c in range(12):
                    T_full[a, b, c] = np.sum(U_r[a, :] * V_r[b, :] * W_r[c, :])
                    
        # Check if T_full matches the target tensor
        # Target tensor for 2x3x6
        T_target = np.zeros((6, 18, 12))
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    a = i * 3 + j
                    b = j * 6 + k
                    c = i * 6 + k
                    T_target[a, b, c] = 1.0
                    
        err_r = np.linalg.norm(T_full - T_target)
        
        if err_r < 1e-5:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            # If rounding failed, try returning the unrounded values?
            # The interface says "integer/half-integer entries".
            # So we must return rounded values.
            # If it fails, we fallback to schoolbook.
            pass

    # Fallback: Schoolbook Rank 36
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
    return {"U": U, "V": V, "W": W}
