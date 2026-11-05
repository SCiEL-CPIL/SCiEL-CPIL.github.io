import time
import sys
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 118.0  # seconds (safety margin for 120s limit)

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 5
    U_rows = n * m  # 16
    V_rows = m * p  # 20
    W_rows = p * n  # 20
    
    # Build the matrix multiplication tensor T[i,j,k] = e_ij * e_jk -> e_ik
    # T is a 3-tensor: T[i,j,k] = 1 if (i,j) pairs with (j,k) to give (i,k)
    # Shape: (n*m, m*p, p*n)
    # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # Non-zero when indices are consistent: same j in a and b, same i and k in b and c
    
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a lower rank decomposition using randomized tensor decomposition
    # Start with schoolbook rank 80 as fallback
    best_U = None
    best_V = None
    best_W = None
    best_rank = 80
    
    # Schoolbook fallback
    U_fb = np.zeros((U_rows, 80))
    V_fb = np.zeros((V_rows, 80))
    W_fb = np.zeros((W_rows, 80))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_fb[a, col] = 1
                V_fb[b, col] = 1
                W_fb[c, col] = 1
                col += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try various ranks from 49 up to 65
    for target_rank in range(49, 66):
        if _time_remaining() < 5.0:
            break
        
        rng = np.random.RandomState(target_rank * 137 + 42)
        
        # Multiple restarts
        for restart in range(30):
            if _time_remaining() < 2.0:
                break
            
            # Random initialization
            U_rand = rng.randn(U_rows, target_rank)
            V_rand = rng.randn(V_rows, target_rank)
            W_rand = rng.randn(W_rows, target_rank)
            
            # Normalize columns
            for r in range(target_rank):
                nu = np.linalg.norm(U_rand[:, r])
                nv = np.linalg.norm(V_rand[:, r])
                nw = np.linalg.norm(W_rand[:, r])
                if nu > 0 and nv > 0 and nw > 0:
                    U_rand[:, r] /= (nu ** (1/3))
                    V_rand[:, r] /= (nv ** (1/3))
                    W_rand[:, r] /= (nw ** (1/3))
            
            # Alternating least squares / tensor power iteration
            U_cur = U_rand.copy()
            V_cur = V_rand.copy()
            W_cur = W_rand.copy()
            
            for iteration in range(500):
                if _time_remaining() < 1.5:
                    break
                
                # Update U: for each column r, minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||
                # Fix V and W, solve for U
                # Residual: R = T - sum_r u_r ⊗ v_r ⊗ w_r
                # For each r: minimize ||T - sum_s u_s ⊗ v_s ⊗ w_s|| with respect to u_r
                # This is a linear problem: T contracted with v_r and w_r gives the target
                # target_r = T * v_r * w_r (tensor contraction)
                # Then u_r = target_r / (||v_r||^2 * ||w_r||^2) approximately
                
                # More precisely, using the tensor structure:
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Contract with V[:,r] and W[:,r]:
                # target_a = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # Then U[:,r] = target / (sum_b V[b,r]^2 * sum_c W[c,r]^2) ... no
                # Actually U[:,r] = target_a / (||V[:,r]||_2^2 * ||W[:,r]||_2^2) is wrong
                # The correct update: U[:,r] = (T * V[:,r] * W[:,r]) / (||V[:,r]||^2 * ||W[:,r]||^2)
                
                # Let me use a different approach: ALS on the matricized tensor
                # T_(1) is U_rows x (V_rows*W_rows) matrix
                # T_(1)[a, (b,c)] = T[a,b,c]
                # T_(1) = U * (V ⊗ W)^T where (V⊗W) has rows [V[:,r] ⊗ W[:,r]]
                
                # Build V⊗W matrix
                # Shape: target_rank x (V_rows * W_rows)
                VW = np.zeros((target_rank, V_rows * W_rows))
                for r in range(target_rank):
                    VW[r, :] = (V_cur[:, r] * W_cur[:, r]).flatten()
                
                # T_(1) shape: U_rows x (V_rows * W_rows)
                T_mat = T.reshape(U_rows, V_rows * W_rows)
                
                # Solve: T_mat ≈ U_cur @ VW^T
                # U_cur = T_mat @ (VW^T)^+ (pseudo-inverse)
                # Using least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW.T, T_mat.T, rcond=None)
                    U_new = U_new.T
                except:
                    break
                
                # Similarly for V and W
                # T_(2): V_rows x (U_rows * W_rows)
                T_mat2 = np.zeros((V_rows, U_rows * W_rows))
                for a in range(U_rows):
                    for c in range(W_rows):
                        T_mat2[:, a * W_rows + c] = T[a, :, c]
                
                UW = np.zeros((target_rank, U_rows * W_rows))
                for r in range(target_rank):
                    UW[r, :] = (U_new[:, r] * W_cur[:, r]).flatten()
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_mat2.T, rcond=None)
                    V_new = V_new.T
                except:
                    break
                
                # T_(3): W_rows x (U_rows * V_rows)
                T_mat3 = np.zeros((W_rows, U_rows * V_rows))
                for a in range(U_rows):
                    for b in range(V_rows):
                        T_mat3[:, a * V_rows + b] = T[a, b, :]
                
                UV = np.zeros((target_rank, U_rows * V_rows))
                for r in range(target_rank):
                    UV[r, :] = (U_new[:, r] * V_new[:, r]).flatten()
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_mat3.T, rcond=None)
                    W_new = W_new.T
                except:
                    break
                
                # Check convergence
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
            
            # Check if decomposition is exact (or very close)
            # Reconstruct T from U, V, W
            T_recon = np.zeros((U_rows, V_rows, W_rows))
            for r in range(target_rank):
                T_recon += np.outer(np.outer(U_cur[:, r], V_cur[:, r]), W_cur[:, r])
            
            error = np.linalg.norm(T_recon - T)
            if error < 1e-8:
                # Verify with integer/half-integer check - we need to round
                # Try rounding to nearest integer
                U_int = np.round(U_cur)
                V_int = np.round(V_cur)
                W_int = np.round(W_cur)
                
                T_recon_int = np.zeros((U_rows, V_rows, W_rows))
                for r in range(target_rank):
                    T_recon_int += np.outer(np.outer(U_int[:, r], V_int[:, r]), W_int[:, r])
                
                error_int = np.linalg.norm(T_recon_int - T)
                if error_int < 1e-8:
                    best_U, best_V, best_W = U_int, V_int, W_int
                    best_rank = target_rank
                    break  # Found exact at this rank, no need to try higher
                else:
                    # Try half-integers
                    U_half = np.round(2 * U_cur) / 2.0
                    V_half = np.round(2 * V_cur) / 2.0
                    W_half = np.round(2 * W_cur) / 2.0
                    
                    T_recon_half = np.zeros((U_rows, V_rows, W_rows))
                    for r in range(target_rank):
                        T_recon_half += np.outer(np.outer(U_half[:, r], V_half[:, r]), W_half[:, r])
                    
                    error_half = np.linalg.norm(T_recon_half - T)
                    if error_half < 1e-8:
                        # Convert to integer representation (multiply by 2)
                        best_U, best_V, best_W = np.round(2 * U_half), np.round(2 * V_half), np.round(2 * W_half)
                        # But we need to use the half-integer values directly
                        # The evaluator accepts half-integers, so store as-is
                        best_U, best_V, best_W = U_half, V_half, W_half
                        best_rank = target_rank
                        break
        
        if best_rank == target_rank:
            break
    
    # If we found a better decomposition, use it
    if best_rank < 80 and best_U is not None:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        # Convert to proper format
        U_list = [[float(x) for x in row] for row in U_out]
        V_list = [[float(x) for x in row] for row in V_out]
        W_list = [[float(x) for x in row] for row in W_out]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to the known rank-65 decomposition
    # Use the double-Strassen + schoolbook approach from the original
    rank = 65
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    # Strassen for 4x4x4 part (49 terms)
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        if super_idx == 0:
            p_blocks = [1, 0, 0, 1]
            q_blocks = [1, 0, 0, 1]
        elif super_idx == 1:
            p_blocks = [0, 0, 1, 1]
            q_blocks = [1, 0, 0, 0]
        elif super_idx == 2:
            p_blocks = [1, 0, 0, 0]
            q_blocks = [0, 1, 0, -1]
        elif super_idx == 3:
            p_blocks = [0, 0, 0, 1]
            q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4:
            p_blocks = [1, 1, 0, 0]
            q_blocks = [0, 0, 0, 1]
        elif super_idx == 5:
            p_blocks = [-1, 0, 1, 0]
            q_blocks = [1, 1, 0, 0]
        elif super_idx == 6:
            p_blocks = [0, 1, 0, -1]
            q_blocks = [0, 0, 1, 1]
        else:
            return u_list, v_list, w_list
        
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0:
                    continue
                if b_idx == 0:
                    gi, gj = r, c
                elif b_idx == 1:
                    gi, gj = r, 2 + c
                elif b_idx == 2:
                    gi, gj = 2 + r, c
                elif b_idx == 3:
                    gi, gj = 2 + r, 2 + c
                res.append((gi, gj, coeff))
            return res
        
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0:
                    continue
                if b_idx == 0:
                    gj, gk = r, c
                elif b_idx == 1:
                    gj, gk = r, 2 + c
                elif b_idx == 2:
                    gj, gk = 2 + r, c
                elif b_idx == 3:
                    gj, gk = 2 + r, 2 + c
                res.append((gj, gk, coeff))
            return res
        
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            c_contrib = [(0, 0, 1), (1, 1, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            c_contrib = [(1, 0, 1), (1, 1, -1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            c_contrib = [(0, 1, 1), (1, 1, 1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            c_contrib = [(0, 0, 1), (1, 0, 1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            c_contrib = [(0, 0, -1), (0, 1, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            c_contrib = [(1, 1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            c_contrib = [(0, 0, 1)]
        
        for p_elem, coeff in p_elems:
            for i, j, c in get_p_element(p_elem):
                u_list.append((i, j, coeff * c))
        
        for q_elem, coeff in q_elems:
            for j, k, c in get_q_element(q_elem):
                v_list.append((j, k, coeff * c))
        
        if super_idx == 0:
            blocks = [(0, 0, 1), (1, 1, 1)]
        elif super_idx == 1:
            blocks = [(1, 0, 1), (1, 1, -1)]
        elif super_idx == 2:
            blocks = [(0, 1, 1), (1, 1, 1)]
        elif super_idx == 3:
            blocks = [(0, 0, 1), (1, 0, 1)]
        elif super_idx == 4:
            blocks = [(0, 0, -1), (0, 1, 1)]
        elif super_idx == 5:
            blocks = [(1, 1, 1)]
        elif super_idx == 6:
            blocks = [(0, 0, 1)]
        
        for br, bc, sign in blocks:
            for lr, lc, local_coeff in c_contrib:
                gi = br * 2 + lr
                gk = bc * 2 + lc
                w_list.append((gk, gi, sign * local_coeff))
        
        return u_list, v_list, w_list
    
    for super_idx in range(7):
        for inner_idx in range(7):
            u_list, v_list, w_list = get_super_term_indices(super_idx, inner_idx)
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 (16 terms)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
