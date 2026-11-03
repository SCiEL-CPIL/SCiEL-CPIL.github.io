import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # Target tensor T (sparse representation)
    # T is of shape (n*m, m*p, p*n)
    # Non-zeros at (i*m+j, j*p+k, k*n+i) = 1
    # We will store the indices of the non-zeros
    indices_u = []
    indices_v = []
    indices_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_u.append(i * m + j)
                indices_v.append(j * p + k)
                indices_w.append(k * n + i)
    
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    # Helper to verify if U, V, W exactly decompose T
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute T_approx[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # We only need to check the non-zero positions of T
        # T_approx[u, v, w] should be 1
        # T_approx[other] should be 0? 
        # The problem says "exactly decompose the matrix multiplication tensor".
        # This implies T_approx == T.
        # Since T is sparse, we must ensure no spurious non-zeros.
        # However, if we use exact integer arithmetic or very precise floats, 
        # and the rank is minimal, spurious terms might cancel out or be zero.
        # But generally, a rank-R decomposition of a rank-R tensor is unique up to basis change.
        # If we find a decomposition with small error on the support, we should check the whole tensor.
        # But checking the whole tensor (60 * 30 * 20 = 36000 elements) is fast.
        
        # Efficient check:
        # T_approx = np.einsum('ur, vr, wr -> uvw', U, V, W)
        # This is (6, 30, 20) = 3600 elements. Very fast.
        
        T_approx = np.einsum('ur, vr, wr -> uvw', U, V, W)
        
        # Create target T
        T_target = np.zeros((n*m, m*p, p*n), dtype=float)
        T_target[indices_u, indices_v, indices_w] = 1.0
        
        err = np.linalg.norm(T_approx - T_target)
        return err < 1e-9

    # Helper to get the rank 55 solution
    def get_rank_55():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part 1: 5 blocks of Strassen
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            u_a00 = 0
            u_a01 = 1
            u_a10 = 3
            u_a11 = 4
            
            v_b00 = k0
            v_b01 = k1
            v_b10 = 10 + k0
            v_b11 = 10 + k1
            
            w_c00 = 2 * k0
            w_c01 = 2 * k1
            w_c10 = 2 * k0 + 1
            w_c11 = 2 * k1 + 1

            # M1
            U[u_a00, col] += 1; U[u_a11, col] += 1
            V[v_b00, col] += 1; V[v_b11, col] += 1
            W[w_c00, col] += 1; W[w_c11, col] += 1
            col += 1
            # M2
            U[u_a10, col] += 1; U[u_a11, col] += 1
            V[v_b00, col] += 1
            W[w_c10, col] += 1; W[w_c11, col] += -1
            col += 1
            # M3
            U[u_a00, col] += 1
            V[v_b01, col] += 1; V[v_b11, col] += -1
            W[w_c01, col] += 1; W[w_c11, col] += 1
            col += 1
            # M4
            U[u_a11, col] += 1
            V[v_b10, col] += 1; V[v_b00, col] += -1
            W[w_c00, col] += 1; W[w_c10, col] += 1
            col += 1
            # M5
            U[u_a00, col] += 1; U[u_a01, col] += 1
            V[v_b11, col] += 1
            W[w_c00, col] += -1; W[w_c01, col] += 1
            col += 1
            # M6
            U[u_a10, col] += 1; U[u_a00, col] += -1
            V[v_b00, col] += 1; V[v_b01, col] += 1
            W[w_c11, col] += 1
            col += 1
            # M7
            U[u_a01, col] += 1; U[u_a11, col] += -1
            V[v_b10, col] += 1; V[v_b11, col] += 1
            W[w_c00, col] += 1
            col += 1

        # Part 2: Schoolbook for A[:,2] B[2,:]
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

        return U, V, W

    # ALS Solver
    def als_solve(R, U_init=None, V_init=None, W_init=None, max_iter=200):
        if U_init is None:
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
        else:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
        # Normalize to prevent divergence
        for r in range(R):
            norm = np.linalg.norm(U[:, r])
            if norm > 0:
                U[:, r] /= norm
                V[:, r] *= norm
        
        for it in range(max_iter):
            # Solve for W
            # Minimize || T - sum U V W ||^2
            # For each w, the equation is linear in W[:, w]
            # T[:, :, w] = sum_r U[:, r] V[:, r] W[r, w]
            # This is a least squares problem for each w?
            # Actually, we can solve for all W at once using matmul tricks.
            # Let M = U * V^T? No.
            # T_{uvw} = \sum_r U_{ur} V_{vr} W_{wr}
            # Fix U, V. Solve for W.
            # \sum_r W_{wr} (U_{ur} V_{vr}) = T_{uvw}
            # Let A_{uv, r} = U_{ur} V_{vr}.
            # Then A \cdot W^T = T^T (reshaped)?
            
            # Efficient ALS step:
            # W = ( (U * V^T) \otimes I )^+ T
            # We can do:
            # Temp = np.einsum('ur, vr -> uvw', U, V, W) # No, W is unknown.
            
            # Standard ALS:
            # For each r, we update one factor? No, update one matrix at a time.
            # Update W:
            # W_new = argmin_W || T - \sum_r U_r V_r W_r ||_F
            # This is equivalent to solving a linear system for each slice of W?
            # No, W is (pn, R).
            # The objective is quadratic in W.
            # Gradient: 2 (T - \sum U_r V_r W_r) \cdot (U_r V_r)^T = 0
            # \sum_w (T_{uvw} - \sum_r U_{ur} V_{vr} W_{wr}) U_{ur'} V_{vr'} = 0
            # This couples all W_{wr}.
            
            # Easier: 
            # Reshape T to (n*m * m*p, p*n)
            # T_mat = T.reshape(n*m * m*p, p*n)
            # U_mat = U.reshape(n*m * m*p, R) ? No.
            # The term is \sum_r U_{ur} V_{vr} W_{wr}
            # This is not a simple matrix product unless we vectorize carefully.
            
            # Let's use the property:
            # T_{uvw} = \langle U_v, V_u, W_w \rangle ? No.
            
            # We can solve for W by iterating over w?
            # For fixed w, T_{uvw} is a matrix of size (nm * mp).
            # T_{uvw} = \sum_r (U_{ur} V_{vr}) W_{wr}
            # Let B_w be the matrix with entries B_w[u, v] = T_{uvw}.
            # Let C_r be the matrix with entries C_r[u, v] = U_{ur} V_{vr}.
            # Then B_w = \sum_r C_r W_{wr}.
            # This is a linear system for W_{wr}.
            # [C_0, C_1, ..., C_{R-1}] [W_{w0}, W_{w1}, ..., W_{w(R-1)}]^T = B_w
            # This system is (nm * mp) x R.
            # nm * mp = 6 * 30 = 180.
            # R is ~50.
            # So we solve a 180x50 least squares problem for each w (20 times).
            # This is fast.
            
            # Precompute C_r
            C_list = []
            for r in range(R):
                C_r = np.outer(U[:, r], V[:, r])
                C_list.append(C_r.flatten())
            C_mat = np.column_stack(C_list) # Shape (180, R)
            
            W_new = np.zeros((p * n, R))
            for w in range(p * n):
                B_w = T_target[:, :, w].flatten() # Shape (180,)
                # Solve C_mat @ W_new[w, :] = B_w
                sol, _, _, _ = np.linalg.lstsq(C_mat, B_w, rcond=None)
                W_new[w, :] = sol
            
            W = W_new
            
            # Update V
            # Symmetric.
            # T_{uvw} = \sum_r U_{ur} V_{vr} W_{wr}
            # For fixed v, T_{uvw} is a matrix of size (nm * pn).
            # T_{uvw} = \sum_r (U_{ur} W_{wr}) V_{vr}
            # Let D_r be matrix with entries D_r[u, w] = U_{ur} W_{wr}.
            # D_mat = column_stack(D_r.flatten()) # Shape (nm * pn, R)
            # nm * pn = 6 * 20 = 120.
            
            D_list = []
            for r in range(R):
                D_r = np.outer(U[:, r], W[:, r])
                D_list.append(D_r.flatten())
            D_mat = np.column_stack(D_list)
            
            V_new = np.zeros((m * p, R))
            for v in range(m * p):
                B_v = T_target[:, v, :].flatten() # Shape (120,)
                sol, _, _, _ = np.linalg.lstsq(D_mat, B_v, rcond=None)
                V_new[v, :] = sol
            V = V_new
            
            # Update U
            # For fixed u, T_{uvw} is a matrix of size (mp * pn).
            # T_{uvw} = \sum_r (V_{vr} W_{wr}) U_{ur}
            # Let E_r be matrix with entries E_r[v, w] = V_{vr} W_{wr}.
            # E_mat = column_stack(E_r.flatten()) # Shape (mp * pn, R)
            # mp * pn = 30 * 20 = 600.
            
            E_list = []
            for r in range(R):
                E_r = np.outer(V[:, r], W[:, r])
                E_list.append(E_r.flatten())
            E_mat = np.column_stack(E_list)
            
            U_new = np.zeros((n * m, R))
            for u in range(n * m):
                B_u = T_target[u, :, :].flatten() # Shape (600,)
                sol, _, _, _ = np.linalg.lstsq(E_mat, B_u, rcond=None)
                U_new[u, :] = sol
            U = U_new
            
            # Check convergence / exactness
            if verify(U, V, W):
                return U, V, W, True
                
        return U, V, W, False

    # Try ranks from 54 down to 50
    # Use the rank 55 solution as initialization for rank 54
    U55, V55, W55 = get_rank_55()
    
    for R in range(54, 49, -1):
        # Initialize by taking first R columns of rank 55 solution
        U_init = U55[:, :R]
        V_init = V55[:, :R]
        W_init = W55[:, :R]
        
        # Add small noise to escape local minima if it's a bad subset
        U_init = U_init + np.random.randn(*U_init.shape) * 1e-3
        V_init = V_init + np.random.randn(*V_init.shape) * 1e-3
        W_init = W_init + np.random.randn(*W_init.shape) * 1e-3
        
        U_sol, V_sol, W_sol, success = als_solve(R, U_init, V_init, W_sol, max_iter=100)
        
        if success:
            # Round to nearest half-integer?
            # The problem allows integer/half-integer.
            # If the solution is close to such values, we should snap.
            # But if it's exact in float, we can just return.
            # The evaluator checks "exactly decompose".
            # If we return float, it might have 1e-15 error.
            # We should snap to rational numbers if possible.
            
            # Snap to nearest 0.5
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            if verify(U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # If snap fails, try returning the float solution if error is tiny
            if verify(U_sol, V_sol, W_sol):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                
        # If time is running out, break
        if time.time() > deadline - 10:
            break
            
    # Fallback to Rank 55
    return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}

