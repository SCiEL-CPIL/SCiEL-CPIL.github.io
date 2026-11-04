import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Precompute the flattened multiplication tensor M
    # M indices: (i, j, jp, kp, kp_out, ip_out)
    # Target: delta_{i,ip_out} delta_{j,jp} delta_{kp,kp_out}
    
    # M_mat_W: rows (i, j, jp, kp), cols (kp_out, ip_out)
    # Size: (n*m*m*p) x (p*n)
    M_mat_W = np.zeros((n*m*m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    idx_row = i*m*m*p + j*m*p + jp*p + kp
                    idx_col = kp*n + i
                    M_mat_W[idx_row, idx_col] = 1.0
    M_vec_W = M_mat_W.flatten()

    # M_mat_V: rows (i, j, kp, ip), cols (jp, k)
    # Wait, the variable being solved is V.
    # Equation: sum_r U[i,j,r] W[kp,ip,r] V[jp,k,r] = M[i,j,jp,kp,kp,ip]?
    # Let's re-verify the index mapping for V.
    # T[i, j, jp, kp, kp_out, ip_out] = sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r]
    # We want T = delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
    
    # For V:
    # Fixed: i, j, kp_out, ip_out.
    # Vars: jp, kp (indices of V).
    # Equation for each (i, j, kp_out, ip_out, jp, kp):
    # sum_r U[i,j,r] W[kp_out,ip_out,r] V[jp,kp,r] = delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
    # This is a linear system in V.
    # Rows of A_V: (i, j, kp_out, ip_out, jp, kp) ? No, the LHS depends on V[jp,kp,r].
    # So for a fixed set of (i, j, kp_out, ip_out), we have a system for V[jp,kp,:].
    # Actually, it's a big linear system for all V entries.
    # Vectorize V: V_vec of size (m*p * R).
    # The coefficient for V[jp, kp, r] is U[i,j,r] * W[kp_out,ip_out,r].
    # The RHS is M[i,j,jp,kp,kp_out,ip_out].
    
    # Let's construct M_mat_V properly.
    # Rows correspond to (i, j, jp, kp, kp_out, ip_out) ?
    # No, the unknowns are V[jp, kp, r].
    # So the matrix should have columns corresponding to (jp, kp, r).
    # And rows corresponding to the equations (i, j, kp_out, ip_out).
    # But the equation is:
    # sum_{jp, kp, r} (U[i,j,r] W[kp_out,ip_out,r]) V[jp,kp,r] = M[i,j,jp,kp,kp_out,ip_out]
    # Wait, the sum is over jp, kp, r?
    # No, the tensor equation is:
    # T[i, j, jp, kp, kp_out, ip_out] = sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r]
    # This must equal delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out} for ALL i, j, jp, kp, kp_out, ip_out.
    
    # So for V, we fix i, j, kp_out, ip_out.
    # The equation is:
    # sum_r U[i,j,r] W[kp_out,ip_out,r] V[jp,kp,r] = delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
    # This must hold for ALL jp, kp.
    # So we have (m*p) equations for the vector V[:, :, r] (flattened).
    # Wait, V has R components.
    # So for each (i, j, kp_out, ip_out), we have a system:
    # sum_{jp, kp, r} C_{jp,kp,r} V_{jp,kp,r} = RHS_{jp,kp}
    # Where C_{jp,kp,r} = U[i,j,r] W[kp_out,ip_out,r].
    # This is a linear system for V_vec (size m*p*R).
    # The matrix A_V has rows indexed by (i, j, kp_out, ip_out, jp, kp).
    # And columns by (jp_col, kp_col, r_col).
    # A_V[row, col] = 1 if (jp, kp) in row matches (jp_col, kp_col) in col?
    # And the coefficient is U[i,j,r] W[kp_out,ip_out,r].
    
    # Let's build M_mat_V.
    # Rows: (i, j, kp_out, ip_out, jp, kp) -> n*m*p*n*m*p
    # Cols: (jp, kp, r) -> m*p*R
    # M_mat_V is huge?
    # n=2, m=3, p=5.
    # Rows: 2*3*5*2*3*5 = 900.
    # Cols: 15*R.
    # This is small.
    
    M_mat_V = np.zeros((n*m*p*n*m*p, m*p)) # Wait, R is not fixed yet in precomputation?
    # The structure of M_mat_V depends on R?
    # No, the linear system is:
    # A_V V_vec = M_vec_V
    # A_V depends on U and W.
    # M_vec_V is the target tensor flattened.
    
    # Let's precompute M_vec_V.
    # M_vec_V index: (i, j, kp_out, ip_out, jp, kp)
    # Value: delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
    M_vec_V = np.zeros(n*m*p*n*m*p)
    for i in range(n):
        for j in range(m):
            for kp_out in range(p):
                for ip_out in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            idx = i*m*p*n*m*p + j*p*n*m*p + kp_out*n*m*p + ip_out*m*p + jp*p + kp
                            if j == jp and kp == kp_out and i == ip_out:
                                M_vec_V[idx] = 1.0

    # Similarly for U.
    # Equation:
    # sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r] = delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
    # Fix jp, kp, kp_out, ip_out.
    # sum_{i,j,r} V[jp,kp,r] W[kp_out,ip_out,r] U[i,j,r] = RHS
    # Rows: (jp, kp, kp_out, ip_out, i, j)
    # Cols: (i, j, r)
    M_vec_U = np.zeros(m*p*p*n*n*m)
    for jp in range(m):
        for kp in range(p):
            for kp_out in range(p):
                for ip_out in range(n):
                    for i in range(n):
                        for j in range(m):
                            idx = jp*p*p*n*n*m + kp*p*n*n*m + kp_out*n*n*m + ip_out*n*m + i*m + j
                            if j == jp and kp == kp_out and i == ip_out:
                                M_vec_U[idx] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Efficient check using einsum
        # T[i, j, jp, kp, kp_out, ip_out]
        T = np.einsum('ijr,jpr,kqr->ijjpkq', U3, V3, W3)
        
        # Check against target
        # Target[i, j, jp, kp, kp_out, ip_out] = delta_{j,jp} delta_{kp,kp_out} delta_{i,ip_out}
        # We can just check a few random indices or all of them.
        # Since it's small, check all.
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        for kp_out in range(p):
                            for ip_out in range(n):
                                expected = 1.0 if (j == jp and kp == kp_out and i == ip_out) else 0.0
                                if abs(T[i, j, jp, kp, kp_out, ip_out] - expected) > tol:
                                    return False
        return True

    def snap_to_half_integers(mat):
        return np.round(mat * 2) / 2

    def alternating_minimization(rank, max_iter=100, seed=0):
        np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            # 1. Solve for W
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            
            # A_W rows: (i, j, jp, kp), cols: (kp_out, ip_out, r) ?
            # Equation: sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r] = M[i,j,jp,kp,kp_out,ip_out]
            # Fix i, j, jp, kp.
            # Unknowns: W[kp_out, ip_out, r].
            # Coeff: U[i,j,r] V[jp,kp,r].
            
            # Construct A_W
            # Size: (n*m*m*p) x (p*n*rank)
            A_W = np.zeros((n*m*m*p, p*n*rank))
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            row = i*m*m*p + j*m*p + jp*p + kp
                            for r in range(rank):
                                coeff = U[i,j,r] * V[jp,kp,r]
                                for kp_out in range(p):
                                    for ip_out in range(n):
                                        col = (kp_out*n + ip_out)*rank + r
                                        A_W[row, col] = coeff
                                        
            # M_vec_W is already flattened
            # But M_vec_W index is (i, j, jp, kp, kp_out, ip_out)
            # A_W row is (i, j, jp, kp).
            # We need to reshape M_vec_W to match A_W rows?
            # M_vec_W is size n*m*m*p * p*n.
            # A_W is size n*m*m*p * p*n*rank.
            # This is a block diagonal structure?
            # No, for each (i,j,jp,kp), we have p*n equations for the rank variables?
            # No, W has rank variables for each (kp_out, ip_out).
            # The equation is:
            # sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r] = M[...]
            # This is a linear system in W.
            # Let W_vec be size (p*n*rank).
            # The matrix A_W has rows corresponding to (i,j,jp,kp, kp_out, ip_out).
            # And columns to (kp_out_col, ip_out_col, r_col).
            
            # Let's rebuild A_W properly.
            A_W = np.zeros((n*m*m*p*p*n, p*n*rank))
            idx_row = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            for kp_out in range(p):
                                for ip_out in range(n):
                                    for r in range(rank):
                                        col = (kp_out*n + ip_out)*rank + r
                                        A_W[idx_row, col] = U[i,j,r] * V[jp,kp,r]
                                    idx_row += 1
                                    
            W_vec, _, _, _ = np.linalg.lstsq(A_W, M_vec_W, rcond=None)
            W = W_vec.reshape(p*n, rank)
            
            # 2. Solve for V
            # Equation: sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r] = M[...]
            # Fix i, j, kp_out, ip_out.
            # Unknowns: V[jp,kp,r].
            A_V = np.zeros((n*m*p*n*m*p, m*p*rank))
            idx_row = 0
            for i in range(n):
                for j in range(m):
                    for kp_out in range(p):
                        for ip_out in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    for r in range(rank):
                                        col = (jp*p + kp)*rank + r
                                        A_V[idx_row, col] = U[i,j,r] * W[kp_out,ip_out,r]
                                    idx_row += 1
                                        
            V_vec, _, _, _ = np.linalg.lstsq(A_V, M_vec_V, rcond=None)
            V = V_vec.reshape(m*p, rank)
            
            # 3. Solve for U
            # Equation: sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r] = M[...]
            # Fix jp, kp, kp_out, ip_out.
            # Unknowns: U[i,j,r].
            A_U = np.zeros((m*p*p*n*n*m, n*m*rank))
            idx_row = 0
            for jp in range(m):
                for kp in range(p):
                    for kp_out in range(p):
                        for ip_out in range(n):
                            for i in range(n):
                                for j in range(m):
                                    for r in range(rank):
                                        col = (i*m + j)*rank + r
                                        A_U[idx_row, col] = V[jp,kp,r] * W[kp_out,ip_out,r]
                                    idx_row += 1
                                    
            U_vec, _, _, _ = np.linalg.lstsq(A_U, M_vec_U, rcond=None)
            U = U_vec.reshape(n*m, rank)
            
            # Calculate error
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijr,jpr,kqr->ijjpkq', U3, V3, W3)
            
            # Create target tensor
            # Target[i, j, jp, kp, kp_out, ip_out]
            Target = np.zeros((n, m, m, p, p, n))
            for i in range(n):
                for j in range(m):
                    Target[i, j, j, :, :, i] = 1.0 # delta_{j,jp} is handled by setting only jp=j
                    # Wait, delta_{kp,kp_out} and delta_{i,ip_out}
                    # Target[i, j, j, kp, kp, i] = 1
                    # Let's re-do this carefully.
                    
            # Easier:
            Target = np.zeros((n, m, m, p, p, n))
            for i in range(n):
                for j in range(m):
                    for kp in range(p):
                        Target[i, j, j, kp, kp, i] = 1.0

            err = np.sum((T - Target)**2)
            
            if abs(prev_err - err) < 1e-10:
                break
            prev_err = err
            
            if err < 1e-10:
                break
        
        return U, V, W, err

    # Try ranks
    for rank in range(15, 26):
        if time.time() > deadline - 20:
            break
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            seed = int(time.time() * 1000) % 10000 + trial
            U, V, W, err = alternating_minimization(rank, max_iter=50, seed=seed)
            
            if err < 1e-8:
                # Snap to half-integers
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                W_snapped = snap_to_half_integers(W)
                
                if check_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-9):
                    best = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return best
