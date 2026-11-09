"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Pre-compute support: list of (i_idx, j_idx, k_idx)
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
    
    # Build support as numpy arrays for vectorized evaluation
    sup_i = np.array([s[0] for s in support])
    sup_j = np.array([s[1] for s in support])
    sup_k = np.array([s[2] for s in support])
    
    def evaluate_tensor(U_mat, V_mat, W_mat):
        """Evaluate tensor values at support points. Returns array of T_ijk values."""
        # U_mat[sup_i, :] shape: (num_support, rank)
        # V_mat[sup_j, :] shape: (num_support, rank)
        # W_mat[sup_k, :] shape: (num_support, rank)
        U_s = U_mat[sup_i, :]
        V_s = V_mat[sup_j, :]
        W_s = W_mat[sup_k, :]
        # Element-wise product then sum over rank axis
        return np.sum(U_s * V_s * W_s, axis=1)
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if decomposition is valid. Returns True if valid."""
        T = evaluate_tensor(U_mat, V_mat, W_mat)
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        # Check off-support: sample random off-support points
        # Total points: nm * mp * pn = 15 * 30 * 18 = 8100
        # Support: 90, so off-support: 8010
        for _ in range(200):
            ii = np.random.randint(0, nm)
            jj = np.random.randint(0, mp)
            kk = np.random.randint(0, pn)
            a, b = divmod(ii, m)
            b2, c = divmod(jj, p)
            c2, a2 = divmod(kk, n)
            if b == b2 and c == c2 and a == a2:
                continue  # on support, skip
            t_val = np.sum(U_mat[ii, :] * V_mat[jj, :] * W_mat[kk, :])
            if abs(t_val) > tol:
                return False
        return True
    
    def als_solve(U_mat, V_mat, W_mat, r, max_iter=100, tol=1e-10):
        """Alternating Least Squares to refine decomposition.
        Minimizes sum over support (T_ijk - 1)^2 + sum over off-support T_ijk^2
        """
        U = U_mat.copy()
        V = V_mat.copy()
        W = W_mat.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each row i, minimize sum_j,k (U[i,:] dot (V[j,:] * W[k,:]) - T_target)^2
            # For support points: T_target = 1, for off-support: T_target = 0
            # This is a least squares problem per row of U
            
            # Build the design matrix for U updates
            # For each support point (i, j, k): the contribution to U[i,:] is V[j,:] * W[k,:]
            # We need: sum over (j,k) pairs for fixed i of (U[i,:] dot (V[j,:]*W[k,:]) - 1)^2
            # Plus off-support terms
            
            # Efficient approach: for each i, collect all (j,k) pairs
            # This is complex, so use a simpler gradient-based ALS
            
            # Gradient for U[i,r] = 2 * sum_{(j,k): (i,j,k) in support} (T_ijk - 1) * V[j,r] * W[k,r]
            #                        + 2 * sum_{(j,k): (i,j,k) not in support} T_ijk * V[j,r] * W[k,r]
            
            # This is still expensive. Let's use a different approach:
            # Use the fact that we can vectorize over all support points
            
            # For U update:
            # grad_U[i, r] = 2 * sum_{j,k: (i,j,k) in support} (T_ijk - 1) * V[j,r] * W[k,r]
            #              + 2 * sum_{j,k: (i,j,k) not in support} T_ijk * V[j,r] * W[k,r]
            
            # This is very expensive to compute exactly. Instead, use a sampled approach.
            
            # Sample-based gradient
            sample_size = 50
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Support contributions
            T_supp = evaluate_tensor(U, V, W)
            for idx in range(len(support)):
                i_idx = sup_i[idx]
                j_idx = sup_j[idx]
                k_idx = sup_k[idx]
                err = T_supp[idx] - 1.0
                vw = V[j_idx, :] * W[k_idx, :]
                u_row = U[i_idx, :]
                v_row = V[j_idx, :]
                w_row = W[k_idx, :]
                grad_U[i_idx, :] += 2.0 * err * vw
                grad_V[j_idx, :] += 2.0 * err * (u_row * w_row)
                grad_W[k_idx, :] += 2.0 * err * (u_row * v_row)
            
            # Off-support contributions (sampled)
            for _ in range(sample_size):
                ii = np.random.randint(0, nm)
                jj = np.random.randint(0, mp)
                kk = np.random.randint(0, pn)
                a, b = divmod(ii, m)
                b2, c = divmod(jj, p)
                c2, a2 = divmod(kk, n)
                if b == b2 and c == c2 and a == a2:
                    continue
                t_val = np.sum(U[ii, :] * V[jj, :] * W[kk, :])
                u_row = U[ii, :]
                v_row = V[jj, :]
                w_row = W[kk, :]
                grad_U[ii, :] += 2.0 * t_val * (v_row * w_row)
                grad_V[jj, :] += 2.0 * t_val * (u_row * w_row)
                grad_W[kk, :] += 2.0 * t_val * (u_row * v_row)
            
            # Adam-like update with learning rate decay
            lr = 0.01 / (1 + iteration * 0.01)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            T_supp = evaluate_tensor(U, V, W)
            max_err = np.max(np.abs(T_supp - 1.0))
            if max_err < tol:
                break
        
        return U, V, W
    
    # Try to find lower rank decompositions
    # Start from rank_school and try to reduce
    for r in range(rank_school - 1, max(50, rank_school - 30), -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_try = np.random.randn(nm, r) * 0.5
            V_try = np.random.randn(mp, r) * 0.5
            W_try = np.random.randn(pn, r) * 0.5
            
            # Run ALS
            U_ref, V_ref, W_ref = als_solve(U_try, V_try, W_try, r, max_iter=50)
            
            # Check if valid
            if check_decomposition(U_ref, V_ref, W_ref, tol=1e-7):
                # Polish with tighter tolerance
                U_pol, V_pol, W_pol = als_solve(U_ref, V_ref, W_ref, r, max_iter=100, tol=1e-12)
                if check_decomposition(U_pol, V_pol, W_pol, tol=1e-9):
                    best_U, best_V, best_W = U_pol, V_pol, W_pol
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
