import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook rank 24
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Try to find rank 22 decomposition using two rank 11 blocks for 2x3 by 3x2
    # First, find rank 11 for 2x3 by 3x2
    n_sub, m_sub, p_sub = 2, 3, 2
    rank_sub = 11
    
    # Target tensor for 2x3 by 3x2
    # Indices: (i,j) for A, (j',k) for B, (k',i') for C
    # M[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k' and i==i'
    # Shape: (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)
    M_target = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                idx_u = i * m_sub + j
                idx_v = j * p_sub + k
                idx_w = k * n_sub + i
                M_target[idx_u, idx_v, idx_w] = 1.0
                
    # Search for rank 11 decomposition
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    best_err = np.inf
    
    # Random restarts
    np.random.seed(42)
    max_restarts = 50
    max_iters = 1000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        U_r = np.random.randn(n_sub * m_sub, rank_sub)
        V_r = np.random.randn(m_sub * p_sub, rank_sub)
        W_r = np.random.randn(p_sub * n_sub, rank_sub)
        
        # Simple gradient descent / projected gradient?
        # Or use scipy.optimize.minimize
        # Objective: || sum_r U_r[:, r, None] * V_r[None, :, r, None] * W_r[None, None, :, r] - M_target ||^2
        # This is high dimensional.
        
        # Alternative: Use the fact that coefficients are likely integers.
        # Try to find integer solutions?
        # For now, let's just do a few steps of gradient descent.
        
        def loss_and_grad(U, V, W):
            # Compute tensor
            T = np.zeros_like(M_target)
            for rr in range(rank_sub):
                T += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
            diff = T - M_target
            loss = np.sum(diff ** 2)
            # Gradient is complex, skip for now and use random search with local refinement?
            # Or just check if a known solution exists.
            return loss
            
        # Since I don't have the exact gradient, I'll rely on the fact that 
        # for small matrices, random integer matrices might work?
        # Actually, let's try to hardcode the known rank 11 solution if I can recall it.
        # If not, I will just return schoolbook.
        
        # Let's try a specific known structure for 2x3 by 3x2.
        # It is known that R(2,3,2) = 11.
        # I will attempt to construct it using a solver if time permits, 
        # but for safety, I will return schoolbook if I can't find it quickly.
        
        # Given the risk of failing to find the rank 11 decomposition,
        # and the time limit, I will stick to the schoolbook decomposition 
        # unless I can implement a fast solver.
        
        # However, 22 is a significant improvement.
        # Let's try one more thing: 
        # If I can't find rank 11, can I find rank 12?
        # R(2,3,2) <= 12 (schoolbook).
        # If I use two rank 12 blocks, total rank 24. No improvement.
        
        # So I must find rank 11.
        
        # I will use a simple iterative method:
        # Fix U, V, solve for W?
        # This is a linear system for W if U, V are fixed.
        # Flatten M_target to vector m.
        # Construct matrix A where A[:, idx] = vec(U_r[:, None, None] * V_r[None, :, None])?
        # No, the term is sum_r U_r[i,j] V_r[j,k] W_r[k,i].
        # This is linear in W for fixed U, V.
        
        U_curr = np.random.randn(n_sub * m_sub, rank_sub)
        V_curr = np.random.randn(m_sub * p_sub, rank_sub)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Solve for W
            # Equation: sum_r U_curr[:, r] V_curr[:, r] W_curr[:, r] = M_target
            # This is a linear system in W_curr.
            # Let's build the matrix.
            # Number of equations: (n_sub*m_sub) * (m_sub*p_sub) * (p_sub*n_sub) = 6 * 6 * 4 = 144.
            # Number of variables: rank_sub * (p_sub*n_sub) = 11 * 4 = 44.
            # Overdetermined. Least squares.
            
            A_mat = np.zeros((144, 44))
            b_vec = M_target.flatten()
            
            for i in range(n_sub * m_sub):
                for j in range(m_sub * p_sub):
                    for k in range(p_sub * n_sub):
                        eq_idx = i * (m_sub * p_sub) + j * (p_sub * n_sub) + k
                        for r in range(rank_sub):
                            # Coefficient of W_curr[k, r] is U_curr[i, r] * V_curr[j, r]
                            A_mat[eq_idx, r * (p_sub * n_sub) + k] = U_curr[i, r] * V_curr[j, r]
                            
            # Solve least squares
            W_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_curr = W_sol.reshape(rank_sub, p_sub * n_sub).T
            
            # Update U or V?
            # Let's alternate.
            # Solve for U given V, W
            A_mat_U = np.zeros((144, n_sub * m_sub * rank_sub))
            for i in range(n_sub * m_sub):
                for j in range(m_sub * p_sub):
                    for k in range(p_sub * n_sub):
                        eq_idx = i * (m_sub * p_sub) + j * (p_sub * n_sub) + k
                        for r in range(rank_sub):
                            A_mat_U[eq_idx, r * (n_sub * m_sub) + i] = V_curr[j, r] * W_curr[k, r]
                            
            U_sol, _, _, _ = np.linalg.lstsq(A_mat_U, b_vec, rcond=None)
            U_curr = U_sol.reshape(rank_sub, n_sub * m_sub).T
            
            # Check error
            T_curr = np.zeros_like(M_target)
            for rr in range(rank_sub):
                T_curr += np.einsum('a,b,c->abc', U_curr[:, rr], V_curr[:, rr], W_curr[:, rr])
            err = np.linalg.norm(T_curr - M_target)
            
            if err < best_err:
                best_err = err
                best_U_sub = U_curr.copy()
                best_V_sub = V_curr.copy()
                best_W_sub = W_curr.copy()
                
            if err < 1e-6:
                break
                
            # Random perturbation to escape local minima
            if _ % 10 == 0:
                U_curr += np.random.randn(*U_curr.shape) * 0.1
                V_curr += np.random.randn(*V_curr.shape) * 0.1
                W_curr += np.random.randn(*W_curr.shape) * 0.1
                
    if best_err < 1e-6:
        # Construct rank 22 for 2x3 by 3x4
        rank_total = 22
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        # Block 1: cols 0,1
        U[:, 0:11] = best_U_sub
        V[:, 0:11] = best_V_sub
        W[:, 0:11] = best_W_sub
        
        # Block 2: cols 2,3
        U[:, 11:22] = best_U_sub
        # Map V indices: local k=0 -> global k=2, local k=1 -> global k=3
        V_block2 = np.zeros_like(best_V_sub)
        for j in range(m_sub):
            for k_local in range(p_sub):
                k_global = 2 + k_local
                V_block2[j * p_sub + k_local, :] = best_V_sub[j * p_sub + k_local, :]
                # Wait, V is indexed by (j, k).
                # In best_V_sub, index is j * 2 + k_local.
                # In global V, index is j * 4 + k_global.
                V[j * p + k_global, 11:22] = best_V_sub[j * p_sub + k_local, :]
                
        # Map W indices: local k=0 -> global k=2, local k=1 -> global k=3
        # W is indexed by (k, i).
        for k_local in range(p_sub):
            k_global = 2 + k_local
            for i in range(n_sub):
                W[k_global * n_sub + i, 11:22] = best_W_sub[k_local * n_sub + i, :]
                
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
