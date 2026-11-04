"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Trivial solution (Schoolbook)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()

    # 2. Block Partitioning Strategy (1x1x1 + 2x4x5)
    # Total Rank = 1 + Rank(2x4x5)
    # Known Rank(2x4x5) = 17
    # Total Rank = 18
    
    # Indices:
    # A: 3x5 (i, j)
    # B: 5x6 (j, k)
    # C: 3x6 (i, k)
    
    # Block 1: i=0, j=0, k=0 (1x1x1)
    # U1[0, 0] = 1, V1[0, 0] = 1, W1[0, 0] = 1
    
    # Block 2: i in {1, 2}, j in {1, 2, 3, 4}, k in {1, 2, 3, 4, 5}
    # This is a 2x4x5 multiplication.
    # We need a rank-17 decomposition for 2x4x5.
    
    # Strassen-like or known low-rank decomp for 2x4x5.
    # Let's construct a specific rank-17 decomposition for 2x4x5.
    # Or use a generic solver if we can find one quickly, but 17 is small.
    # Actually, for 2x4x5, the rank is known to be 17.
    # Let's try to find or construct one.
    
    # Alternative: Use the HOOI solver from the original code but target rank 18 directly?
    # The original code tried ranks from 89 down. It likely failed to find 18 because
    # random initialization + HOOI is hard for such low rank, or the snap-to-half-integer
    # failed.
    
    # Let's try to explicitly construct the 2x4x5 part using a known algorithm or search.
    # Since we have time, we can use the HOOI solver but with a better initialization 
    # or by fixing the 1x1x1 part and solving for the rest.
    
    # Strategy:
    # 1. Fix the 1x1x1 component (index 0 in the rank dimension).
    # 2. Use HOOI to find the remaining 17 components for the 2x4x5 sub-tensor.
    # 3. The sub-tensor is T_sub[i, j, k] = 1 if i in {1,2}, j in {1,2,3,4}, k in {1,2,3,4,5}
    #    Note: The global indices are different.
    #    Global U index: i*m + j
    #    Global V index: j*p + k
    #    Global W index: k*n + i
    
    # Let's define the sub-problem in local indices for 2x4x5.
    # n_sub = 2, m_sub = 4, p_sub = 5.
    # Local U shape: (2*4, 17) = (8, 17)
    # Local V shape: (4*5, 17) = (20, 17)
    # Local W shape: (5*2, 17) = (10, 17)
    
    # We will run a search for this sub-problem.
    
    def solve_subproblem(n_sub, m_sub, p_sub, target_rank, max_time):
        """Solves matrix multiplication tensor decomposition for given dimensions."""
        start = time.time()
        deadline = start + max_time
        
        # Target tensor
        T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0
        
        # Unfolds
        T1 = T_sub.reshape(n_sub * m_sub, -1)
        T2 = np.transpose(T_sub, (1, 0, 2)).reshape(m_sub * p_sub, -1)
        T3 = np.transpose(T_sub, (2, 0, 1)).reshape(p_sub * n_sub, -1)
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        # Try to find a solution of target_rank
        R = target_rank
        
        # Multiple random restarts
        num_restarts = 200 # Adjust based on time
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n_sub * m_sub, R)
            V = np.random.randn(m_sub * p_sub, R)
            W = np.random.randn(p_sub * n_sub, R)
            
            # HOOI iterations
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                B_U = np.zeros((m_sub * p_sub * p_sub * n_sub, R))
                for col in range(R):
                    B_U[:, col] = np.outer(V[:, col], W[:, col]).flatten()
                try:
                    BtB = B_U.T @ B_U
                    BtT = B_U.T @ T1.T
                    U_new = np.linalg.solve(BtB, BtT).T
                except:
                    continue
                U = U_new
                
                # Update V
                B_V = np.zeros((n_sub * m_sub * p_sub * n_sub, R))
                for col in range(R):
                    B_V[:, col] = np.outer(U[:, col], W[:, col]).flatten()
                try:
                    BtB = B_V.T @ B_V
                    BtT = B_V.T @ T2.T
                    V_new = np.linalg.solve(BtB, BtT).T
                except:
                    continue
                V = V_new
                
                # Update W
                B_W = np.zeros((n_sub * m_sub * m_sub * p_sub, R))
                for col in range(R):
                    B_W[:, col] = np.outer(U[:, col], V[:, col]).flatten()
                try:
                    BtB = B_W.T @ B_W
                    BtT = B_W.T @ T3.T
                    W_new = np.linalg.solve(BtB, BtT).T
                except:
                    continue
                W = W_new
            
            # Check error
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.sum((T_approx - T_sub) ** 2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-9:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    T_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    error_snap = np.sum((T_approx_snap - T_sub) ** 2)
                    if error_snap < 1e-9:
                        return U_snap, V_snap, W_snap
                    else:
                        # Keep the unsnapped one if it's very close, but we need exact.
                        # If snap fails, we might need to accept a non-half-integer?
                        # No, contract says integer/half-integer.
                        # If HOOI converges to exact, snap should work.
                        # If not, continue searching.
                        pass
        
        return best_U, best_V, best_W, best_error

    # Time budget for sub-problem
    time_left = 140 - (time.time() - time.time()) # Just use a fixed budget
    budget_sub = 120
    
    U_sub, V_sub, W_sub, err_sub = solve_subproblem(2, 4, 5, 17, budget_sub)
    
    # If we found a solution for the sub-problem, construct the full solution
    if U_sub is not None and err_sub < 1e-9:
        # Full Rank = 1 + 17 = 18
        R_full = 18
        U_full = np.zeros((n * m, R_full))
        V_full = np.zeros((m * p, R_full))
        W_full = np.zeros((p * n, R_full))
        
        # Block 1: 1x1x1 (i=0, j=0, k=0)
        # Global indices:
        # U index: 0*5 + 0 = 0
        # V index: 0*6 + 0 = 0
        # W index: 0*3 + 0 = 0
        U_full[0, 0] = 1
        V_full[0, 0] = 1
        W_full[0, 0] = 1
        
        # Block 2: 2x4x5 (i in {1,2}, j in {1,2,3,4}, k in {1,2,3,4,5})
        # Local to Global mapping:
        # Local i' in {0,1} -> Global i = i' + 1
        # Local j' in {0,1,2,3} -> Global j = j' + 1
        # Local k' in {0,1,2,3,4} -> Global k = k' + 1
        
        # U_sub shape: (8, 17) -> Local (i', j')
        # V_sub shape: (20, 17) -> Local (j', k')
        # W_sub shape: (10, 17) -> Local (k', i')
        
        for r in range(17):
            # U_sub
            for i_prime in range(2):
                for j_prime in range(4):
                    local_u_idx = i_prime * 4 + j_prime
                    global_i = i_prime + 1
                    global_j = j_prime + 1
                    global_u_idx = global_i * 5 + global_j
                    U_full[global_u_idx, r + 1] = U_sub[local_u_idx, r]
                    
            # V_sub
            for j_prime in range(4):
                for k_prime in range(5):
                    local_v_idx = j_prime * 5 + k_prime
                    global_j = j_prime + 1
                    global_k = k_prime + 1
                    global_v_idx = global_j * 6 + global_k
                    V_full[global_v_idx, r + 1] = V_sub[local_v_idx, r]
                    
            # W_sub
            for k_prime in range(5):
                for i_prime in range(2):
                    local_w_idx = k_prime * 2 + i_prime
                    global_k = k_prime + 1
                    global_i = i_prime + 1
                    global_w_idx = global_k * 3 + global_i
                    W_full[global_w_idx, r + 1] = W_sub[local_w_idx, r]
        
        # Verify full solution
        T_full_approx = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        T_full_target = np.zeros((n * m, m * p, p * n))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    T_full_target[a * m + b, b * p + c, c * n + a] = 1.0
                    
        error_full = np.sum((T_full_approx - T_full_target) ** 2)
        
        if error_full < 1e-9:
            # Snap to half-integers
            U_snap = np.round(U_full * 2) / 2
            V_snap = np.round(V_full * 2) / 2
            W_snap = np.round(W_full * 2) / 2
            
            T_full_approx_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            error_full_snap = np.sum((T_full_approx_snap - T_full_target) ** 2)
            
            if error_full_snap < 1e-9:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }

    # Fallback to trivial
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
# EVOLVE-BLOCK-END
