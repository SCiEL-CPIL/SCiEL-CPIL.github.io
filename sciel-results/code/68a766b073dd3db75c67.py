import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 48

    # Build the target tensor T for 2x4x6
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Split m=4 into two m'=2 blocks.
    # C = A_1 B_1 + A_2 B_2
    # We find a rank R decomposition for the 2x2x6 tensor T_sub.
    # Then the full rank is 2*R.
    
    n_sub = 2
    m_sub = 2
    p_sub = 6
    
    T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.0

    def compute_tensor(U_mat, V_mat, W_mat):
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)

    def objective_and_grads(U_mat, V_mat, W_mat, T_target):
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T_target
        obj = np.sum(diff**2)
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return obj, 2*dU, 2*dV, 2*dW

    def try_rank_sub(rank, max_restarts=20):
        """Try to find a rank decomposition for the 2x2x6 sub-problem"""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None, None, None, False
            
            # Initialize
            U_mat = np.random.randn(n_sub*m_sub, rank) * 0.5
            V_mat = np.random.randn(m_sub*p_sub, rank) * 0.5
            W_mat = np.random.randn(p_sub*n_sub, rank) * 0.5
            
            lr = 0.05
            obj, _, _, _ = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
            best_obj = obj
            stall_count = 0
            max_iter = 2000
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None, None, None, False
                
                obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
                
                # Adaptive step
                grad_norm = np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW))
                step_scale = lr / (1 + grad_norm)
                
                U_new = U_mat - step_scale * dU
                V_new = V_mat - step_scale * dV
                W_new = W_mat - step_scale * dW
                
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_mat, T_sub) # Note: reusing W_mat in check might be slightly off, let's use W_new
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_new, T_sub)
                
                if obj_new < best_obj - 1e-12:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj = obj_new
                    stall_count = 0
                    lr *= 1.02
                else:
                    stall_count += 1
                    lr *= 0.8
                    if lr < 1e-8 or stall_count > 50:
                        break
                
                if best_obj < 1e-6:
                    break
            
            if best_obj < 1e-4:
                # Polish
                lr = 1e-3
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat, T_sub)
                    grad_norm = np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW))
                    step_scale = lr / (1 + grad_norm)
                    U_mat -= step_scale * dU
                    V_mat -= step_scale * dV
                    W_mat -= step_scale * dW
                    lr *= 0.99
                    if obj < 1e-8:
                        break
                
                final_obj = objective_and_grads(U_mat, V_mat, W_mat, T_sub)[0]
                if final_obj < 1e-4:
                    return U_mat, V_mat, W_mat, True
        
        return None, None, None, False

    # Try ranks for sub-problem
    # Schoolbook for 2x2x6 is 24.
    # Lower bound is max(4, 12, 12) = 12.
    # Try 12, 13, 14, 15, 16...
    
    best_sub_rank = 24
    best_U_sub = np.zeros((n_sub*m_sub, 24))
    best_V_sub = np.zeros((m_sub*p_sub, 24))
    best_W_sub = np.zeros((p_sub*n_sub, 24))
    
    # Initialize with schoolbook for sub
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                best_U_sub[i*m_sub+j, col] = 1
                best_V_sub[j*p_sub+k, col] = 1
                best_W_sub[k*n_sub+i, col] = 1
                col += 1

    np.random.seed(42)
    
    for rank in range(12, best_sub_rank + 1):
        if time.time() > deadline:
            break
            
        # Increase restarts for lower ranks as they are harder
        restarts = 30 if rank < 15 else 10
        U_s, V_s, W_s, found = try_rank_sub(rank, max_restarts=restarts)
        
        if found:
            best_sub_rank = rank
            best_U_sub = U_s.copy()
            best_V_sub = V_s.copy()
            best_W_sub = W_s.copy()
            # Don't break immediately, maybe we can find lower? 
            # But we iterate from low to high, so first found is best.
            break

    # Construct full solution from sub-solution
    # Full U: shape (8, 2*best_sub_rank)
    # Full V: shape (24, 2*best_sub_rank)
    # Full W: shape (12, 2*best_sub_rank)
    
    full_rank = 2 * best_sub_rank
    U_out = np.zeros((n*m, full_rank))
    V_out = np.zeros((m*p, full_rank))
    W_out = np.zeros((p*n, full_rank))
    
    # Block 0: i in 0..1, j in 0..1, k in 0..5
    # Indices in full U: i*m + j. For m=4, j=0,1 -> rows 0,1,4,5?
    # Wait. A is 2x4. A[i, j].
    # Block 1 uses A[i, 0:2]. Block 2 uses A[i, 2:4].
    # In U (size 8 x R), row index is i*m + j.
    # For Block 1 (j=0,1): rows are 0,1,4,5? No.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 4. i=1, j=1 -> 5.
    # So rows 0,1,4,5 of U_full correspond to U_sub rows 0,1,2,3.
    # U_sub row index is i*m_sub + j = i*2 + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 2. i=1, j=1 -> 3.
    
    # Mapping U_sub (4 rows) to U_full (8 rows) for Block 1:
    # U_sub[0] (i=0,j=0) -> U_full[0]
    # U_sub[1] (i=0,j=1) -> U_full[1]
    # U_sub[2] (i=1,j=0) -> U_full[4]
    # U_sub[3] (i=1,j=1) -> U_full[5]
    
    # For Block 2 (j=2,3):
    # U_sub[0] (i=0,j=0) -> U_full[2] (i=0, j=2)
    # U_sub[1] (i=0,j=1) -> U_full[3] (i=0, j=3)
    # U_sub[2] (i=1,j=0) -> U_full[6] (i=1, j=2)
    # U_sub[3] (i=1,j=1) -> U_full[7] (i=1, j=3)
    
    # V mapping:
    # V is (m*p, R) = (24, R).
    # V_sub is (12, R).
    # V index is j*p + k.
    # Block 1 (j=0,1):
    # j=0, k=0..5 -> V_full[0..5]
    # j=1, k=0..5 -> V_full[6..11]
    # V_sub index is j*p_sub + k.
    # j=0, k=0..5 -> V_sub[0..5]
    # j=1, k=0..5 -> V_sub[6..11]
    # So V_full[0:12] = V_sub for Block 1.
    
    # Block 2 (j=2,3):
    # j=2, k=0..5 -> V_full[12..17]
    # j=3, k=0..5 -> V_full[18..23]
    # V_sub index is j*p_sub + k.
    # j=0, k=0..5 -> V_sub[0..5]
    # j=1, k=0..5 -> V_sub[6..11]
    # So V_full[12:24] = V_sub for Block 2.
    
    # W mapping:
    # W is (p*n, R) = (12, R).
    # W index is k*n + i.
    # W_sub is (12, R).
    # W index is k*n_sub + i.
    # Since n=2 and n_sub=2, the indices are identical!
    # k=0..5, i=0..1.
    # W_full[c] = W_sub[c] for both blocks?
    # Wait. The decomposition is C = A1 B1 + A2 B2.
    # The W factor represents the output C.
    # The tensor is T = T1 + T2.
    # T1 uses W factors for C. T2 uses W factors for C.
    # So yes, W_full is just W_sub repeated?
    # No, the rank is 2*R_sub.
    # The first R_sub columns of W_full correspond to Block 1.
    # The next R_sub columns of W_full correspond to Block 2.
    # Since the output C is the same, the W factors for Block 1 and Block 2 are identical in terms of which output element they affect.
    # So W_full[:, 0:R_sub] = W_sub
    #    W_full[:, R_sub:2*R_sub] = W_sub
    
    # Let's fill U
    for r in range(best_sub_rank):
        # Block 1
        U_out[0, r] = best_U_sub[0, r]
        U_out[1, r] = best_U_sub[1, r]
        U_out[4, r] = best_U_sub[2, r]
        U_out[5, r] = best_U_sub[3, r]
        
        # Block 2
        U_out[2, r + best_sub_rank] = best_U_sub[0, r]
        U_out[3, r + best_sub_rank] = best_U_sub[1, r]
        U_out[6, r + best_sub_rank] = best_U_sub[2, r]
        U_out[7, r + best_sub_rank] = best_U_sub[3, r]
        
        # Fill V
        V_out[0:12, r] = best_V_sub[:, r]
        V_out[12:24, r + best_sub_rank] = best_V_sub[:, r]
        
        # Fill W
        W_out[:, r] = best_W_sub[:, r]
        W_out[:, r + best_sub_rank] = best_W_sub[:, r]

    # Convert to list of lists
    U_list = [[0.0] * full_rank for _ in range(n * m)]
    V_list = [[0.0] * full_rank for _ in range(m * p)]
    W_list = [[0.0] * full_rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(full_rank):
            U_list[a][r] = float(U_out[a, r])
    for b in range(m * p):
        for r in range(full_rank):
            V_list[b][r] = float(V_out[b, r])
    for c in range(p * n):
        for r in range(full_rank):
            W_list[c][r] = float(W_out[c, r])

    return {"U": U_list, "V": V_list, "W": W_list}
