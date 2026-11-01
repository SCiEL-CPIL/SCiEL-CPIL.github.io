import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Target Tensor T (sparse representation)
    # T[i, j, k] = 1 if it's a valid multiplication term
    # i in 0..1, j in 0..1, k in 0..7
    # Flatten indices:
    # A: i * m + j = i * 2 + j
    # B: j * p + k = j * 8 + k
    # C: k * n + i = k * 2 + i (Trace convention W[k*n+i])
    
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term A[i,j] * B[j,k] -> C[i,k]
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target_entries.append((a_idx, b_idx, c_idx, 1.0))
                
    # Function to check if a decomposition is exact
    def check_exact(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute sum of outer products
        # This is O(R * nm * mp * pn) which is small here
        # But we only need to check the target entries
        # Actually, we must ensure ALL entries match.
        # Non-target entries must be 0.
        
        R = U.shape[1]
        # Initialize tensor
        # Size: 4, 16, 16
        # We can just compute the full tensor and check
        # Or iterate over target entries and check zeros elsewhere?
        # Full tensor is 4*16*16 = 1024 entries. Very small.
        
        T_comp = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T_comp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        # Check target entries
        for a, b, c, val in target_entries:
            if abs(T_comp[a, b, c] - val) > 1e-9:
                return False
                
        # Check non-target entries are 0
        # Create a mask of target entries
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for a, b, c, _ in target_entries:
            mask[a, b, c] = True
            
        # Sum of absolute values of non-target entries
        non_target_sum = np.sum(np.abs(T_comp[~mask]))
        if non_target_sum > 1e-9:
            return False
            
        return True

    # Function to try ALS for a given rank
    def try_als(rank, max_iters=50, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            # Solve for V and W given U
            # We want to minimize || T - sum U_r outer V_r outer W_r ||
            # This is linear in V and W? No, it's bilinear in V and W.
            # Wait, if U is fixed, the term is sum_r U_{i,r} (V_{j,r} W_{k,r})
            # This is NOT linear in V and W jointly.
            # It is linear in V if W is fixed, and linear in W if V is fixed.
            # So standard ALS alternates:
            # 1. Fix U, W. Solve for V.
            # 2. Fix V, U. Solve for W.
            # 3. Fix V, W. Solve for U.
            
            # Solve for V:
            # Residual R_{i,j,k} = T_{i,j,k} - sum_r U_{i,r} V_{j,r} W_{k,r}
            # For fixed U, W, the term involving V_{j,r} is:
            # sum_k sum_i T_{i,j,k} ... no.
            # The objective is sum_{i,j,k} (T_{i,j,k} - sum_r U_{i,r} V_{j,r} W_{k,r})^2
            # Derivative wrt V_{j,r}:
            # -2 sum_{i,k} (T_{i,j,k} - ...) U_{i,r} W_{k,r} = 0
            # sum_{i,k} T_{i,j,k} U_{i,r} W_{k,r} = sum_{i,k} (sum_{r'} U_{i,r'} V_{j,r'} W_{k,r'}) U_{i,r} W_{k,r}
            # Let A_{r, r'} = sum_{i,k} U_{i,r} U_{i,r'} W_{k,r} W_{k,r'}? No.
            # The RHS is sum_{r'} V_{j,r'} (sum_{i,k} U_{i,r'} U_{i,r} W_{k,r'} W_{k,r})
            # Let M_{r, r'} = sum_{i,k} U_{i,r} U_{i,r'} W_{k,r} W_{k,r'}
            # Then M V_j = b_j, where b_{j,r} = sum_{i,k} T_{i,j,k} U_{i,r} W_{k,r}
            
            # Construct M (rank x rank)
            M = np.zeros((rank, rank))
            # M_{r, r'} = sum_{i,k} U_{i,r} U_{i,r'} W_{k,r} W_{k,r'}
            # This can be computed as (U.T @ U) elementwise? No.
            # M = (U.T @ U) * (W.T @ W)? No, indices don't match.
            # M_{r, r'} = sum_{i} U_{i,r} U_{i,r'} * sum_{k} W_{k,r} W_{k,r'}? No.
            # The sum is over i and k together.
            # M_{r, r'} = sum_{i,k} (U_{i,r} W_{k,r}) (U_{i,r'} W_{k,r'})
            # Let Z_{i,k,r} = U_{i,r} W_{k,r}.
            # M_{r, r'} = sum_{i,k} Z_{i,k,r} Z_{i,k,r'}
            # This is the Gram matrix of the vectors Z_r flattened.
            
            Z = np.zeros((n*m, p*n, rank))
            for i in range(n*m):
                for k in range(p*n):
                    for r in range(rank):
                        Z[i, k, r] = U[i, r] * W[k, r]
                        
            M = np.zeros((rank, rank))
            for r in range(rank):
                for rp in range(rank):
                    M[r, rp] = np.sum(Z[:, :, r] * Z[:, :, rp])
                    
            # Construct b (mp x rank)
            # b_{j, r} = sum_{i,k} T_{i,j,k} U_{i,r} W_{k,r}
            b = np.zeros((m*p, rank))
            for j in range(m*p):
                # T_{i,j,k} is non-zero only for specific i,k
                # We can just loop over target entries
                # Or precompute
                pass
                
            # Efficiently compute b:
            # b_{j, r} = sum_{(i,k) in Target(j)} U_{i,r} W_{k,r}
            # Target(j) are the (i,k) pairs such that T_{i,j,k}=1.
            # For a given j (which is j_B = j_row * p + k_col? No, j is index in B)
            # B index j corresponds to row j//p, col j%p.
            # In our tensor, B index is `b_idx` in target_entries.
            # Let's just loop over target entries to fill b.
            b = np.zeros((m*p, rank))
            for a, b_idx, c_idx, val in target_entries:
                # T[a, b_idx, c_idx] = val
                # We need to add val * U[a, r] * W[c_idx, r] to b[b_idx, r]
                # Wait, the derivative equation was:
                # sum_{i,k} T_{i,j,k} U_{i,r} W_{k,r}
                # Here i is A index, k is C index?
                # In T_{i,j,k}, i is A, j is B, k is C.
                # So for a fixed B index j, we sum over A index i and C index k.
                # So b[j, r] += val * U[i, r] * W[k, r]
                for r in range(rank):
                    b[b_idx, r] += val * U[a, r] * W[c_idx, r]
                    
            # Solve M V_j = b_j for each j
            V_new = np.zeros((m*p, rank))
            try:
                # M might be singular. Add small regularization
                M_reg = M + 1e-6 * np.eye(rank)
                for j in range(m*p):
                    V_new[j, :] = np.linalg.solve(M_reg, b[j, :])
            except np.linalg.LinAlgError:
                return None
                
            # Solve for W similarly
            # M'_{r, r'} = sum_{i,j} U_{i,r} U_{i,r'} V_{j,r} V_{j,r'}
            # b'_{k, r} = sum_{i,j} T_{i,j,k} U_{i,r} V_{j,r}
            
            Z2 = np.zeros((n*m, m*p, rank))
            for i in range(n*m):
                for j in range(m*p):
                    for r in range(rank):
                        Z2[i, j, r] = U[i, r] * V[j, r]
                        
            M2 = np.zeros((rank, rank))
            for r in range(rank):
                for rp in range(rank):
                    M2[r, rp] = np.sum(Z2[:, :, r] * Z2[:, :, rp])
                    
            b2 = np.zeros((p*n, rank))
            for a, b_idx, c_idx, val in target_entries:
                for r in range(rank):
                    b2[c_idx, r] += val * U[a, r] * V[b_idx, r]
                    
            W_new = np.zeros((p*n, rank))
            try:
                M2_reg = M2 + 1e-6 * np.eye(rank)
                for k in range(p*n):
                    W_new[k, :] = np.linalg.solve(M2_reg, b2[k, :])
            except np.linalg.LinAlgError:
                return None
                
            # Solve for U
            # M''_{r, r'} = sum_{j,k} V_{j,r} V_{j,r'} W_{k,r} W_{k,r'}
            # b''_{i, r} = sum_{j,k} T_{i,j,k} V_{j,r} W_{k,r}
            
            Z3 = np.zeros((m*p, p*n, rank))
            for j in range(m*p):
                for k in range(p*n):
                    for r in range(rank):
                        Z3[j, k, r] = V[j, r] * W[k, r]
                        
            M3 = np.zeros((rank, rank))
            for r in range(rank):
                for rp in range(rank):
                    M3[r, rp] = np.sum(Z3[:, :, r] * Z3[:, :, rp])
                    
            b3 = np.zeros((n*m, rank))
            for a, b_idx, c_idx, val in target_entries:
                for r in range(rank):
                    b3[a, r] += val * V[b_idx, r] * W[c_idx, r]
                    
            U_new = np.zeros((n*m, rank))
            try:
                M3_reg = M3 + 1e-6 * np.eye(rank)
                for i in range(n*m):
                    U_new[i, :] = np.linalg.solve(M3_reg, b3[i, :])
            except np.linalg.LinAlgError:
                return None
                
            U, V, W = U_new, V_new, W_new
            
            # Check residual
            # Compute T_comp
            T_comp = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_comp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            # Check target
            err = 0
            for a, b_idx, c_idx, val in target_entries:
                err += (T_comp[a, b_idx, c_idx] - val)**2
            if err < 1e-8:
                # Check non-target
                mask = np.zeros((n*m, m*p, p*n), dtype=bool)
                for a, b_idx, c_idx, _ in target_entries:
                    mask[a, b_idx, c_idx] = True
                non_target_err = np.sum(np.abs(T_comp[~mask]))**2
                if non_target_err < 1e-8:
                    # Try to round
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if check_exact(U_round, V_round, W_round):
                        return U_round, V_round, W_round
                    # If rounding fails, maybe return float? 
                    # Task says "integer/half-integer entries".
                    # So we must return exact.
                    return None
                    
        return None

    # Try rank 27
    start_time = time.time()
    result = try_als(27, max_iters=20, seed=42)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Strassen
    # ... (existing Strassen code) ...
    
    # Re-include Strassen code here to avoid duplication in edit
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
