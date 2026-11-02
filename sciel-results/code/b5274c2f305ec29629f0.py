import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition()
    
    def try_rank_reduction(U, V, W, rank):
        """Try to reduce rank by zeroing out columns that don't contribute."""
        if rank <= 1:
            return U, V, W, rank
        
        # Check if any column is all zeros or can be removed
        for col in range(rank - 1, -1, -1):
            if time.time() - start_time > deadline - 5:
                break
            # Check if this column contributes to any (i,j,k) term
            col_contributes = False
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j, col]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j*p+k, col]
                        if v_val == 0:
                            continue
                        w_val = W[k*n+i, col]
                        if w_val == 0:
                            continue
                        # This column contributes to (i,j,k)
                        # Check if other columns also contribute
                        other_contrib = False
                        for c2 in range(rank):
                            if c2 == col:
                                continue
                            if U[i*m+j, c2] != 0 and V[j*p+k, c2] != 0 and W[k*n+i, c2] != 0:
                                other_contrib = True
                                break
                        if not other_contrib:
                            col_contributes = True
                            break
                    if col_contributes:
                        break
                if col_contributes:
                    break
            if not col_contributes:
                # Remove this column
                mask = np.ones(rank, dtype=bool)
                mask[col] = False
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                rank -= 1
                if verify_decomposition(U, V, W, rank):
                    continue
                else:
                    # Restore
                    break
        return U, V, W, rank
    
    def random_search_mnp(n, m, p, target_rank, max_attempts=50, seed_offset=0):
        """Random search for M_{n,m,p} decomposition."""
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 10:
                return None
            
            np.random.seed(attempt * 1000003 + target_rank + seed_offset)
            
            # Use entries from {-1, 0, 1} for better chance of exact solutions
            U = np.random.choice([-1, 0, 1], size=(n*m, target_rank)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m*p, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
            
            # Solve for W: W[k*n+i, c] = M[i,k,c] / ||M[i,k,:]||^2
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to integers (try both integer and half-integer)
            for round_mode in ['int', 'half']:
                if round_mode == 'int':
                    U_r = np.round(U).astype(float)
                    V_r = np.round(V).astype(float)
                    W_r = np.round(W).astype(float)
                else:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                
                # Verify
                is_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                            if abs(s - 1.0) > 1e-8:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        break
                if is_valid:
                    # Try to reduce rank
                    U_r, V_r, W_r, new_rank = try_rank_reduction(U_r, V_r, W_r, target_rank)
                    return (U_r, V_r, W_r), new_rank
        return None
    
    # Strategy 1: Tensor product M_{3,4,4} ⊗ M_{1,1,2}
    # First find good M_{3,4,4}
    n2, m2, p2 = 3, 4, 4
    
    def verify_344(U, V, W, rank):
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def schoolbook_344():
        rank = n2 * m2 * p2
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[i*m2+j, col] = 1
                    V[j*p2+k, col] = 1
                    W[k*n2+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    best_344, best_344_rank = schoolbook_344()
    
    # Search for better M_{3,4,4}
    for target_rank in range(40, 10, -1):
        if time.time() - start_time > deadline - 30:
            break
        
        result = random_search_mnp(n2, m2, p2, target_rank, max_attempts=30, seed_offset=100)
        if result is not None:
            U344, V344, W344 = result
            new_rank = result[1] if isinstance(result, tuple) and len(result) > 1 else target_rank
            if new_rank < best_344_rank:
                best_344 = (U344, V344, W344)
                best_344_rank = new_rank
                break
    
    # Combine via tensor product
    U_112 = np.array([[1, 1]], dtype=float)
    V_112 = np.array([[1, 0], [0, 1]], dtype=float)
    W_112 = np.array([[1, 0], [0, 1]], dtype=float)
    
    U344, V344, W344 = best_344
    rank_total = best_344_rank * 2
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    for i1 in range(3):
        for j1 in range(4):
            for k1 in range(4):
                for c1 in range(best_344_rank):
                    for c2 in range(2):
                        c = c1 * 2 + c2
                        # U[(i1)*4 + j1, c] = U344[i1*4+j1, c1] * U_112[0, c2]
                        u_val = U344[i1*4+j1, c1] * U_112[0, c2]
                        if u_val != 0:
                            U[i1*4 + j1, c] = u_val
                        
                        # V[j1*8 + k1*2 + c2, c] = V344[j1*4+k1, c1] * V_112[c2, c2]
                        # Wait, V_112 is 2x2, V_112[j2, c2] where j2 is the "m" index (0 or 1)
                        # For M_{1,1,2}, V has shape (1*2, 2) = (2, 2)
                        # V_112[j2, c2] where j2=0,1
                        # In the tensor product, V_total[j1*1+j2, k1*2+k2, c1*2+c2]
                        # = V_344[j1, k1, c1] * V_112[j2, c2]
                        # But j2 only goes 0..0 (since m=1 for M_{1,1,2})
                        # So j2=0 always
                        v_val = V344[j1*4+k1, c1] * V_112[0, c2]
                        if v_val != 0:
                            V[j1*8 + k1*2 + c2, c] = v_val
                        
                        # W[(k1*1+c2)*3 + i1, c] = W344[k1*3+i1, c1] * W_112[c2, c2]
                        # W_112 has shape (2*1, 2) = (2, 2)
                        # W_112[k2, c2] where k2=0,1
                        w_val = W344[k1*3+i1, c1] * W_112[c2, c2]
                        if w_val != 0:
                            W[(k1 + c2)*3 + i1, c] = w_val
    
    if verify_decomposition(U, V, W, rank_total):
        if rank_total < best_rank:
            best_rank = rank_total
            best_result = (U, V, W)
    
    # Strategy 2: Direct random search for M_{3,4,8}
    for target_rank in [best_rank - 1, best_rank - 2, best_rank - 4, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8]:
        if time.time() - start_time > deadline - 5:
            break
        if target_rank >= best_rank or target_rank < 1:
            continue
        
        result = random_search_mnp(n, m, p, target_rank, max_attempts=40, seed_offset=200)
        if result is not None:
            U_new, V_new, W_new = result
            new_rank = result[1] if isinstance(result, tuple) and len(result) > 1 else target_rank
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = (U_new, V_new, W_new)
                break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
