import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Precompute schoolbook solution as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                c += 1
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Greedy tensor decomposition search
    # T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Precompute all possible index combinations
    idx_U = list(range(n * m))
    idx_V = list(range(m * p))
    idx_W = list(range(p * n))
    
    # For each rank r, we need to choose (u_r, v_r, w_r) such that
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    
    # Greedy approach: iteratively pick terms that maximize residual coverage
    # Start with full tensor T[i,j,k] = 1
    residual = np.ones((n, m, p))
    
    max_rank_attempt = min(rank_schoolbook, 40)  # Try to find rank <= 40
    
    for target_rank in range(1, max_rank_attempt + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Reset residual
        residual = np.ones((n, m, p))
        U_greedy = np.zeros((n * m, target_rank))
        V_greedy = np.zeros((m * p, target_rank))
        W_greedy = np.zeros((p * n, target_rank))
        
        success = True
        
        for r in range(target_rank):
            if time.time() - start_time > DEADLINE - 5:
                success = False
                break
            
            # Find the term that best covers the residual
            # A term is defined by (i, j, k) indices with coefficients
            # For simplicity, try unit vectors first, then small integer vectors
            
            best_score = -1
            best_i, best_j, best_k = 0, 0, 0
            best_u_coeff, best_v_coeff, best_w_coeff = 1, 1, 1
            
            # Try all combinations of (i, j, k) with small coefficients
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute how well this term fits the residual
                        # The term contributes to T[i,j,k]
                        # We want to maximize the reduction in residual norm
                        
                        # Simple heuristic: pick where residual is largest
                        score = residual[i, j, k]
                        
                        if score > best_score:
                            best_score = score
                            best_i, best_j, best_k = i, j, k
                            best_u_coeff, best_v_coeff, best_w_coeff = 1, 1, 1
            
            if best_score < 1e-10:
                # Residual is essentially zero
                break
            
            # Set the term
            U_greedy[best_i * m + best_j, r] = best_u_coeff
            V_greedy[best_j * p + best_k, r] = best_v_coeff
            W_greedy[best_k * n + best_i, r] = best_w_coeff
            
            # Update residual
            # The contribution to T[i',j',k'] from this term is:
            # U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
            # But our term only has non-zero at specific (i,j,k)
            # Actually, this greedy approach is too simplistic
            
            # Let me try a different approach: use random sampling with verification
            
            success = False
            break
        
        if not success:
            continue
        
        # Verify the decomposition
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U_greedy[i*m+j, r] * V_greedy[j*p+k, r] * W_greedy[k*n+i, r]
                    if abs(s - 1) > max_err:
                        max_err = abs(s - 1)
        
        if max_err < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U_greedy, V_greedy, W_greedy
            break
    
    # If greedy didn't find anything better, try the block decomposition
    if best_rank >= rank_schoolbook:
        if time.time() - start_time < DEADLINE - 10:
            # Try the original block decomposition
            rank_total = 54
            U = np.zeros((n * m, rank_total))
            V = np.zeros((m * p, rank_total))
            W = np.zeros((p * n, rank_total))
            
            # Helper to get Strassen 2x2x2 matrices
            def get_strassen_222():
                U_s = np.zeros((4, 7))
                V_s = np.zeros((4, 7))
                W_s = np.zeros((4, 7))
                
                # Term 1: M1 = (A00+A11)(B00+B11)
                U_s[0, 0] = 1; U_s[3, 0] = 1
                V_s[0, 0] = 1; V_s[3, 0] = 1
                W_s[0, 0] = 1; W_s[3, 0] = 1
                
                # Term 2: M2 = (A10+A11)B00
                U_s[2, 1] = 1; U_s[3, 1] = 1
                V_s[0, 1] = 1
                W_s[2, 1] = 1
                
                # Term 3: M3 = A00(B01-B11)
                U_s[0, 2] = 1
                V_s[1, 2] = 1; V_s[3, 2] = -1
                W_s[2, 2] = 1; W_s[3, 2] = -1
                
                # Term 4: M4 = A11(B10-B00)
                U_s[3, 3] = 1
                V_s[2, 3] = 1; V_s[0, 3] = -1
                W_s[0, 3] = 1; W_s[2, 3] = 1
                
                # Term 5: M5 = (A00+A10)B11
                U_s[0, 4] = 1; U_s[2, 4] = 1
                V_s[3, 4] = 1
                W_s[0, 4] = -1; W_s[2, 4] = 1
                
                # Term 6: M6 = (A01-A00)(B00+B10)
                U_s[1, 5] = 1; U_s[0, 5] = -1
                V_s[0, 5] = 1; V_s[2, 5] = 1
                W_s[3, 5] = 1
                
                # Term 7: M7 = (A11-A10)(B01+B11)
                U_s[3, 6] = 1; U_s[2, 6] = -1
                V_s[1, 6] = 1; V_s[3, 6] = 1
                W_s[0, 6] = 1
                
                return U_s, V_s, W_s
            
            # Helper to get 2x1x1 matrices
            def get_trivial_211():
                U_t = np.zeros((2, 2))
                V_t = np.zeros((1, 2))
                W_t = np.zeros((2, 2))
                
                # Term 1: A00 * B00 -> C00
                U_t[0, 0] = 1
                V_t[0, 0] = 1
                W_t[0, 0] = 1
                
                # Term 2: A10 * B00 -> C10
                U_t[1, 1] = 1
                V_t[0, 1] = 1
                W_t[1, 1] = 1
                
                return U_t, V_t, W_t
            
            U_s, V_s, W_s = get_strassen_222()
            U_t, V_t, W_t = get_trivial_211()
            
            col = 0
            
            # Process 2x2x6 blocks (j=0,1 and j=2,3)
            for block_idx in range(2):
                j_start = block_idx * 2
                for k_block in range(3):
                    k_start = k_block * 2
                    
                    for r in range(7):
                        for i in range(2):
                            for j_loc in range(2):
                                val = U_s[i*2+j_loc, r]
                                if val != 0:
                                    j_glob = j_start + j_loc
                                    U[i*5 + j_glob, col] = val
                                    
                        for j_loc in range(2):
                            for k_loc in range(2):
                                val = V_s[j_loc*2+k_loc, r]
                                if val != 0:
                                    j_glob = j_start + j_loc
                                    k_glob = k_start + k_loc
                                    V[j_glob*6 + k_glob, col] = val
                                    
                        for k_loc in range(2):
                            for i in range(2):
                                val = W_s[k_loc*2+i, r]
                                if val != 0:
                                    k_glob = k_start + k_loc
                                    W[k_glob*2 + i, col] = val
                                    
                        col += 1
                        
            # Process 2x1x6 block (j=4)
            for k in range(6):
                for r in range(2):
                    for i in range(2):
                        val = U_t[i, r]
                        if val != 0:
                            U[i*5 + 4, col] = val
                            
                    val = V_t[0, r]
                    if val != 0:
                        V[4*6 + k, col] = val
                        
                    for i in range(2):
                        val = W_t[i, r]
                        if val != 0:
                            W[k*2 + i, col] = val
                            
                    col += 1
                    
            # Verify block decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank_total):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1) > max_err:
                            max_err = abs(s - 1)
            
            if max_err < 1e-6:
                best_rank = rank_total
                best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
