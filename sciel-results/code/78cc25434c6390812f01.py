"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Strategy: Decompose into 3 blocks of 2x3 by 3x3
    # Each block has rank 15 (known result). Total rank 45.
    # We will try to find the rank 15 decomposition for 2x3 by 3x3.
    
    n_block, m_block, p_block = 2, 3, 3
    dim_u_b = n_block * m_block  # 6
    dim_v_b = m_block * p_block  # 9
    dim_w_b = p_block * n_block  # 6
    target_rank_b = 15
    
    def find_block_decomposition(n_b, m_b, p_b, rank_b, max_iters=500, trials=20):
        """Find a rank-rank_b decomposition for n_b x m_b by m_b x p_b."""
        dim_u_b = n_b * m_b
        dim_v_b = m_b * p_b
        dim_w_b = p_b * n_b
        
        # Build target tensor for the block
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # a in [0, dim_u_b), b in [0, dim_v_b), c in [0, dim_w_b)
        
        for trial in range(trials):
            if time.time() > deadline - 10:
                return None
            
            rng = np.random.RandomState(trial)
            U = rng.randn(dim_u_b, rank_b) * 0.5
            V = rng.randn(dim_v_b, rank_b) * 0.5
            W = rng.randn(dim_w_b, rank_b) * 0.5
            
            # Alternating minimization
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U
                A_rows = []
                b_vec = []
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            a = i*m_b+j
                            b_idx = j*p_b+k
                            c_idx = k*n_b+i
                            # Equation: sum_d U[a,d] V[b_idx,d] W[c_idx,d] = 1
                            row = np.zeros(dim_u_b * rank_b)
                            for d in range(rank_b):
                                row[a*rank_b + d] = V[b_idx, d] * W[c_idx, d]
                            A_rows.append(row)
                            b_vec.append(1.0)
                
                # Add zero constraints (sampled)
                # To ensure exactness, we should check all zeros, but for speed, we can rely on the structure
                # Actually, for small dimensions, we can include all zero constraints.
                # Total equations: dim_u_b * dim_v_b * dim_w_b = 6*9*6 = 324.
                # Unknowns: 6*15 = 90.
                # Overdetermined.
                
                # Let's add all zero constraints
                for a in range(dim_u_b):
                    for b in range(dim_v_b):
                        for c in range(dim_w_b):
                            i = a // m_b
                            j = a % m_b
                            j2 = b // p_b
                            k = b % p_b
                            k2 = c // n_b
                            i2 = c % n_b
                            if not (j == j2 and k == k2 and i == i2):
                                row = np.zeros(dim_u_b * rank_b)
                                for d in range(rank_b):
                                    row[a*rank_b + d] = V[b, d] * W[c, d]
                                A_rows.append(row)
                                b_vec.append(0.0)
                
                A = np.array(A_rows)
                b = np.array(b_vec)
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U = U_flat.reshape(dim_u_b, rank_b)
                except:
                    continue
                
                # Solve for V
                A_rows = []
                b_vec = []
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            a = i*m_b+j
                            b_idx = j*p_b+k
                            c_idx = k*n_b+i
                            row = np.zeros(dim_v_b * rank_b)
                            for d in range(rank_b):
                                row[b_idx*rank_b + d] = U[a, d] * W[c_idx, d]
                            A_rows.append(row)
                            b_vec.append(1.0)
                
                for a in range(dim_u_b):
                    for b in range(dim_v_b):
                        for c in range(dim_w_b):
                            i = a // m_b
                            j = a % m_b
                            j2 = b // p_b
                            k = b % p_b
                            k2 = c // n_b
                            i2 = c % n_b
                            if not (j == j2 and k == k2 and i == i2):
                                row = np.zeros(dim_v_b * rank_b)
                                for d in range(rank_b):
                                    row[b*rank_b + d] = U[a, d] * W[c, d]
                                A_rows.append(row)
                                b_vec.append(0.0)
                
                A = np.array(A_rows)
                b = np.array(b_vec)
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V = V_flat.reshape(dim_v_b, rank_b)
                except:
                    continue
                
                # Solve for W
                A_rows = []
                b_vec = []
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            a = i*m_b+j
                            b_idx = j*p_b+k
                            c_idx = k*n_b+i
                            row = np.zeros(dim_w_b * rank_b)
                            for d in range(rank_b):
                                row[c_idx*rank_b + d] = U[a, d] * V[b_idx, d]
                            A_rows.append(row)
                            b_vec.append(1.0)
                
                for a in range(dim_u_b):
                    for b in range(dim_v_b):
                        for c in range(dim_w_b):
                            i = a // m_b
                            j = a % m_b
                            j2 = b // p_b
                            k = b % p_b
                            k2 = c // n_b
                            i2 = c % n_b
                            if not (j == j2 and k == k2 and i == i2):
                                row = np.zeros(dim_w_b * rank_b)
                                for d in range(rank_b):
                                    row[c*rank_b + d] = U[a, d] * V[b, d]
                                A_rows.append(row)
                                b_vec.append(0.0)
                
                A = np.array(A_rows)
                b = np.array(b_vec)
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W = W_flat.reshape(dim_w_b, rank_b)
                except:
                    continue
                
                # Check error
                err = 0.0
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            a = i*m_b+j
                            b_idx = j*p_b+k
                            c_idx = k*n_b+i
                            val = np.sum(U[a,:] * V[b_idx,:] * W[c_idx,:])
                            err += (val - 1)**2
                for a in range(dim_u_b):
                    for b in range(dim_v_b):
                        for c in range(dim_w_b):
                            i = a // m_b
                            j = a % m_b
                            j2 = b // p_b
                            k = b % p_b
                            k2 = c // n_b
                            i2 = c % n_b
                            if not (j == j2 and k == k2 and i == i2):
                                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                                err += val**2
                
                if err < 1e-10:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_r = 0.0
                    for i in range(n_b):
                        for j in range(m_b):
                            for k in range(p_b):
                                a = i*m_b+j
                                b_idx = j*p_b+k
                                c_idx = k*n_b+i
                                val = np.sum(U_r[a,:] * V_r[b_idx,:] * W_r[c_idx,:])
                                err_r += (val - 1)**2
                    for a in range(dim_u_b):
                        for b in range(dim_v_b):
                            for c in range(dim_w_b):
                                i = a // m_b
                                j = a % m_b
                                j2 = b // p_b
                                k = b % p_b
                                k2 = c // n_b
                                i2 = c % n_b
                                if not (j == j2 and k == k2 and i == i2):
                                    val = np.sum(U_r[a,:] * V_r[b,:] * W_r[c,:])
                                    err_r += val**2
                            
                    if err_r < 1e-10:
                        return U_r, V_r, W_r
        
        return None
    
    # Try to find the block decomposition
    block_decomp = find_block_decomposition(n_block, m_block, p_block, target_rank_b)
    
    if block_decomp is not None:
        U_b, V_b, W_b = block_decomp
        # Construct full decomposition
        # p = 9 = 3 * 3
        # We have 3 blocks
        full_rank = target_rank_b * 3
        U_full = np.zeros((dim_u, full_rank))
        V_full = np.zeros((dim_v, full_rank))
        W_full = np.zeros((dim_w, full_rank))
        
        # U is repeated
        for block_idx in range(3):
            U_full[:, block_idx*target_rank_b : (block_idx+1)*target_rank_b] = U_b
        
        # V and W are shifted
        for block_idx in range(3):
            k_offset = block_idx * p_block
            for j in range(m_block):
                for k in range(p_block):
                    v_idx = j * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + 0 # i=0
                    # Wait, W indices depend on i.
                    # In the block decomposition, W is 6x15.
                    # W[k*n+i, c].
                    # For i=0, W[k, c]. For i=1, W[3+k, c].
                    # So we need to map the block W to the full W.
                    pass
        
        # Let's do it more carefully
        for block_idx in range(3):
            k_offset = block_idx * p_block
            for c in range(target_rank_b):
                full_c = block_idx * target_rank_b + c
                # U is same for all blocks
                U_full[:, full_c] = U_b[:, c]
                
                # V: indices j*p + k
                for j in range(m_block):
                    for k in range(p_block):
                        v_block_idx = j * p_block + k
                        v_full_idx = j * p + (k_offset + k)
                        V_full[v_full_idx, full_c] = V_b[v_block_idx, c]
                        
                # W: indices k*n + i
                for k in range(p_block):
                    for i in range(n_block):
                        w_block_idx = k * n_block + i
                        w_full_idx = (k_offset + k) * n + i
                        W_full[w_full_idx, full_c] = W_b[w_block_idx, c]
        
        # Verify full decomposition
        # Check non-zero entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U_full[a,:] * V_full[b,:] * W_full[c,:])
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Check zero entries (sample)
            rng = np.random.RandomState(42)
            for _ in range(100):
                a = rng.randint(dim_u)
                b = rng.randint(dim_v)
                c = rng.randint(dim_w)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    val = np.sum(U_full[a,:] * V_full[b,:] * W_full[c,:])
                    if abs(val) > 1e-10:
                        valid = False
                        break
            
            if valid:
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = full_rank
    
    return best_result
# EVOLVE-BLOCK-END
