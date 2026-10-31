import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    Attempts to find linear dependencies to potentially reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Build the decomposition by recursively applying Strassen.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # A decomposition of rank r gives U, V, W where:
    # U: shape (16, r) -- indices (i*m+j) for i,j in {0,1,2,3}, m=4
    # V: shape (16, r) -- indices (j*p+k) for j,k in {0,1,2,3}, p=4  
    # W: shape (16, r) -- indices (k*n+i) for k,i in {0,1,2,3}, n=4
    # with T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strassen 2x2 decomposition
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # S1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # S2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # S3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # S4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # S5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # S6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # S7
    ]
    
    outer_strassen = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # M2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # M5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # M6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # M7
    ]
    
    # Build all 49 rank-1 terms
    terms = []
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_strassen):
        for inner_idx, (a_comb, b_comb, c_comb) in enumerate(strassen_2x2):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (lr, lc, ca) in a_comb:
                for (br, bc, coeff) in a_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    u_idx = gr * 4 + gc
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
            
            for (lr, lc, cb) in b_comb:
                for (br, bc, coeff) in b_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    v_idx = gr * 4 + gc
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
            
            for (lr, lc, cc) in c_comb:
                for (br, bc, coeff) in c_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    w_idx = gc * 4 + gr
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
            
            terms.append((u_dict, v_dict, w_dict))
    
    rank = len(terms)  # 49
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_dict.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_dict.items():
            if val != 0:
                W[idx][col] = val
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*4+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies among the 49 terms
    # The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r
    # If some linear combination of the rank-1 tensors equals zero, we have a dependency
    # We can check this by flattening each term into a vector of size 16*16*16 = 4096
    # and checking the rank of the 49x4096 matrix
    
    if time.time() < deadline - 30:  # Only if we have time
        # Flatten each rank-1 term into a vector
        flat_terms = []
        for u_dict, v_dict, w_dict in terms:
            vec = np.zeros(16*16*16)
            for u_idx, u_val in u_dict.items():
                for v_idx, v_val in v_dict.items():
                    for w_idx, w_val in w_dict.items():
                        vec[u_idx * 256 + v_idx * 16 + w_idx] += u_val * v_val * w_val
            flat_terms.append(vec)
        
        flat_matrix = np.array(flat_terms)
        
        # Check rank using SVD
        try:
            svd_rank = np.linalg.matrix_rank(flat_matrix, tol=1e-10)
            
            if svd_rank < rank:
                # There are linear dependencies! Try to find a minimal generating set
                # Use QR decomposition to find independent vectors
                Q, R = np.linalg.qr(flat_matrix.T, mode='reduced')
                
                # Find pivot columns
                diag = np.abs(np.diag(R))
                threshold = 1e-10 * max(diag)
                pivots = np.where(diag > threshold)[0]
                
                if len(pivots) < rank:
                    # Use only the independent terms
                    new_terms = [terms[i] for i in pivots]
                    new_rank = len(new_terms)
                    
                    # Rebuild U, V, W
                    U_new = [[0] * new_rank for _ in range(16)]
                    V_new = [[0] * new_rank for _ in range(16)]
                    W_new = [[0] * new_rank for _ in range(16)]
                    
                    for col, (u_dict, v_dict, w_dict) in enumerate(new_terms):
                        for idx, val in u_dict.items():
                            if val != 0:
                                U_new[idx][col] = val
                        for idx, val in v_dict.items():
                            if val != 0:
                                V_new[idx][col] = val
                        for idx, val in w_dict.items():
                            if val != 0:
                                W_new[idx][col] = val
                    
                    # Verify
                    U_np_new = np.array(U_new, dtype=float)
                    V_np_new = np.array(V_new, dtype=float)
                    W_np_new = np.array(W_new, dtype=float)
                    
                    max_err_new = 0.0
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                s = 0.0
                                for r in range(new_rank):
                                    s += U_np_new[i*4+j, r] * V_np_new[j*4+k, r] * W_np_new[k*4+i, r]
                                max_err_new = max(max_err_new, abs(s - 1.0))
                    
                    if max_err_new < 1e-9:
                        return {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass
    
    # Return the original rank-49 decomposition
    return {"U": U, "V": V, "W": W}
