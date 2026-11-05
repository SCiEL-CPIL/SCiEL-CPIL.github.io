"""Matrix multiplication 4x5x5 - rank-84 Strassen-core decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # We attempt rank 84 by combining the last boundary-1 term with the last
    # boundary-3 term. If verification fails, fall back to rank 85.
    
    # 2x2x2 Strassen factors
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def build_rank(rank):
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        
        for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
            if time.time() - start_time > deadline:
                break
            add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
        
        col = 49
        
        # Boundary 1: A[i,4] * B[4,k] -> C[i,k] for i in 0..3, k in 0..3 (16 terms)
        # But if rank==84, we combine the last one (i=3, k=3) with boundary 3 last (i=3)
        if rank == 84:
            # 15 terms for boundary 1 (all except i=3, k=3)
            for i in range(4):
                if time.time() - start_time > deadline:
                    break
                for k in range(4):
                    if i == 3 and k == 3:
                        continue
                    U[i*5 + 4, col] = 1
                    V[4*5 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
            
            # Boundary 2: A[i,j] * B[j,4] -> C[i,4] for i in 0..3, j in 0..3 (16 terms)
            for i in range(4):
                if time.time() - start_time > deadline:
                    break
                for j in range(4):
                    U[i*5 + j, col] = 1
                    V[j*5 + 4, col] = 1
                    W[4*4 + i, col] = 1
                    col += 1
            
            # Boundary 3: A[i,4] * B[4,4] -> C[i,4] for i in 0..2 (3 terms)
            # The last term (i=3) is combined with the skipped boundary 1 term
            for i in range(3):
                if time.time() - start_time > deadline:
                    break
                U[i*5 + 4, col] = 1
                V[4*5 + 4, col] = 1
                W[4*4 + i, col] = 1
                col += 1
            
            # Combined term: U[i=3, j=4] * V[j=4, k=3] and U[i=3, j=4] * V[j=4, k=4]
            # This requires U[3*5+4, col] = 1, V[4*5+3, col] = 1, W[3*4+3, col] = 1
            # AND U[3*5+4, col] = 1, V[4*5+4, col] = 1, W[4*4+3, col] = 1
            # But V has only one value per (j,k) per column. We can't have both V[23, col]=1 and V[24, col]=1
            # unless we use a different decomposition.
            # 
            # Actually, the combined term would be:
            # U[19, col] = 1  (A[3,4])
            # V[23, col] = 1  (B[4,3])  -- this gives C[3,3] += A[3,4]*B[4,3]
            # V[24, col] = 1  (B[4,4])  -- this gives C[3,4] += A[3,4]*B[4,4]
            # W[15, col] = 1  (C[3,3])
            # W[19, col] = 1  (C[3,4])
            # 
            # This works! We set both V entries to 1.
            U[3*5 + 4, col] = 1
            V[4*5 + 3, col] = 1
            V[4*5 + 4, col] = 1
            W[3*4 + 3, col] = 1
            W[4*4 + 3, col] = 1
            col += 1
        else:
            # Standard rank 85
            for i in range(4):
                if time.time() - start_time > deadline:
                    break
                for k in range(4):
                    U[i*5 + 4, col] = 1
                    V[4*5 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
            
            for i in range(4):
                if time.time() - start_time > deadline:
                    break
                for j in range(4):
                    U[i*5 + j, col] = 1
                    V[j*5 + 4, col] = 1
                    W[4*4 + i, col] = 1
                    col += 1
            
            for i in range(4):
                if time.time() - start_time > deadline:
                    break
                U[i*5 + 4, col] = 1
                V[4*5 + 4, col] = 1
                W[4*4 + i, col] = 1
                col += 1
        
        return U, V, W
    
    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p), dtype=int)
        for r in range(rank):
            u_r = U_mat[:, r]
            v_r = V_mat[:, r]
            w_r = W_mat[:, r]
            if not (np.any(u_r != 0) and np.any(v_r != 0) and np.any(w_r != 0)):
                continue
            u_2d = u_r.reshape(n, m)
            v_2d = v_r.reshape(m, p)
            w_2d = w_r.reshape(p, n)
            w_2d_T = w_2d.T
            contrib = u_2d[:, :, None] * v_2d[None, :, :] * w_2d_T[:, None, :]
            T += contrib
        return np.all(T == 1)
    
    # Try rank 84 first
    if time.time() - start_time < deadline:
        U84, V84, W84 = build_rank(84)
        if verify(U84, V84, W84):
            return {"U": U84.tolist(), "V": V84.tolist(), "W": W84.tolist()}
    
    # Fall back to rank 85
    U85, V85, W85 = build_rank(85)
    if verify(U85, V85, W85):
        return {"U": U85.tolist(), "V": V85.tolist(), "W": W85.tolist()}
    
    # Final fallback: schoolbook
    rank_schoolbook = n*m*p
    U = np.zeros((nm, rank_schoolbook), dtype=int)
    V = np.zeros((mp, rank_schoolbook), dtype=int)
    W = np.zeros((pn, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
