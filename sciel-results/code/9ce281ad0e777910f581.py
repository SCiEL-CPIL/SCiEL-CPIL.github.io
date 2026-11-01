import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Target: Decompose M_{2,3,3} and replicate
    # M_{2,3,3} tensor
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Build sparse tensor T for M_{2,3,3}
    # Shape: (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub) = (6, 9, 6)
    T = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                T[a, b, c] = 1.0

    best_rank = n_sub * m_sub * p_sub # 18
    best_factors = None
    
    # Try to find low rank decomposition for M_{2,3,3}
    for r in range(10, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W
            U = np.random.randn(n_sub*m_sub, r)
            V = np.random.randn(m_sub*p_sub, r)
            W = np.random.randn(p_sub*n_sub, r)
            
            # ALS iterations
            for it in range(50):
                # Update U
                for c in range(r):
                    # vec(u_c) = ((w_c ⊗ v_c)^T (w_c ⊗ v_c))^-1 (w_c ⊗ v_c)^T vec(R)
                    # R = T - sum_{c' != c} u_c' ⊗ v_c' ⊗ w_c'
                    # This is expensive to compute R each time.
                    # Instead, use the fact that we are minimizing ||T - sum U_c ⊗ V_c ⊗ W_c||
                    # We can update U by solving a least squares problem on the flattened tensor.
                    # Let's use a simpler heuristic: 
                    # Project T onto the subspace spanned by V_c ⊗ W_c
                    # Actually, the standard ALS update for U_c is:
                    # U_c = ( (W_c^T ⊗ V_c^T) T ) / ( (W_c^T ⊗ V_c^T) (U_c ⊗ V_c ⊗ W_c) ) ? No.
                    
                    # Let's use the normal equation for the single factor update.
                    # Minimize || T - u ⊗ v_c ⊗ w_c - Rest ||
                    # Let M = T - Rest.
                    # vec(M) ≈ (w_c ⊗ v_c) vec(u).
                    # vec(u) = pinv(w_c ⊗ v_c) @ vec(M).
                    
                    # To avoid computing Rest explicitly (O(r * size)), we can accumulate.
                    # But size is small (324). r is ~10. 50 iters.
                    # 5 * 10 * 50 * 324 * 10 ~ 800k ops. Fast.
                    
                    Rest = np.zeros_like(T)
                    for c2 in range(r):
                        if c2 == c: continue
                        Rest += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                    
                    M = T - Rest
                    vec_M = M.flatten()
                    
                    # Construct matrix A = w_c ⊗ v_c
                    # Shape (6*9, 6) = (54, 6)
                    A_mat = np.kron(W[:, c], V[:, c])
                    
                    # Solve A_mat @ u_vec = vec_M
                    u_vec, residuals, rank_s, sv = np.linalg.lstsq(A_mat, vec_M, rcond=None)
                    U[:, c] = u_vec.reshape(-1)
                    
                # Update V
                for c in range(r):
                    Rest = np.zeros_like(T)
                    for c2 in range(r):
                        if c2 == c: continue
                        Rest += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                    M = T - Rest
                    vec_M = M.flatten()
                    A_mat = np.kron(W[:, c], U[:, c]) # Shape (6*6, 9) = (36, 9)
                    v_vec, _, _, _ = np.linalg.lstsq(A_mat, vec_M, rcond=None)
                    V[:, c] = v_vec.reshape(-1)
                    
                # Update W
                for c in range(r):
                    Rest = np.zeros_like(T)
                    for c2 in range(r):
                        if c2 == c: continue
                        Rest += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                    M = T - Rest
                    vec_M = M.flatten()
                    A_mat = np.kron(V[:, c], U[:, c]) # Shape (9*6, 6) = (54, 6)
                    w_vec, _, _, _ = np.linalg.lstsq(A_mat, vec_M, rcond=None)
                    W[:, c] = w_vec.reshape(-1)
                    
            # Check error
            T_approx = np.zeros_like(T)
            for c in range(r):
                T_approx += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            error = np.linalg.norm(T - T_approx)
            
            if error < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_snap = np.zeros_like(T)
                for c in range(r):
                    T_snap += np.einsum('i,j,k->ijk', U_snap[:, c], V_snap[:, c], W_snap[:, c])
                
                if np.allclose(T, T_snap, atol=1e-9):
                    best_rank = r
                    best_factors = (U_snap, V_snap, W_snap)
                    found = True
                    break
        
        if found:
            break
            
    if best_factors is None:
        # Fall back to schoolbook for M_{2,3,3}
        r = n_sub * m_sub * p_sub
        U = np.zeros((n_sub*m_sub, r))
        V = np.zeros((m_sub*p_sub, r))
        W = np.zeros((p_sub*n_sub, r))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i*m_sub+j, col] = 1
                    V[j*p_sub+k, col] = 1
                    W[k*n_sub+i, col] = 1
                    col += 1
        best_factors = (U, V, W)
        best_rank = r
        
    # Construct final answer for M_{2,3,9}
    # Replicate the M_{2,3,3} decomposition 3 times
    U_sub, V_sub, W_sub = best_factors
    r_sub = U_sub.shape[1]
    R = r_sub * 3
    
    U_final = np.zeros((n*m, R))
    V_final = np.zeros((m*p, R))
    W_final = np.zeros((p*n, R))
    
    # Map indices
    # n=2, m=3, p=9
    # Sub-blocks correspond to k ranges: 0-2, 3-5, 6-8
    for block in range(3):
        k_offset = block * 3
        
        # U is shared? No, U depends on i, j. It is the same for all blocks.
        # So U_final[:, block*r_sub : (block+1)*r_sub] = U_sub
        U_final[:, block*r_sub : (block+1)*r_sub] = U_sub
        
        # V depends on j, k.
        # V_sub has indices j in 0..2, k in 0..2.
        # V_final has indices j in 0..2, k in 0..8.
        # For block b, k_final = k_offset + k_sub.
        for j in range(m_sub):
            for k_sub in range(p_sub):
                k_final = k_offset + k_sub
                v_idx = j * p + k_final
                v_sub_idx = j * p_sub + k_sub
                V_final[v_idx, block*r_sub : (block+1)*r_sub] = V_sub[v_sub_idx, :]
                
        # W depends on k, i.
        # W_sub has indices k in 0..2, i in 0..1.
        # W_final has indices k in 0..8, i in 0..1.
        # For block b, k_final = k_offset + k_sub.
        for k_sub in range(p_sub):
            for i in range(n_sub):
                k_final = k_offset + k_sub
                w_idx = k_final * n + i
                w_sub_idx = k_sub * n_sub + i
                W_final[w_idx, block*r_sub : (block+1)*r_sub] = W_sub[w_sub_idx, :]
                
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
