import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Strassen steps for 2x2x2
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Construct the rank-21 solution first as a fallback
    U21 = np.zeros((4, 21))
    V21 = np.zeros((12, 21))
    W21 = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U21[l, r] = step['u'][l]
                # V mapping: local l -> global b
                # l=0 (00): j=0, k=2g -> b = 0*6 + 2g
                # l=1 (01): j=0, k=2g+1 -> b = 0*6 + 2g+1
                # l=2 (10): j=1, k=2g -> b = 1*6 + 2g
                # l=3 (11): j=1, k=2g+1 -> b = 1*6 + 2g+1
                if l == 0: V21[2*g, r] = step['v'][l]
                elif l == 1: V21[2*g+1, r] = step['v'][l]
                elif l == 2: V21[6+2*g, r] = step['v'][l]
                elif l == 3: V21[6+2*g+1, r] = step['v'][l]
                
                # W mapping: local l -> global c
                # l=0 (00): k=2g, i=0 -> c = 2g*2 + 0
                # l=1 (01): k=2g+1, i=0 -> c = (2g+1)*2 + 0
                # l=2 (10): k=2g, i=1 -> c = 2g*2 + 1
                # l=3 (11): k=2g+1, i=1 -> c = (2g+1)*2 + 1
                if l == 0: W21[4*g, r] = step['w'][l]
                elif l == 1: W21[4*g+2, r] = step['w'][l]
                elif l == 2: W21[4*g+1, r] = step['w'][l]
                elif l == 3: W21[4*g+3, r] = step['w'][l]
                
    # Verify rank 21
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    T21 = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T21, T_exp, atol=1e-8):
        # Fallback to schoolbook if Strassen construction failed (should not happen)
        U_s = np.zeros((4, 24))
        V_s = np.zeros((12, 24))
        W_s = np.zeros((12, 24))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Attempt to find Rank 20
    # We use the first 7 terms (Strassen for block 0) and search for 13 more terms
    # that cover the residual.
    
    U0 = U21[:, :7]
    V0 = V21[:, :7]
    W0 = W21[:, :7]
    
    T0 = np.einsum('ar,br,cr->abc', U0, V0, W0)
    T_res = T_exp - T0
    
    # T_res should be zero for B indices 0,1 (block 0) and non-zero for B indices 2-5
    # We need to find 13 vectors u, v, w such that sum(u_i v_i w_i) = T_res
    
    # Random search with sparse vectors
    # To speed up, we only sample vectors that are non-zero in the relevant parts
    # U: always 4 dims
    # V: only indices 4-11 (B cols 2-5) can be non-zero? 
    #    Actually, V can have non-zeros anywhere, but to cancel T_res, 
    #    the contributions to B indices 0,1 must cancel out or be zero.
    #    Since T0 already perfectly matches T_exp for B indices 0,1, 
    #    the new terms must have ZERO contribution to B indices 0,1.
    #    Therefore, V vectors must be zero at indices 0,1,6,7 (B cols 0,1).
    #    Wait, B indices for block 0 are 0,1 (j=0,1; k=0,1).
    #    Block 1 B indices are 2,3 (j=0,1; k=2,3).
    #    Block 2 B indices are 4,5 (j=0,1; k=4,5).
    #    So V indices 0,1,6,7 correspond to k=0,1.
    #    We require V[0]=V[1]=V[6]=V[7]=0 for the extra terms.
    
    # W indices:
    # C indices for k=0,1 are 0,1,2,3.
    # C indices for k=2,3 are 4,5,6,7.
    # C indices for k=4,5 are 8,9,10,11.
    # T_res is non-zero only for C indices corresponding to k=2,3,4,5.
    # So W vectors must be zero at indices 0,1,2,3.
    
    # So we search for 13 terms with:
    # u: (4,)
    # v: (12,) with v[0]=v[1]=v[6]=v[7]=0
    # w: (12,) with w[0]=w[1]=w[2]=w[3]=0
    
    # We will perform a random search.
    # To make it more structured, we can try to match the pattern of the next Strassen block.
    # The next block (g=1) covers k=2,3.
    # The Strassen block for g=1 uses 7 terms.
    # If we just take the next 7 Strassen terms, we cover k=2,3.
    # Then we need 6 more terms to cover k=4,5.
    # Total 7 (block 0) + 7 (block 1) + 6 (block 2 partial) = 20?
    # But block 2 is a 2x2x2 block, which needs 7 terms.
    # So 7+7+7=21.
    # Can we save 1 term by mixing?
    
    # Let's try a random search for 13 terms.
    # We'll use a set of candidate vectors.
    
    def random_sparse_vector(dim, non_zero_indices, max_val=2):
        v = np.zeros(dim)
        for idx in non_zero_indices:
            if random.random() < 0.5: # 50% chance to be non-zero
                v[idx] = random.choice([-max_val, -1, 1, max_val])
        return v
    
    u_indices = [0, 1, 2, 3]
    v_indices = [2, 3, 4, 5, 8, 9, 10, 11] # B indices for k>=2
    w_indices = [4, 5, 6, 7, 8, 9, 10, 11] # C indices for k>=2
    
    best_rank = 21
    best_U, best_V, best_W = U21, V21, W21
    
    # Try to find 13 terms
    num_extra = 13
    max_iters = 5000
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline - 10:
            break
            
        U_extra = np.zeros((4, num_extra))
        V_extra = np.zeros((12, num_extra))
        W_extra = np.zeros((12, num_extra))
        
        for r in range(num_extra):
            U_extra[:, r] = random_sparse_vector(4, u_indices)
            V_extra[:, r] = random_sparse_vector(12, v_indices)
            W_extra[:, r] = random_sparse_vector(12, w_indices)
            
        T_calc = np.einsum('ar,br,cr->abc', U_extra, V_extra, W_extra)
        if np.allclose(T_calc, T_res, atol=1e-8):
            # Found a solution!
            U_new = np.hstack([U0, U_extra])
            V_new = np.hstack([V0, V_extra])
            W_new = np.hstack([W0, W_extra])
            
            # Verify full tensor
            T_full = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            if np.allclose(T_full, T_exp, atol=1e-8):
                best_rank = 20
                best_U, best_V, best_W = U_new, V_new, W_new
                break
                
        iter_count += 1
        
    if best_rank == 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # Fallback to Rank 21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
