import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def check_round(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=1e-9):
            return U_r, V_r, W_r
        return None
    
    # Strassen steps for 2x2
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def build_block(g, s, U, V, W, r):
        """Add Strassen step s for block g at column r"""
        step = strassen_steps[s]
        k0 = 2 * g
        k1 = 2 * g + 1
        b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
        c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
        for l in range(4):
            U[l, r] = step['u'][l]
            V[b_map[l], r] = step['v'][l]
            W[c_map[l], r] = step['w'][l]
    
    # Build base rank-21
    U21 = np.zeros((nm, 21))
    V21 = np.zeros((mp, 21))
    W21 = np.zeros((pn, 21))
    for g in range(3):
        for s in range(7):
            build_block(g, s, U21, V21, W21, 7*g + s)
    
    # Verify base
    T_check = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T_check, T, atol=1e-10):
        # Fallback schoolbook rank 24
        U_s = np.zeros((nm, 24))
        V_s = np.zeros((mp, 24))
        W_s = np.zeros((pn, 24))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Structural approach: Try to find rank R < 21 by identifying
    # linear combinations of columns that sum to zero in the tensor product
    
    # Key insight: The three blocks have identical U structure.
    # U[:, 7g+s] is the same for all g (only V and W differ by block).
    # So we can write: sum_g sum_s U_s ⊗ V_{g,s} ⊗ W_{g,s}
    # = sum_s (U_s ⊗ (sum_g V_{g,s} ⊗ W_{g,s})^T) -- no, this doesn't work directly.
    
    # Better: Group by Strassen step s. For each s, we have 3 terms:
    # U_s ⊗ V_{0,s} ⊗ W_{0,s} + U_s ⊗ V_{1,s} ⊗ W_{1,s} + U_s ⊗ V_{2,s} ⊗ W_{2,s}
    # = U_s ⊗ (V_{0,s} ⊗ W_{0,s} + V_{1,s} ⊗ W_{1,s} + V_{2,s} ⊗ W_{2,s})^T
    # The middle factor is a 12x12 matrix. If its rank is 1, we can reduce.
    
    # For each s, compute M_s = sum_g V_{g,s} ⊗ W_{g,s} (as 12x12 matrix)
    # Check if M_s has rank 1 for any s.
    
    # Actually, let's try a different structural approach:
    # Use the fact that the problem is a "sum of 3 identical 2x2 problems"
    # and look for a rank-20 decomposition by finding one linear dependency.
    
    # Try: Can we find alpha_0, alpha_1, alpha_2 not all zero such that
    # sum_g alpha_g * (block g contribution) is a rank-1 tensor?
    
    # Build block tensors
    block_tensors = []
    for g in range(3):
        BT = np.zeros((nm, mp, pn))
        for s in range(7):
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
            c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
            for l in range(4):
                BT[l, b_map[l], c_map[l]] = step['u'][l] * step['v'][l] * step['w'][l]
        block_tensors.append(BT)
    
    # Search for rank-1 combination of blocks with small integer coefficients
    found_rank1 = None
    for a0 in range(-2, 3):
        for a1 in range(-2, 3):
            for a2 in range(-2, 3):
                if a0 == 0 and a1 == 0 and a2 == 0:
                    continue
                combined = a0 * block_tensors[0] + a1 * block_tensors[1] + a2 * block_tensors[2]
                # Check if rank 1 via matricization
                mat = combined.reshape(nm * mp, pn)
                try:
                    sv = np.linalg.svd(mat, compute_uv=False)
                except:
                    continue
                if len(sv) > 1 and sv[1] < 1e-10 and sv[0] > 1e-10:
                    found_rank1 = (a0, a1, a2, combined)
                    break
            if found_rank1:
                break
        if found_rank1:
            break
    
    if found_rank1:
        a0, a1, a2, combined = found_rank1
        # Extract rank-1 vectors
        nz = np.unravel_index(np.argmax(np.abs(combined)), combined.shape)
        u_vec = combined[:, nz[1], nz[2]] / combined[nz]
        v_vec = combined[nz[0], :, nz[2]] / combined[nz]
        w_vec = combined[nz[0], nz[1], :] / combined[nz]
        
        # Verify
        recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        if np.allclose(recon, combined, atol=1e-9):
            # Now build rank-20: use the rank-1 term to replace 3 terms
            # We need to express: a0*B0 + a1*B1 + a2*B2 = u⊗v⊗w
            # So: B0 + B1 + B2 = (1/a0)*u⊗v⊗w + (1-a0/a0)*B0 + ... 
            # Simpler: keep 6 terms from each block = 18, plus 1 rank-1 = 19? 
            # Actually: 21 terms - 3 terms (one from each block) + 1 rank-1 = 19
            # But we need to identify which specific terms to remove.
            
            # For now, try ALS with rank 20 from the rank-21 init with one column zeroed
            U = U21[:, :20].copy() + np.random.randn(nm, 20) * 0.05
            V = V21[:, :20].copy() + np.random.randn(mp, 20) * 0.05
            W = W21[:, :20].copy() + np.random.randn(pn, 20) * 0.05
            
            for it in range(80):
                if time.time() > deadline - 5:
                    break
                
                # Update W
                A_mat = np.zeros((24, pn * 20))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(20):
                                    A_mat[idx, c * 20 + r] = U[a, r] * V[b, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W = W_flat.reshape(pn, 20)
                except:
                    continue
                
                # Update U
                A_mat = np.zeros((24, nm * 20))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(20):
                                    A_mat[idx, a * 20 + r] = V[b, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_flat.reshape(nm, 20)
                except:
                    continue
                
                # Update V
                A_mat = np.zeros((24, mp * 20))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(20):
                                    A_mat[idx, b * 20 + r] = U[a, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V = V_flat.reshape(mp, 20)
                except:
                    continue
                
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_calc - T)
                if err < 1e-8:
                    result = check_round(U, V, W)
                    if result is not None:
                        return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
    
    # Try direct rank-20 ALS with multiple restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        np.random.seed(42 + restart * 7919)
        
        U = U21[:, :20].copy() + np.random.randn(nm, 20) * 0.1
        V = V21[:, :20].copy() + np.random.randn(mp, 20) * 0.1
        W = W21[:, :20].copy() + np.random.randn(pn, 20) * 0.1
        
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            # Update W
            A_mat = np.zeros((24, pn * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, c * 20 + r] = U[a, r] * V[b, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(pn, 20)
            except:
                continue
            
            # Update U
            A_mat = np.zeros((24, nm * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, a * 20 + r] = V[b, r] * W[c, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U = U_flat.reshape(nm, 20)
            except:
                continue
            
            # Update V
            A_mat = np.zeros((24, mp * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, b * 20 + r] = U[a, r] * W[c, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V = V_flat.reshape(mp, 20)
            except:
                continue
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < 1e-8:
                result = check_round(U, V, W)
                if result is not None:
                    return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
    
    # Fallback: return rank 21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
