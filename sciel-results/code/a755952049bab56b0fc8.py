"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (12, rank)  
    # W: (p*n, rank) = (12, rank)
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_3blocks():
        """Rank 21 decomposition using Strassen's algorithm for 3 blocks."""
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for A (U matrix)
        # Indices: 0: A00, 1: A01, 2: A10, 3: A11
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # M1 -> C00, C11
            [(1, 0, 1), (1, 1, -1)],  # M2 -> C10, C11
            [(0, 1, 1), (1, 1, 1)],   # M3 -> C01, C11
            [(0, 0, 1), (1, 0, 1)],   # M4 -> C00, C10
            [(0, 0, -1), (0, 1, 1)],  # M5 -> C00, C01
            [(1, 1, 1)],              # M6 -> C11
            [(0, 0, 1)]               # M7 -> C00
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    def try_rank_20_by_elimination():
        """
        Try to find rank-20 by eliminating one of the 21 Strassen terms.
        The idea: if we can express one term as a linear combination of others
        in a way that preserves the tensor identity, we reduce rank by 1.
        
        Strategy: For each of the 21 terms, check if removing it and adjusting
        the remaining terms (by adding a correction) can still satisfy the identity.
        This is equivalent to checking if the term is in the span of the others.
        """
        base = strassen_3blocks()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)
        
        rank21 = 21
        U_dim = n * m  # 4
        V_dim = m * p  # 12
        W_dim = p * n  # 12
        
        # Build target tensor
        target = np.zeros((U_dim, V_dim, W_dim))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target[u_idx, v_idx, w_idx] = 1.0
        
        # For each term t in 0..20, check if it's linearly dependent
        # on the others by seeing if the residual after removing it
        # can be absorbed by the remaining terms.
        
        for t_remove in range(rank21):
            if time.time() > deadline:
                break
                
            # Remove term t_remove
            U_rem = np.delete(U_base, t_remove, axis=1)
            V_rem = np.delete(V_base, t_remove, axis=1)
            W_rem = np.delete(W_base, t_remove, axis=1)
            
            # Compute residual: target - sum of remaining terms
            T_rec = np.zeros((U_dim, V_dim, W_dim))
            for r in range(rank21 - 1):
                T_rec += np.einsum('u,v,w->uvw', U_rem[:, r], V_rem[:, r], W_rem[:, r])
            
            residual = target - T_rec
            
            # Check if residual can be expressed as a rank-1 tensor
            # (which would mean we can add it back with a modified term)
            # Actually, we need to check if residual is in the span of
            # rank-1 tensors formed by the remaining U, V, W vectors.
            
            # More precisely: we need to find u_vec, v_vec, w_vec such that
            # outer(u_vec, v_vec, w_vec) = residual
            # This means residual must be a rank-1 tensor.
            
            # Check if residual is rank-1
            # Reshape to matrix and check rank
            # For a 3-tensor to be rank-1, all its matrix unfoldings must be rank-1
            
            # Matrix unfolding along mode 0: (U_dim) x (V_dim * W_dim)
            R0 = residual.reshape(U_dim, V_dim * W_dim)
            rank0 = np.linalg.matrix_rank(R0)
            
            if rank0 > 1:
                continue  # Not rank-1, can't absorb
            
            # Check other unfoldings
            R1 = residual.reshape(V_dim, U_dim * W_dim)
            rank1 = np.linalg.matrix_rank(R1)
            if rank1 > 1:
                continue
            
            R2 = residual.reshape(W_dim, U_dim * V_dim)
            rank2 = np.linalg.matrix_rank(R2)
            if rank2 > 1:
                continue
            
            # Residual appears to be rank-1. Extract the vectors.
            # From R0, the columns are proportional to u_vec
            if np.linalg.norm(R0) < 1e-10:
                continue
                
            # Use SVD to extract
            U_svd, s, Vt = np.linalg.svd(R0, full_matrices=False)
            u_vec = U_svd[:, 0] * np.sqrt(s[0])
            
            # Verify: residual should be u_vec outer v_vec outer w_vec
            # R0 = u_vec (v_vec outer w_vec)^T
            # So v_vec outer w_vec = R0 / u_vec[:, 0] (if u_vec[0] != 0)
            # Better: use the full reconstruction
            # residual[u,v,w] = u_vec[u] * v_vec[v] * w_vec[w]
            # R0[u, v*W_dim+w] = u_vec[u] * (v_vec[v] * w_vec[w])
            
            # Find v_vec and w_vec from R0
            # For each u, R0[u, :] = u_vec[u] * (v_vec * w_vec)
            # So (v_vec * w_vec) = R0[u, :] / u_vec[u] for any u with u_vec[u] != 0
            
            # Find a u with significant value
            u_max = np.argmax(np.abs(u_vec))
            if abs(u_vec[u_max]) < 1e-10:
                continue
                
            vw_vec = R0[u_max, :] / u_vec[u_max]
            
            # Now vw_vec = v_vec * w_vec (element-wise)
            # We need to factor this into v_vec (length V_dim) and w_vec (length W_dim)
            # vw_vec has length V_dim * W_dim
            # This is an underdetermined system unless we have constraints.
            
            # Actually, for the decomposition to work, we need:
            # sum_r U[u,r] V[v,r] W[w,r] = target[u,v,w]
            # The new term would be: u_vec[u] * v_vec[v] * w_vec[w]
            # where v_vec[v] * w_vec[w] = vw_vec[v*W_dim+w]
            
            # We can choose v_vec and w_vec freely as long as their element-wise
            # product gives vw_vec. The simplest choice:
            # v_vec[v] = sqrt(vw_vec[v*W_dim+0]) for w=0, and w_vec[w] = vw_vec[v*W_dim+w] / v_vec[v]
            
            # But this may not work if vw_vec has zeros.
            
            # Alternative: just use the rank-1 decomposition directly.
            # The new term contributes: u_vec[u] * (v_vec[v] * w_vec[w])
            # We can set v_vec[v] = 1 for all v, and w_vec[w] = vw_vec[v*W_dim+w]
            # But w_vec must be the same for all v, so this only works if
            # vw_vec[v*W_dim+w] doesn't depend on v, which it generally does.
            
            # Correct approach: the rank-1 tensor is uniquely determined (up to scaling).
            # We need v_vec and w_vec such that v_vec[v]*w_vec[w] = vw_vec[v*W_dim+w]
            # for all v, w. This requires that vw_vec[v*W_dim+w] = f(v)*g(w)
            # for some functions f and g.
            
            # Check this condition:
            # For all v1, v2, w1, w2: vw_vec[v1*W_dim+w1] * vw_vec[v2*W_dim+w2] 
            #   = vw_vec[v1*W_dim+w2] * vw_vec[v2*W_dim+w1]
            
            valid_rank1 = True
            for v1 in range(V_dim):
                for v2 in range(V_dim):
                    for w1 in range(W_dim):
                        for w2 in range(W_dim):
                            lhs = vw_vec[v1*W_dim+w1] * vw_vec[v2*W_dim+w2]
                            rhs = vw_vec[v1*W_dim+w2] * vw_vec[v2*W_dim+w1]
                            if abs(lhs - rhs) > 1e-8 * max(1, abs(lhs), abs(rhs)):
                                valid_rank1 = False
                                break
                        if not valid_rank1:
                            break
                    if not valid_rank1:
                        break
                if not valid_rank1:
                    break
            
            if not valid_rank1:
                continue
            
            # Factor vw_vec into v_vec and w_vec
            # v_vec[v] * w_vec[w] = vw_vec[v*W_dim+w]
            # Choose v_vec[0] = 1, then w_vec[w] = vw_vec[0*W_dim+w]
            # Then v_vec[v] = vw_vec[v*W_dim+0] / w_vec[0] (if w_vec[0] != 0)
            
            if abs(vw_vec[0]) < 1e-10:
                # Try different starting point
                found = False
                for v0 in range(V_dim):
                    for w0 in range(W_dim):
                        if abs(vw_vec[v0*W_dim+w0]) > 1e-10:
                            v_vec = np.zeros(V_dim)
                            w_vec = np.zeros(W_dim)
                            v_vec[v0] = np.sqrt(abs(vw_vec[v0*W_dim+w0]))
                            if vw_vec[v0*W_dim+w0] < 0:
                                v_vec[v0] = -v_vec[v0]
                            w_vec[w0] = vw_vec[v0*W_dim+w0] / v_vec[v0]
                            
                            # Fill in rest
                            for v in range(V_dim):
                                for w in range(W_dim):
                                    expected = v_vec[v] * w_vec[w]
                                    actual = vw_vec[v*W_dim+w]
                                    if abs(v_vec[v]) > 1e-10:
                                        if abs(w_vec[w]) > 1e-10:
                                            if abs(expected - actual) > 1e-8:
                                                v_vec = None
                                                break
                                    elif abs(actual) > 1e-8:
                                        v_vec = None
                                        break
                                if v_vec is None:
                                    break
                            if v_vec is not None:
                                found = True
                                break
                    if found:
                        break
                if not found:
                    continue
            else:
                v_vec = np.zeros(V_dim)
                w_vec = np.zeros(W_dim)
                v_vec[0] = 1.0
                for w in range(W_dim):
                    w_vec[w] = vw_vec[0*W_dim+w]
                for v in range(1, V_dim):
                    if abs(w_vec[0]) > 1e-10:
                        v_vec[v] = vw_vec[v*W_dim+0] / w_vec[0]
                    else:
                        # Find a w where w_vec[w] != 0
                        found_w = False
                        for w in range(W_dim):
                            if abs(w_vec[w]) > 1e-10:
                                v_vec[v] = vw_vec[v*W_dim+w] / w_vec[w]
                                found_w = True
                                break
                        if not found_w:
                            v_vec = None
                            break
                if v_vec is None:
                    continue
            
            # Verify the factorization
            for v in range(V_dim):
                for w in range(W_dim):
                    if abs(v_vec[v] * w_vec[w] - vw_vec[v*W_dim+w]) > 1e-8:
                        v_vec = None
                        break
                if v_vec is None:
                    break
            
            if v_vec is None:
                continue
            
            # Now we have a valid rank-1 correction.
            # The new rank-20 decomposition is:
            # Original terms 0..t_remove-1, t_remove+1..20, plus the new term.
            
            U_new = np.zeros((U_dim, rank21 - 1))
            V_new = np.zeros((V_dim, rank21 - 1))
            W_new = np.zeros((W_dim, rank21 - 1))
            
            for r in range(rank21 - 1):
                if r < t_remove:
                    U_new[:, r] = U_base[:, r]
                    V_new[:, r] = V_base[:, r]
                    W_new[:, r] = W_base[:, r]
                else:
                    U_new[:, r] = U_base[:, r+1]
                    V_new[:, r] = V_base[:, r+1]
                    W_new[:, r] = W_base[:, r+1]
            
            # Add the new term
            U_new[:, rank21-2] = u_vec
            V_new[:, rank21-2] = v_vec
            W_new[:, rank21-2] = w_vec
            
            # Verify
            T_check = np.zeros((U_dim, V_dim, W_dim))
            for r in range(rank21 - 1):
                T_check += np.einsum('u,v,w->uvw', U_new[:, r], V_new[:, r], W_new[:, r])
            
            if np.linalg.norm(T_check - target) < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                
                # Verify after snapping
                T_snap = np.zeros((U_dim, V_dim, W_dim))
                for r in range(rank21 - 1):
                    T_snap += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                
                if np.linalg.norm(T_snap - target) < 1e-6:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
        return None
    
    # First try rank 20
    result_20 = try_rank_20_by_elimination()
    if result_20 is not None:
        return result_20
    
    # Fallback to rank 21
    return strassen_3blocks()

# EVOLVE-BLOCK-END
