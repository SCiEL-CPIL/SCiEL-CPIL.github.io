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
        Uses efficient rank-1 tensor verification via matrix unfoldings.
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
        
        # Precompute all 21 rank-1 tensors for quick residual computation
        rank1_tensors = []
        for r in range(rank21):
            T_r = np.einsum('u,v,w->uvw', U_base[:, r], V_base[:, r], W_base[:, r])
            rank1_tensors.append(T_r)
        
        # Total sum of all rank-1 tensors
        total_sum = np.zeros((U_dim, V_dim, W_dim))
        for T_r in rank1_tensors:
            total_sum += T_r
        
        for t_remove in range(rank21):
            if time.time() > deadline:
                break
                
            # Residual = target - (total_sum - rank1_tensors[t_remove])
            residual = target - (total_sum - rank1_tensors[t_remove])
            
            # Quick norm check
            if np.linalg.norm(residual) < 1e-12:
                # Removing this term already gives exact decomposition (rank 20)
                U_new = np.delete(U_base, t_remove, axis=1)
                V_new = np.delete(V_base, t_remove, axis=1)
                W_new = np.delete(W_base, t_remove, axis=1)
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                T_snap = np.zeros((U_dim, V_dim, W_dim))
                for r in range(rank21 - 1):
                    T_snap += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                if np.linalg.norm(T_snap - target) < 1e-6:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                continue
            
            # Check if residual is rank-1 using matrix unfoldings
            # Mode-0 unfolding: (U_dim) x (V_dim * W_dim)
            R0 = residual.reshape(U_dim, -1)
            s0 = np.linalg.svd(R0, compute_uv=False)
            if len(s0) > 1 and s0[1] > 1e-8 * s0[0]:
                continue
            
            # Mode-1 unfolding: (V_dim) x (U_dim * W_dim)
            R1 = residual.transpose(1, 0, 2).reshape(V_dim, -1)
            s1 = np.linalg.svd(R1, compute_uv=False)
            if len(s1) > 1 and s1[1] > 1e-8 * s1[0]:
                continue
            
            # Mode-2 unfolding: (W_dim) x (U_dim * V_dim)
            R2 = residual.transpose(2, 0, 1).reshape(W_dim, -1)
            s2 = np.linalg.svd(R2, compute_uv=False)
            if len(s2) > 1 and s2[1] > 1e-8 * s2[0]:
                continue
            
            # Residual is rank-1. Extract vectors.
            # From mode-0 SVD: R0 = u_vec @ (v_w_vec)^T
            U_svd, s, Vt = np.linalg.svd(R0, full_matrices=False)
            u_vec = U_svd[:, 0] * np.sqrt(s[0])
            
            # v_w_vec = v_vec * w_vec (element-wise), length V_dim * W_dim
            # R0[u, idx] = u_vec[u] * v_w_vec[idx]
            u_max = np.argmax(np.abs(u_vec))
            if abs(u_vec[u_max]) < 1e-10:
                continue
            v_w_vec = R0[u_max, :] / u_vec[u_max]
            
            # Factor v_w_vec into v_vec (V_dim) and w_vec (W_dim)
            # v_w_vec[v * W_dim + w] = v_vec[v] * w_vec[w]
            # This is equivalent to: the matrix M = v_w_vec.reshape(V_dim, W_dim) has rank 1
            M = v_w_vec.reshape(V_dim, W_dim)
            sM = np.linalg.svd(M, compute_uv=False)
            if len(sM) > 1 and sM[1] > 1e-8 * sM[0]:
                continue
            
            # Extract v_vec and w_vec from M
            UM, sM_full, VM = np.linalg.svd(M, full_matrices=False)
            v_vec = UM[:, 0] * np.sqrt(sM_full[0])
            w_vec = VM[0, :] * np.sqrt(sM_full[0])
            
            # Verify factorization
            if np.linalg.norm(np.outer(v_vec, w_vec) - M) > 1e-6:
                continue
            
            # Build new decomposition
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
            
            U_new[:, rank21-2] = u_vec
            V_new[:, rank21-2] = v_vec
            W_new[:, rank21-2] = w_vec
            
            # Verify
            T_check = np.zeros((U_dim, V_dim, W_dim))
            for r in range(rank21 - 1):
                T_check += np.einsum('u,v,w->uvw', U_new[:, r], V_new[:, r], W_new[:, r])
            
            if np.linalg.norm(T_check - target) < 1e-6:
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                T_snap = np.zeros((U_dim, V_dim, W_dim))
                for r in range(rank21 - 1):
                    T_snap += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                if np.linalg.norm(T_snap - target) < 1e-6:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        return None
    
    # First try rank 20
    result_20 = try_rank_20_by_elimination()
    if result_20 is not None:
        return result_20
    
    # Try rank 19 by eliminating two terms simultaneously
    def try_rank_19_by_double_elimination():
        """
        Try to find rank-19 by eliminating two of the 21 Strassen terms.
        Check if the sum of two removed terms is rank-1.
        """
        if time.time() > deadline:
            return None
        
        base = strassen_3blocks()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)
        
        rank21 = 21
        U_dim = n * m
        V_dim = m * p
        W_dim = p * n
        
        target = np.zeros((U_dim, V_dim, W_dim))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        rank1_tensors = []
        for r in range(rank21):
            T_r = np.einsum('u,v,w->uvw', U_base[:, r], V_base[:, r], W_base[:, r])
            rank1_tensors.append(T_r)
        
        total_sum = sum(rank1_tensors)
        
        # Try all pairs
        for t1 in range(rank21):
            if time.time() > deadline:
                return None
            for t2 in range(t1 + 1, rank21):
                if time.time() > deadline:
                    return None
                
                # Residual = target - (total - T_t1 - T_t2)
                residual = target - (total_sum - rank1_tensors[t1] - rank1_tensors[t2])
                
                if np.linalg.norm(residual) < 1e-12:
                    # Both terms are redundant
                    keep = [r for r in range(rank21) if r != t1 and r != t2]
                    U_new = U_base[:, keep]
                    V_new = V_base[:, keep]
                    W_new = W_base[:, keep]
                    U_snap = np.round(U_new * 2) / 2
                    V_snap = np.round(V_new * 2) / 2
                    W_snap = np.round(W_new * 2) / 2
                    T_snap = np.zeros((U_dim, V_dim, W_dim))
                    for r in range(len(keep)):
                        T_snap += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    if np.linalg.norm(T_snap - target) < 1e-6:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    continue
                
                # Check if residual is rank-1
                R0 = residual.reshape(U_dim, -1)
                s0 = np.linalg.svd(R0, compute_uv=False)
                if len(s0) > 1 and s0[1] > 1e-8 * s0[0]:
                    continue
                R1 = residual.transpose(1, 0, 2).reshape(V_dim, -1)
                s1 = np.linalg.svd(R1, compute_uv=False)
                if len(s1) > 1 and s1[1] > 1e-8 * s1[0]:
                    continue
                R2 = residual.transpose(2, 0, 1).reshape(W_dim, -1)
                s2 = np.linalg.svd(R2, compute_uv=False)
                if len(s2) > 1 and s2[1] > 1e-8 * s2[0]:
                    continue
                
                # Extract vectors
                U_svd, s, Vt = np.linalg.svd(R0, full_matrices=False)
                u_vec = U_svd[:, 0] * np.sqrt(s[0])
                u_max = np.argmax(np.abs(u_vec))
                if abs(u_vec[u_max]) < 1e-10:
                    continue
                v_w_vec = R0[u_max, :] / u_vec[u_max]
                M = v_w_vec.reshape(V_dim, W_dim)
                sM = np.linalg.svd(M, compute_uv=False)
                if len(sM) > 1 and sM[1] > 1e-8 * sM[0]:
                    continue
                
                UM, sM_full, VM = np.linalg.svd(M, full_matrices=False)
                v_vec = UM[:, 0] * np.sqrt(sM_full[0])
                w_vec = VM[0, :] * np.sqrt(sM_full[0])
                
                if np.linalg.norm(np.outer(v_vec, w_vec) - M) > 1e-6:
                    continue
                
                # Build new decomposition
                keep = [r for r in range(rank21) if r != t1 and r != t2]
                U_new = np.zeros((U_dim, len(keep) + 1))
                V_new = np.zeros((V_dim, len(keep) + 1))
                W_new = np.zeros((W_dim, len(keep) + 1))
                
                for idx, r in enumerate(keep):
                    U_new[:, idx] = U_base[:, r]
                    V_new[:, idx] = V_base[:, r]
                    W_new[:, idx] = W_base[:, r]
                
                U_new[:, len(keep)] = u_vec
                V_new[:, len(keep)] = v_vec
                W_new[:, len(keep)] = w_vec
                
                T_check = np.zeros((U_dim, V_dim, W_dim))
                for r in range(len(keep) + 1):
                    T_check += np.einsum('u,v,w->uvw', U_new[:, r], V_new[:, r], W_new[:, r])
                
                if np.linalg.norm(T_check - target) < 1e-6:
                    U_snap = np.round(U_new * 2) / 2
                    V_snap = np.round(V_new * 2) / 2
                    W_snap = np.round(W_new * 2) / 2
                    T_snap = np.zeros((U_dim, V_dim, W_dim))
                    for r in range(len(keep) + 1):
                        T_snap += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    if np.linalg.norm(T_snap - target) < 1e-6:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        return None
    
    # Try rank 19
    result_19 = try_rank_19_by_double_elimination()
    if result_19 is not None:
        return result_19
    
    # Fallback to rank 21
    return strassen_3blocks()

# EVOLVE-BLOCK-END
