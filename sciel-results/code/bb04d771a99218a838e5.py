import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def to_half_integer(x):
        """Round to nearest half-integer"""
        return np.round(x * 2) / 2.0
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact"""
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(result, T, atol=tol)
    
    def try_structured_rank(rank):
        """Try to find a valid decomposition with given rank using structured W"""
        if time.time() > DEADLINE:
            return None
        
        # Strategy: fix W to be sparse with specific structure, solve for U, V
        # W has shape (pn, rank)
        # For each c (output index), W[c,:] should have few nonzeros
        
        # Try: W is a permutation matrix extended to rank columns
        # i.e., each row of W has exactly one 1, and each column has at most one 1
        # This means rank >= pn = 24
        
        if rank < pn:
            return None
        
        # Try several random sparse W patterns
        for attempt in range(200):
            if time.time() > DEADLINE:
                break
            
            # Initialize W with sparse structure
            W = np.zeros((pn, rank))
            
            # Each row gets one or a few 1s
            for c in range(pn):
                # Choose 1-2 columns for this row
                n_nonzero = min(2, rank)
                cols = random.sample(range(rank), n_nonzero)
                for col in cols:
                    W[c, col] = 1.0
            
            # Now solve for U and V
            # The equation is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For each c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is bilinear, so we fix one and solve for the other iteratively
            
            # Initialize U and V randomly
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() > DEADLINE:
                    break
                
                # Solve for U given V and W
                # For each c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed b,c: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This gives a linear system for each (b,c) pair
                # But it's over-determined in general
                
                # Better: for each r where W[c,r] != 0 and V[b,r] != 0:
                # U[a,r] contributes to T[a,b,c]
                
                # Let's use a different approach: for each column r, 
                # the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # We want sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
                
                # Fix V and W, solve for U:
                # For each (b,c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # Let M_bc[r] = V[b,r] * W[c,r]
                # Then: U @ M_bc = T[:,b,c] for each (b,c)
                # This is: sum_r U[a,r] * M_bc[r] = T[a,b,c]
                # Which is: (U * M_bc[None,:]).sum(axis=1) = T[:,b,c]
                
                # This is a linear system: U @ M_bc^T = T[:,b,c]
                # Stack all (b,c): U @ (M_bc for all bc)^T = T.reshape(nm, -1)
                
                # Build the system
                # For each (b,c), M_bc = V[b,:] * W[c,:] (element-wise, shape rank)
                # System: U @ M_bc = T[:,b,c]
                # Stack: U @ M_all = T_all where M_all has shape (rank, mp*pn)
                
                M_all = np.zeros((rank, mp * pn))
                idx = 0
                for b in range(mp):
                    for c in range(pn):
                        M_all[:, idx] = V[b, :] * W[c, :]
                        idx += 1
                
                T_all = T.reshape(nm, mp * pn)
                
                # Solve U @ M_all = T_all using least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_all.T, T_all.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Snap to half-integers
                U = to_half_integer(U)
                
                # Similarly solve for V
                # For each (a,c): sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c]
                # M_ac[r] = U[a,r] * W[c,r]
                # V @ M_ac = T[a,:,c]
                
                M_all_v = np.zeros((rank, nm * pn))
                idx = 0
                for a in range(nm):
                    for c in range(pn):
                        M_all_v[:, idx] = U[a, :] * W[c, :]
                        idx += 1
                
                T_all_v = T.reshape(nm * pn, mp)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_all_v.T, T_all_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                V = to_half_integer(V)
                
                # Check if we're close
                if verify(U, V, W):
                    return U, V, W, rank
        
        return None
    
    def try_fixed_w_identity(rank):
        """Try with W as identity-like (each row has one 1)"""
        if rank < pn:
            return None
        
        # W is a permutation: W[c, perm[c]] = 1
        for attempt in range(100):
            if time.time() > DEADLINE:
                break
            
            perm = list(range(rank))
            random.shuffle(perm)
            W = np.zeros((pn, rank))
            for c in range(pn):
                W[c, perm[c]] = 1.0
            
            # Now T[:,:,c] must equal U[:,perm[c]] ⊗ V[:,perm[c]]
            # For each c: T[:,:,c] is a matrix of shape (nm, mp)
            # It must be rank 1: U[:,perm[c]] * V[:,perm[c]].T
            
            # Check if T[:,:,c] is rank 1 for all c
            valid = True
            for c in range(pn):
                slice_c = T[:, :, c]
                # Check rank
                sv = np.linalg.svd(slice_c, compute_uv=False)
                if sv[0] > 0 and (len(sv) > 1 and sv[1] > 1e-10):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Extract U and V from the rank-1 factorizations
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            
            for c in range(pn):
                slice_c = T[:, :, c]
                if np.allclose(slice_c, 0):
                    continue
                # slice_c = u_vec * v_vec.T
                # u_vec = U[:,perm[c]], v_vec = V[:,perm[c]]
                # Find the factorization
                sv = np.linalg.svd(slice_c, compute_uv=False)
                if sv[0] < 1e-10:
                    continue
                # u = slice_c @ v / (v.T @ v) for some v
                # Use the top singular vectors
                U_s, S, Vt = np.linalg.svd(slice_c, full_matrices=False)
                if S[0] < 1e-10:
                    continue
                u_vec = U_s[:, 0] * np.sqrt(S[0])
                v_vec = Vt[0, :] * np.sqrt(S[0])
                
                # But we need u_vec * v_vec.T = slice_c
                # Check
                if not np.allclose(np.outer(u_vec, v_vec), slice_c, atol=1e-10):
                    continue
                
                U[:, perm[c]] = u_vec
                V[:, perm[c]] = v_vec
            
            # Snap to half-integers
            U = to_half_integer(U)
            V = to_half_integer(V)
            
            if verify(U, V, W):
                return U, V, W, rank
        
        return None
    
    # Try to find lower rank decompositions
    for target_rank in range(max(24, rank_sb - 1), rank_sb):
        if time.time() > DEADLINE:
            break
        
        # Try fixed W identity approach
        result = try_fixed_w_identity(target_rank)
        if result:
            best_U, best_V, best_W, best_rank = result
            break
        
        # Try structured sparse W
        if target_rank >= pn:
            result = try_structured_rank(target_rank)
            if result:
                best_U, best_V, best_W, best_rank = result
                break
    
    # Convert to list format
    U_out = [[U_out_val for U_out_val in row] for row in best_U.tolist()]
    V_out = [[V_out_val for V_out_val in row] for row in best_V.tolist()]
    W_out = [[W_out_val for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
