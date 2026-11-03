"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(6,12,8) = 12
    min_rank = max(n * m, m * p, n * p)
    
    # Precompute the non-zero entries of the target tensor
    # T[(i,j), (j,k), (k,i)] = 1 for each (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (u,v,w), the sum should be 0
    
    # The full tensor has nm*mp*pn = 6*12*8 = 576 entries
    # Only 24 are non-zero (the diagonal-like entries)
    
    # For efficient residual computation, we can use the structure
    # Let's precompute the indices of non-zero entries
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_indices.append((u, v, w))
    
    def compute_residual_vectorized(U, V, W, rank):
        """Compute the residual for all 576 tensor entries efficiently."""
        nm, mp, pn = n*m, m*p, p*n
        
        # Compute C = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # C[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        
        # Use einsum for efficiency
        # C = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # But this creates a full 6x12x8 tensor which is fine
        C = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Target tensor T
        T = np.zeros((nm, mp, pn))
        for u, v, w in nonzero_indices:
            T[u, v, w] = 1.0
        
        return C - T
    
    def snap_to_half_int(arr):
        """Snap to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def verify_exact(U, V, W, rank):
        """Verify that the decomposition is exact with half-integer entries."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-10):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-10):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-10):
            return False
        
        # Check exact decomposition
        C = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        T = np.zeros((n*m, m*p, p*n))
        for u, v, w in nonzero_indices:
            T[u, v, w] = 1.0
        
        return np.allclose(C, T, atol=1e-8)
    
    def try_rank(rank, max_restarts=30):
        """Try to find a valid decomposition of the given rank."""
        nm, mp, pn = n*m, m*p, p*n
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart % 4 == 0:
                # Random initialization
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
            elif restart % 4 == 1:
                # Structured initialization based on schoolbook subset
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                # Add small random perturbation
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            elif restart % 4 == 2:
                # Sparse initialization
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    # Randomly assign non-zero entries
                    u_idx = np.random.randint(0, nm)
                    v_idx = np.random.randint(0, mp)
                    w_idx = np.random.randint(0, pn)
                    U[u_idx, r] = random.choice([-1, 1])
                    V[v_idx, r] = random.choice([-1, 1])
                    W[w_idx, r] = random.choice([-1, 1])
            else:
                # Hybrid initialization
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3
                # Make some entries exactly 0 or 1
                for r in range(rank):
                    if random.random() < 0.3:
                        U[random.randint(0, nm-1), r] = 1
                    if random.random() < 0.3:
                        V[random.randint(0, mp-1), r] = 1
                    if random.random() < 0.3:
                        W[random.randint(0, pn-1), r] = 1
            
            # Alternating least squares optimization
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update each factor using ALS
                for c in range(rank):
                    # Compute residual excluding this component
                    residual = np.zeros((nm, mp, pn))
                    for c2 in range(rank):
                        if c2 != c:
                            residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                    
                    # Update U[:, c]
                    # For each (v,w), we have: sum_u U[u,c] * V[v,c] * W[w,c] = residual[u,v,w]
                    # This is a linear system in U[:,c]
                    # A[v*w, u] = V[v,c] * W[w,c]
                    # b[v*w] = residual[:, v, w].flatten()
                    
                    A_U = np.zeros((mp * pn, nm))
                    b_U = np.zeros(mp * pn)
                    idx = 0
                    for v in range(mp):
                        for w in range(pn):
                            factor = V[v, c] * W[w, c]
                            A_U[idx, :] = factor
                            b_U[idx] = residual[:, v, w].flatten()
                            idx += 1
                    
                    U[:, c], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                    
                    if time.time() > deadline - 10:
                        break
                    
                    # Update V[:, c]
                    residual = np.zeros((nm, mp, pn))
                    for c2 in range(rank):
                        if c2 != c:
                            residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                    
                    A_V = np.zeros((nm * pn, mp))
                    b_V = np.zeros(nm * pn)
                    idx = 0
                    for u in range(nm):
                        for w in range(pn):
                            factor = U[u, c] * W[w, c]
                            A_V[idx, :] = factor
                            b_V[idx] = residual[u, :, w].flatten()
                            idx += 1
                    
                    V[:, c], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                    
                    if time.time() > deadline - 10:
                        break
                    
                    # Update W[:, c]
                    residual = np.zeros((nm, mp, pn))
                    for c2 in range(rank):
                        if c2 != c:
                            residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                    
                    A_W = np.zeros((nm * mp, pn))
                    b_W = np.zeros(nm * mp)
                    idx = 0
                    for u in range(nm):
                        for v in range(mp):
                            factor = U[u, c] * V[v, c]
                            A_W[idx, :] = factor
                            b_W[idx] = residual[u, v, :].flatten()
                            idx += 1
                    
                    W[:, c], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                
                # Check convergence
                C = np.einsum('ur,vr,wr->uvw', U, V, W)
                T = np.zeros((nm, mp, pn))
                for u, v, w in nonzero_indices:
                    T[u, v, w] = 1.0
                
                error = np.linalg.norm(C - T)
                if error < 1e-6:
                    # Try to snap to half-integers
                    U_snap = snap_to_half_int(U)
                    V_snap = snap_to_half_int(V)
                    W_snap = snap_to_half_int(W)
                    
                    if verify_exact(U_snap, V_snap, W_snap, rank):
                        U_list = [[float(val) for val in row] for row in U_snap]
                        V_list = [[float(val) for val in row] for row in V_snap]
                        W_list = [[float(val) for val in row] for row in W_snap]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # If snapping fails, try without snapping (but need half-integers)
                    # Try rounding to integers first
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify_exact(U_int, V_int, W_int, rank):
                        U_list = [[float(val) for val in row] for row in U_int]
                        V_list = [[float(val) for val in row] for row in V_int]
                        W_list = [[float(val) for val in row] for row in W_int]
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, max_restarts=20)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END

