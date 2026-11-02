"""Tensor decomposition search for 2x3x10 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _build_tensor(n, m, p):
    """Build the n x m x p matrix multiplication tensor (flattened)."""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _verify_decomposition_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using the tensor structure directly."""
    rank = len(U[0])
    # Check each term of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The (i*m+j, j*p+k, k*n+i) entry should be 1
                # All other entries should be 0
                pass  # We'll do full reconstruction
    
    # Full reconstruction check
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    T_recon[a, b, c] += U[a][r] * V[b][r] * W[c][r]
    
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T_recon[i*m+j, j*p+k, k*n+i]
                if abs(actual - expected) > tol:
                    return False
    
    # Check that all other entries are approximately 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i2 in range(n):
                        # Skip the diagonal entry we already checked
                        if (i*m+j, j*p+k, k*n+i) == (i*m+j, j*p+k2, k2*n+i2):
                            continue
                        if abs(T_recon[i*m+j, j*p+k, k2*n+i2]) > tol:
                            return False
    return True

def _rank_of_matrix(M):
    """Compute numerical rank of a matrix."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        return np.sum(s > 1e-10)
    except:
        return M.shape[0]

def _try_als_decomposition(n, m, p, target_rank, max_iters=30, tol=1e-8):
    """Try ALS decomposition for a given target rank."""
    nm, mp, pn = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    
    # Try multiple random initializations
    for seed in range(20):
        np.random.seed(seed)
        
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Normalize columns
        for r in range(target_rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0:
                U[:, r] /= norm_u
            if norm_v > 0:
                V[:, r] /= norm_v
            if norm_w > 0:
                W[:, r] /= norm_w
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U: for each column i of U, solve least squares
            M_vw = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, target_rank)
            T_flat = T.reshape(nm, mp * pn)
            U_new = T_flat @ np.linalg.pinv(M_vw.T)
            
            # Update V
            N_uw = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, target_rank)
            T_flat_v = T.reshape(mp, nm * pn)
            V_new = T_flat_v @ np.linalg.pinv(N_uw.T)
            
            # Update W
            P_uv = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, target_rank)
            T_flat_w = T.reshape(pn, nm * mp)
            W_new = T_flat_w @ np.linalg.pinv(P_uv.T)
            
            # Check convergence
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            error = np.max(np.abs(T_check - T))
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if error < tol:
                break
        
        if best_error < tol:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
            if np.max(np.abs(T_check - T)) < 1e-10:
                return U_rounded, V_rounded, W_rounded, True
    
    return None, None, None, False

def _structured_initialization(n, m, p, rank):
    """Create structured initialization based on the tensor's block structure."""
    nm, mp, pn = n*m, m*p, p*n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Use the natural structure: each (i,j,k) triple corresponds to one term
    # But we want to merge some terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_school = n * m * p  # 60
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook decomposition as fallback
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    deadline = time.time() + 140  # 10s safety margin
    
    # First, compute lower bounds on rank using matrix unfoldings
    # The rank of the tensor is at least the rank of any unfolding
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.reshape(mp, nm * pn)
    T_unfold3 = T.reshape(pn, nm * mp)
    
    rank_bound1 = _rank_of_matrix(T_unfold1)
    rank_bound2 = _rank_of_matrix(T_unfold2)
    rank_bound3 = _rank_of_matrix(T_unfold3)
    
    min_rank_bound = max(rank_bound1, rank_bound2, rank_bound3)
    
    # Try ranks from min_rank_bound up to schoolbook rank
    for r in range(min_rank_bound, rank_school):
        if time.time() > deadline:
            break
        
        # Try structured initialization first
        U_init, V_init, W_init = _structured_initialization(n, m, p, r)
        
        # Run ALS with structured initialization
        U_als = U_init.copy()
        V_als = V_init.copy()
        W_als = W_init.copy()
        
        for it in range(20):
            if time.time() > deadline:
                break
            
            M_vw = np.einsum('jr,kr->jkr', V_als, W_als).reshape(mp * pn, r)
            T_flat = T.reshape(nm, mp * pn)
            U_new = T_flat @ np.linalg.pinv(M_vw.T)
            
            N_uw = np.einsum('ir,kr->ikr', U_new, W_als).reshape(nm * pn, r)
            T_flat_v = T.reshape(mp, nm * pn)
            V_new = T_flat_v @ np.linalg.pinv(N_uw.T)
            
            P_uv = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, r)
            T_flat_w = T.reshape(pn, nm * mp)
            W_new = T_flat_w @ np.linalg.pinv(P_uv.T)
            
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            error = np.max(np.abs(T_check - T))
            
            U_als, V_als, W_als = U_new, V_new, W_new
            
            if error < 1e-8:
                # Try to round
                U_rounded = np.round(U_als * 2) / 2
                V_rounded = np.round(V_als * 2) / 2
                W_rounded = np.round(W_als * 2) / 2
                
                T_check2 = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check2 - T)) < 1e-10:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = r
                    break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
