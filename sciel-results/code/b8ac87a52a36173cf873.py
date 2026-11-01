"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - (_time.time() - _START)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check decomposition against tensor T."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    return np.max(np.abs(result - T)) <= tol

def _strassen_222():
    """Rank-7 Strassen decomposition for 2x2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _block_decomposition(n, m, p):
    """Block decomposition: pairs of k using Strassen, remainder using rank-3."""
    U_s, V_s, W_s = _strassen_222()
    num_pairs = p // 2
    has_odd = (p % 2 == 1)
    
    # Rank-3 for 2x2x1
    U_r = np.array([[1,0,0],[0,0,1],[0,1,0],[0,0,1]], dtype=float)
    V_r = np.array([[1,0,1],[0,1,1]], dtype=float)
    W_r = np.array([[1,0,1],[0,1,1]], dtype=float)
    
    rank_s = U_s.shape[1]
    rank_r = U_r.shape[1]
    total_rank = num_pairs * rank_s + (rank_r if has_odd else 0)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for pair_idx in range(num_pairs):
        k0 = pair_idx * 2
        U[:, col:col+rank_s] = U_s
        for j in range(m):
            for kl in range(2):
                V[j*p + k0+kl, col:col+rank_s] = V_s[j*2+kl]
        for kl in range(2):
            for i in range(n):
                W[(k0+kl)*n+i, col:col+rank_s] = W_s[kl*2+i]
        col += rank_s
    
    if has_odd:
        k0 = num_pairs * 2
        U[:, col:col+rank_r] = U_r
        for j in range(m):
            V[j*p+k0, col:col+rank_r] = V_r[j]
        for i in range(n):
            W[k0*n+i, col:col+rank_r] = W_r[i]
        col += rank_r
    
    return U, V, W

def _try_direct_rank(n, m, p, target_rank, max_iters=500, seed=42):
    """
    Attempt to find a decomposition of exactly target_rank using iterative optimization.
    Uses the structure of the tensor to guide initialization.
    """
    if _remaining() < 10:
        return None
    
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    # Initialize with a perturbed version of the block decomposition
    U0, V0, W0 = _block_decomposition(n, m, p)
    current_rank = U0.shape[1]
    
    if target_rank >= current_rank:
        return None  # No point, we already have a better one
    
    # Try to find target_rank decomposition via iterative method
    # Strategy: start with random initialization and use ALS (alternating least squares)
    
    for attempt in range(3):
        if _remaining() < 15:
            return None
        
        # Random initialization
        U = rng.randn(N, target_rank)
        V = rng.randn(M, target_rank)
        W = rng.randn(P, target_rank)
        # Normalize
        for k in range(target_rank):
            norm_u = np.linalg.norm(U[:,k])
            norm_v = np.linalg.norm(V[:,k])
            norm_w = np.linalg.norm(W[:,k])
            if norm_u > 1e-10:
                U[:,k] /= norm_u
            if norm_v > 1e-10:
                V[:,k] /= norm_v
            if norm_w > 1e-10:
                W[:,k] /= norm_w
        
        best_err = np.inf
        for it in range(max_iters):
            if _remaining() < 10:
                break
            
            # Alternate: fix V,W solve for U; fix U,W solve for V; fix U,V solve for W
            # For each k: U[:,k] should satisfy sum_k U[:,k] outer(V[:,k]) outer(W[:,k]) = T
            # This is a 3-way tensor decomposition - use ALS
            
            # Update U: for each k, U[:,k] = argmin ||T - sum U[:,k] V[:,k] W[:,k]||
            # Given V,W, the contribution to T from term k is U[:,k] V[:,k] W[:,k]
            # So T = sum_k U[:,k] (V[:,k] W[:,k])^T reshaped... 
            # Actually: T[i,j,l] = sum_k U[i,k] V[j,k] W[l,k]
            # For fixed V,W: T[i,j,l] = sum_k U[i,k] (V[j,k]*W[l,k])
            # So for each i: T[i,:,:] = sum_k U[i,k] * (V[:,k] W[:,k])^T (outer product)
            # This is: T[i] = U[i,:] @ (V*W) where (V*W)[k,:] = V[:,k]*W[:,k]
            
            # Build B matrix: B[i,j,l] = sum_k V[j,k]*W[l,k] for the part not involving U
            # Actually, for ALS on U:
            # T[i,j,l] = sum_k U[i,k] * V[j,k] * W[l,k]
            # Let C[j,l] = sum_k V[j,k]*W[l,k] for the "rest"
            # No, we need: for each i, T[i,:,:] = sum_k U[i,k] * (V[:,k] * W[:,k])
            # So define D_k[j,l] = V[j,k]*W[l,k]
            # Then T[i] = sum_k U[i,k] * D_k (as a matrix)
            # This is a linear system: for each i, T[i,j,l] = sum_k U[i,k] * D_k[j,l]
            # Vectorizing: T[i] (flattened) = sum_k U[i,k] * D_k (flattened)
            
            M_flat = M * P  # 900
            D = np.zeros((target_rank, M_flat))
            for k in range(target_rank):
                D[k, :] = (V[:, k] * W[:, k]).flatten()  # Wait, V[j,k]*W[l,k] for all j,l
            
            # Actually V[j,k]*W[l,k] for j in range(M), l in range(P)
            # This is outer product of V[:,k] and W[:,k]
            for k in range(target_rank):
                D[k, :] = np.outer(V[:, k], W[:, k]).flatten()
            
            # T_flat[i, j, l] -> T_flat[i, j*P+l]
            T_flat = T.reshape(N, M_flat)
            
            # Solve: T_flat[i] = U[i,:] @ D  for each i
            # So U[i,:] = T_flat[i] @ pinv(D)
            D_pinv = np.linalg.pinv(D)
            U_new = T_flat @ D_pinv
            
            # Update V similarly
            C = np.zeros((target_rank, N * P))
            for k in range(target_rank):
                C[k, :] = np.outer(U[:, k], W[:, k]).flatten()
            T_flat_v = T.reshape(N * M, P).reshape(N * M, P)
            # T[i,j,l] = sum_k V[j,k] * U[i,k] * W[l,k]
            # For fixed U,W: T[i,j,l] = sum_k V[j,k] * (U[i,k]*W[l,k])
            # So for each j: T[:,j,:] = sum_k V[j,k] * (U[:,k]*W[:,k])
            # T[:,j] (flattened over i,l) = V[j,:] @ C
            T_flat_j = T.reshape(M, N * P)  # T[j, i, l] = T[i,j,l] permuted
            # Actually T[i,j,l], so for fixed j, we have T[:,j,:] which is (N,P)
            T_j = np.zeros((M, N*P))
            for j in range(M):
                T_j[j, :] = T[:, j, :].flatten()
            C_pinv = np.linalg.pinv(C)
            V_new = T_j @ C_pinv
            
            # Update W
            E = np.zeros((target_rank, N * M))
            for k in range(target_rank):
                E[k, :] = np.outer(U[:, k], V[:, k]).flatten()
            T_l = np.zeros((P, N*M))
            for l in range(P):
                T_l[l, :] = T[:, :, l].flatten()
            E_pinv = np.linalg.pinv(E)
            W_new = T_l @ E_pinv
            
            # Check convergence
            err = np.max(np.abs(T - np.einsum('ik,jk,lk->ijl', U_new, V_new, W_new)))
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if err < 1e-10:
                break
            if _remaining() < 10:
                break
        
        if best_err < 1e-9:
            return U_best, V_best, W_best
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    if _remaining() < 5:
        # Emergency fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T = _build_tensor(n, m, p)
    
    # Get block decomposition as baseline
    U_blk, V_blk, W_blk = _block_decomposition(n, m, p)
    baseline_rank = U_blk.shape[1]
    
    if not _verify(U_blk, V_blk, W_blk, T):
        # Should not happen, but fallback
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U_blk, V_blk, W_blk
    best_rank = baseline_rank
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is at least max(n*m, m*p, p*n) / something
    # For 2x2x15: N=4, M=30, P=30
    # Lower bound on rank: at least max(N, M, P) in some cases, but for matrix mult it's different
    
    # Try ranks from baseline-1 down to a reasonable lower bound
    lower_bound = max(n*m, m*p, p*n)  # 30
    # Actually for matrix multiplication, the rank is at least n*m*p / min(n,m,p) or similar
    # Let's try a range
    
    search_start = max(lower_bound, 20)
    search_end = min(baseline_rank - 1, 60)
    
    # Try a few target ranks
    for target_rank in range(search_start, search_end + 1):
        if _remaining() < 20:
            break
        
        result = _try_direct_rank(n, m, p, target_rank, max_iters=200, seed=target_rank)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, T, tol=1e-9):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                # Found a valid lower rank, stop
                break
    
    # Round to nearest half-integer if close
    for mat in [best_U, best_V, best_W]:
        rounded = np.round(mat * 2) / 2
        if _verify(rounded, best_V if mat is best_U else (best_W if mat is best_V else best_U), 
                   best_W if mat is best_U else (best_V if mat is best_V else best_U), T, tol=1e-6):
            pass  # Could try to use rounded, but keep original for safety
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
