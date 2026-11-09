"""Find a low-rank exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Convention: W[k*n+i, r]
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if (i,j,k) is a valid multiplication entry, else 0.
    Actually T[i,j,k] = delta_{i,j} * delta_{j,k} in the sense of matrix mult.
    For C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor form: T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes.
    Wait, the tensor is: T[i,j,k] = 1 (it's the identity in the middle dimension pairing)
    Actually the matrix multiplication tensor has entries T[i,j,k] = δ_{j,j} which is always 1
    for valid indices? No. The tensor T ∈ R^{n×m×p} where T[i,j,k] = 1 if we're computing
    C[i,k] += A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    But that's just an all-ones tensor which has rank 1. That can't be right.
    
    Let me reconsider. The matrix multiplication tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    No wait - the tensor decomposition represents the bilinear (actually trilinear) map.
    The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    That's an all-ones tensor of shape (n,m,p) which has rank 1.
    
    That's clearly wrong. Let me re-read the problem.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
    
    The matrix multiplication tensor T ∈ R^{(n*m) x (m*p) x (p*n)} where:
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    
    So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k, and 0 otherwise.
    This is the correct matrix multiplication tensor.
    """
    rank = U.shape[1]
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
    
    n, m, p = n, m, p
    # Target tensor entries
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor
    # For efficiency, compute via einsum or direct
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is the outer product sum
    # Use np.einsum
    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(decomposed, target, atol=1e-9, rtol=1e-9)

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 48
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 24
    lower_bound = max(nm, mp, pn)
    
    rng = np.random.RandomState(42)
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        max_attempts = 200 if rank < 40 else 500
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Use ALS (Alternating Least Squares) to refine
            # For each r, we have a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # We want sum_r U[:,r]⊗V[:,r]⊗W[:,r] ≈ target
            
            # ALS: fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, U[a,:] should minimize ||target[a,:,:] - sum_r U[a,r] V[:,r]⊗W[:,r]||
                # target[a,:,:] is a (mp, pn) matrix
                # The r-th term contributes U[a,r] * (V[:,r] ⊗ W[:,r]) reshaped to (mp, pn)
                # So we need: target[a,:,:] ≈ sum_r U[a,r] * outer(V[:,r], W[:,r]).T reshaped
                # outer(V[:,r], W[:,r]) has shape (mp, pn), so:
                # target[a,:,:] ≈ sum_r U[a,r] * (V[:,r] * W[:,r].T)
                # This is a linear system: for each a, target[a] ≈ U[a,:] @ M_r where M_r[r_idx] = V[:,r_idx]*W[:,r_idx].T
                # Actually: target[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed a: target[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # where (V[:,r] ⊗ W[:,r]) reshaped to (mp,pn) is V[:,r,None]*W[None,:,r]
                # So: target[a] (mp×pn) ≈ sum_r U[a,r] * (V[:,r] * W[:,r].T)
                # Let M be a (mp*pn, rank) matrix where M[b*pn+c, r] = V[b,r]*W[c,r]
                # Then target_2d[a] (mp*pn,) ≈ U[a,:] @ M.T... no
                # target[a,b,c] = sum_r U[a,r] * M[b*pn+c, r]
                # So target_2d[a] (mp*pn,) = U[a,:] @ M.T where M is (mp*pn, rank)
                # U[a,:] = target_2d[a] @ M.T @ (M @ M.T)^{-1} ... least squares
                # U[a,:] = (M.T @ M)^{-1} M.T @ target_2d[a]
                
                # Build M: M[b*pn+c, r] = V[b,r] * W[c,r]
                # M = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, rank)
                M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
                # M shape: (mp*pn, rank)
                # For each a: U[a,:] solves M.T @ U[a,:] ≈ target[a,:,:].flatten()
                # U[a,:] = (M @ M.T)^{-1} M @ target[a]... 
                # Actually: minimize ||U[a,:] @ M.T - target_2d[a]||
                # => U[a,:] = target_2d[a] @ M @ (M.T @ M)^{-1}
                # or equivalently solve (M.T @ M) U[a,:] = M.T @ target_2d[a]
                
                target_2d = target.reshape(nm, mp * pn)
                # Solve for all a at once: U = (M.T @ M)^{-1} M.T @ target_2d.T
                # U shape should be (nm, rank)
                # target_2d.T has shape (mp*pn, nm)
                # M.T @ target_2d.T has shape (rank, nm)
                # (M.T @ M) has shape (rank, rank)
                try:
                    MMt = M.T @ M  # (rank, rank)
                    M_target = M.T @ target_2d.T  # (rank, nm)
                    U_new = np.linalg.solve(MMt, M_target).T  # (nm, rank)
                    U = U_new
                except np.linalg.LinAlgError:
                    try:
                        U_new = np.linalg.lstsq(M, target_2d.T, rcond=None)[0].T
                        U = U_new
                    except:
                        break
                
                # Update V: for each b, V[b,:] solves similar system
                # target[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For fixed b: target[:,b,:] (nm, pn) = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
                # N[a*pn+c, r] = U[a,r] * W[c,r]
                N = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, rank)
                target_2d_v = target.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)
                try:
                    NNt = N.T @ N
                    N_target = N.T @ target_2d_v.T
                    V_new = np.linalg.solve(NNt, N_target).T
                    V = V_new
                except:
                    try:
                        V_new = np.linalg.lstsq(N, target_2d_v.T, rcond=None)[0].T
                        V = V_new
                    except:
                        break
                
                # Update W: for each c, W[c,:] 
                # target[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
                # For fixed c: target[:,:,c] (nm, mp) = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
                P = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, rank)
                target_2d_w = target.transpose(2, 0, 1).reshape(pn, nm * mp)  # (pn, nm*mp)
                try:
                    PPt = P.T @ P
                    P_target = P.T @ target_2d_w.T
                    W_new = np.linalg.solve(PPt, P_target).T
                    W = W_new
                except:
                    try:
                        W_new = np.linalg.lstsq(P, target_2d_w.T, rcond=None)[0].T
                        W = W_new
                    except:
                        break
                
                # Check convergence
                if it % 10 == 9:
                    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(decomposed - target) / np.linalg.norm(target)
                    if err < 1e-10:
                        break
            
            # Final check
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, target, atol=1e-8, rtol=1e-8):
                # Round to clean values
                U_round = np.round(U, 6)
                V_round = np.round(V, 6)
                W_round = np.round(W, 6)
                # Verify rounded version
                decomp_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.allclose(decomp_round, target, atol=1e-6, rtol=1e-6):
                    best_rank = rank
                    best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    found = True
                    break
                else:
                    # Use unrounded
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            break
        if time.time() > deadline:
            break
    
    return best_result
