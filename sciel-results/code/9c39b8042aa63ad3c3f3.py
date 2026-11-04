"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
Uses an alternating minimization / structured search approach with a deadline guard.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 120  # seconds, with safety margin

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = delta_{i,k} (in the appropriate sense for matmul)
    Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor has indices (i, j, k) with T[i,j,k] = 1 if... 
    Actually the standard tensor for matmul is:
    T[i, j, k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    Wait, let me think about this more carefully.
    
    The decomposition is: for each column c,
    U[:,c] is a vector in R^{n*m}, V[:,c] in R^{m*p}, W[:,c] in R^{p*n}
    
    The condition is that for all A (n x m) and B (m x p):
    sum_c (U[:,c]^T vec(A)) * (V[:,c]^T vec(B)) = vec(C)^T W[:,c] summed...
    
    Let me use the standard formulation. The matmul tensor T is in R^{n*m} x R^{m*p} x R^{p*n}
    where T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    
    Wait, let me re-read the problem. W[k*n+i, r] follows the trace convention.
    
    The decomposition requires:
    sum_r U[a,r] * V[b,r] = W[c,r]... no.
    
    The tensor decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k, and 0 otherwise.
    But actually it's more subtle - the tensor is defined such that the bilinear map
    (A, B) -> C is recovered.
    
    Let me just verify by checking: for the schoolbook, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    Then sum_r U[a,r] V[b,r] W[c,r] should equal T[a,b,c].
    
    For a = i*m+j, b = j'*p+k', c = k''*n+i'':
    sum_r U[a,r] V[b,r] W[c,r] = sum over all (i,j,k) tuples where:
    U[i*m+j, r] = 1 means r corresponds to (i,j,k) and a = i*m+j (so i,j match)
    V[j'*p+k', r] = 1 means j',k' match the j,k of that r
    W[k''*n+i'', r] = 1 means k'',i'' match the k,i of that r
    
    So the sum is 1 iff there exists (i,j,k) such that i*m+j = a, j*p+k = b, k*n+i = c.
    This means i = i', j = j', k = k' and a=i*m+j, b=j*p+k, c=k*n+i.
    So T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for the same i,j,k.
    
    This is correct for the matmul tensor.
    """
    rank = U.shape[1]
    # Build the tensor T from U, V, W
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is equivalent to: for each r, outer product
    # We can check: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T_target
    
    # More efficient: compute the "flattened" check
    # For each r, we have a rank-1 tensor. The sum should equal T_target.
    
    # T_target is sparse: T_target[i*m+j, j*p+k, k*n+i] = 1
    
    # Check by computing: for each (a, b), sum_r U[a,r] V[b,r] W[c,r] should be 0 for all c except the correct one
    # Actually let's just do: M = U @ V.T  (n*m x m*p)
    # Then for each c, the vector M[:, c]... no.
    
    # Let me think differently. Define:
    # For each r, let u_r = U[:,r] (n*m x 1), v_r = V[:,r] (m*p x 1), w_r = W[:,r] (p*n x 1)
    # The tensor sum is: sum_r u_r ⊗ v_r ⊗ w_r
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # We can check this by: for each c, define W_c = W[:,c]... no, W is (p*n, rank)
    
    # Let's just compute it directly for small sizes
    # n=3, m=4, p=6, so n*m=12, m*p=24, p*n=18
    # rank is at most 72 for schoolbook
    
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # = (U @ V.T @ diag... no)
    
    # Actually: fix a, b. Then T[a,b,c] for all c is:
    # sum_r U[a,r] V[b,r] W[c,r] = (U[a,:] * V[b,:]) @ W
    # where U[a,:] is a row vector of length rank, V[b,:] is a row vector of length rank
    # So T[a,b,:] = (U[a,:] * V[b,:]) @ W
    
    # Let's verify this matches the target
    
    n, m, p = n, m, p
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute T from factors
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # For each a, b: T[a,b,:] = (U[a,:] * V[b,:]) @ W
    # We can vectorize: T = np.einsum('ar,br,c r->abc', U, V, W) but that's memory heavy
    
    # For small sizes, let's just do it
    # n*m=12, m*p=24, p*n=18, rank <= 72
    # T will be 12 x 24 x 18 = 5184 entries, fine
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute T from factors
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Use einsum
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check exactness (with tolerance for floating point)
    diff = np.abs(T_computed - T_target)
    max_diff = np.max(diff)
    
    return max_diff < 1e-10, max_diff


def _check_half_integer(U, V, W):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        # Check if 2*M has integer entries
        if not np.allclose(2 * M, np.round(2 * M), atol=1e-10):
            return False
    return True


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try to reduce rank by removing linearly dependent or zero columns
    # and by searching for lower-rank decompositions
    
    def try_reduce_rank(U_cur, V_cur, W_cur, max_rank, time_limit):
        """Try to find a decomposition of rank <= max_rank."""
        deadline = start_time + time_limit
        
        # Strategy: try to find a lower rank decomposition
        # Use the fact that if we have a rank-r decomposition, we can try to merge columns
        
        # First, check if current decomposition is valid
        valid, max_diff = _verify(U_cur, V_cur, W_cur, n, m, p)
        if not valid:
            return None
        
        # Try to reduce by finding linear dependencies
        # If two columns of U are linearly dependent (and similarly for V and W), we might be able to merge
        
        # Alternative approach: use the structure of the matmul tensor
        # The matmul tensor for (3,4,6) has known lower bounds
        # Let's try a structured search
        
        # Try ranks from lower to upper
        for target_rank in range(1, max_rank + 1):
            if time.time() > deadline - 5:
                break
            
            # Try to find a rank-target_rank decomposition
            # Initialize with random factors
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                U_try = np.random.randn(nm, target_rank)
                V_try = np.random.randn(mp, target_rank)
                W_try = np.random.randn(pn, target_rank)
                
                # Alternating minimization
                for iteration in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    # Update U: for each a, U[a,:] should minimize
                    # sum_b,c T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
                    # This is linear in U[a,:]
                    # Fix V and W, solve for U
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                    # This is a linear system: T[a,:, :] flattened = U[a,:] @ (V⊗W flattened)
                    
                    # More efficiently:
                    # For each a, b: T[a,b,:] = (U[a,:] * V[b,:]) @ W
                    # Let M_ab = T[a,b,:] (pn x 1)
                    # We want: M_ab = (U[a,:] * V[b,:]) @ W
                    # This is: M_ab[c] = sum_r U[a,r] V[b,r] W[c,r]
                    
                    # For fixed a: T[a,b,c] for all b,c
                    # T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                    # Let A_a be the matrix where A_a[b,c] = T[a,b,c]
                    # Let B_r be the matrix where B_r[b,c] = V[b,r] * W[c,r]
                    # Then A_a = sum_r U[a,r] B_r
                    # This is: vec(A_a) = sum_r U[a,r] vec(B_r)
                    # So vec(A_a) = U[a,:] @ B_flat
                    # where B_flat[r, :] = vec(B_r)
                    
                    # Build B_flat
                    B_flat = np.zeros((target_rank, mp * pn))
                    for r in range(target_rank):
                        B_flat[r, :] = (V_try[:, r] * W_try[:, r]).reshape(-1)  # mp*pn
                    
                    # For each a, solve: vec(T[a,:,:]) = U_try[a,:] @ B_flat
                    # This is an underdetermined system (target_rank <= mp*pn)
                    # Use least squares
                    for a in range(nm):
                        target_vec = T_target[a, :, :].reshape(-1)
                        # Solve U_try[a,:] @ B_flat = target_vec
                        # This is: B_flat.T @ U_try[a,:].T = target_vec
                        # Or: U_try[a,:] = (B_flat.T @ B_flat)^{-1} B_flat.T @ target_vec (if full rank)
                        # Use least squares
                        try:
                            sol, _, _, _ = np.linalg.lstsq(B_flat.T, target_vec, rcond=None)
                            U_try[a, :] = sol
                        except:
                            pass
                    
                    # Update V similarly
                    A_flat = np.zeros((target_rank, nm * pn))
                    for r in range(target_rank):
                        A_flat[r, :] = (U_try[:, r] * W_try[:, r]).reshape(-1)
                    
                    for b in range(mp):
                        target_vec = T_target[:, b, :].reshape(-1)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_flat.T, target_vec, rcond=None)
                            V_try[b, :] = sol
                        except:
                            pass
                    
                    # Update W similarly
                    C_flat = np.zeros((target_rank, nm * mp))
                    for r in range(target_rank):
                        C_flat[r, :] = (U_try[:, r] * V_try[:, r]).reshape(-1)
                    
                    for c in range(pn):
                        target_vec = T_target[:, :, c].reshape(-1)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(C_flat.T, target_vec, rcond=None)
                            W_try[c, :] = sol
                        except:
                            pass
                    
                    # Check convergence
                    valid, max_diff = _verify(U_try, V_try, W_try, n, m, p)
                    if valid and _check_half_integer(U_try, V_try, W_try):
                        return U_try, V_try, W_try, target_rank
                    
                    if max_diff < 1e-8:
                        # Check if it's close to half-integers
                        if _check_half_integer(U_try, V_try, W_try):
                            return U_try, V_try, W_try, target_rank
                    
                    if time.time() > deadline - 5:
                        break
                
                if time.time() > deadline - 5:
                    break
        
        return None
    
    # Try to find lower rank decompositions
    # First, let's try some known structured decompositions
    
    # The matmul tensor (3,4,6) - let's try to find a rank < 72 decomposition
    # Use the fact that we can try to merge some columns
    
    # Try ranks from 60 down to 40 (reasonable range for improvement)
    for target_rank in range(min(60, schoolbook_rank - 1), 39, -1):
        if time.time() > DEADLINE - 30:
            break
        
        result = try_reduce_rank(U, V, W, target_rank, DEADLINE - 30 - (time.time() - start_time))
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
            break
    
    # Convert to list format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
