"""Tensor decomposition search for 3x3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
SEED = 42
RANDOM_SEED = 12345

# EVOLVE-BLOCK-START

def _schoolbook():
    """Standard rank-54 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c] == delta_{k*n+i, ...} 
    Actually the tensor T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]
    and the output is W[k*n+i,c] for the (i,k) pair.
    
    The condition is: for all i,j,k:
        delta_{i,k} (if output dimension matched) ... 
    Wait, let me re-derive.
    
    Matrix mult: (A * B)[i,k] = sum_j A[i,j]*B[j,k]
    Tensor: T[i,j,k] = 1 if we're looking at the (i,j,k) trilinear form.
    Actually the matrix multiplication tensor is:
        M = sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k}
    as a trilinear form: (A,B) -> sum_{i,j,k} A_{i,j} B_{j,k} (output)_{i,k}
    
    The rank decomposition says:
        sum_{i,j,k} A_{i,j} B_{j,k} output_{i,k} = sum_c (sum_{i,j} U_{i,j}^{(c)} A_{i,j}) (sum_{j,k} V_{j,k}^{(c)} B_{j,k}) * W^{(c)}
    
    So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But T[i,j,k] = 1 for all i,j,k? No.
    
    Let me think again. The matrix multiplication tensor T has:
        T[i,j,k] = 1 if we consider it as the identity in some sense.
        
    Actually, the correct identity is:
        (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        
    As a tensor in F^{n*m} ⊗ F^{m*p} ⊗ F^{p*n} (using the output indexed by (k,n)):
        T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    where e_{i,j} is the basis vector in F^{n*m} with 1 at position i*m+j,
    e_{j,k} is basis in F^{m*p} at position j*p+k,
    e_{k,i} is basis in F^{p*n} at position k*n+i.
    
    So T as a 3D tensor: T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k), else 0.
    
    A rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    where u_c ∈ F^{n*m}, v_c ∈ F^{m*p}, w_c ∈ F^{p*n}.
    
    This means: T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]
    
    So for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i): sum = 0.
    
    The "other" indices are those where the middle index doesn't match: 
    i.e., if a = i'*m+j', b = j''*p+k'', c = k'*n+i'' and j' != j'' or i'' != k' etc.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    n, m, p = DIMENSIONS
    rank = len(U_arr[0]) if U_arr.ndim == 2 else U_arr.shape[1]
    
    # Build the tensor T as a flattened check
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    # Check: for each i,j,k:
    #   sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # For other combos where indices don't align, should be 0.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    # Check off-diagonal: for each a,b,c where the indices don't form a valid (i,j,k)
    # a valid combo: a = i*m+j, b = j'*p+k, c = k'*n+i' where j=j' and k=k' and i=i'
    # Invalid: j != j' or k != k' or i != i'
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Decode
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue  # valid, already checked
                val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
                err = abs(val)
                max_err = max(max_err, err)
    
    return max_err < tol, max_err


def _als_decomposition(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """ALS (Alternating Least Squares) for tensor decomposition.
    
    Initialize randomly, then alternate:
      - Fix V, W, solve for U
      - Fix U, W, solve for V
      - Fix U, V, solve for W
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Target tensor (flattened over one mode)
    # We'll use the identity: T[a,b,c] = 1 if valid, 0 otherwise
    # For ALS, we fix one factor and solve a linear system for the other.
    
    # Precompute target values
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    for it in range(max_iters):
        # Solve for U: for each a, U[a,:] should satisfy
        #   sum_b,c target[a,b,c] V[b,:]^T W[c,:] ... 
        # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Fix V,W: for each a:
        #   T[a,:,:] = U[a,:] @ (V^T ⊙ W^T)  ... using Khatri-Rao
        #   T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        #   So T[a,:,:].flatten() = U[a,:] @ (khatri(V[:,], W[:,])).T  where khatri has rows indexed by (b,c)
        
        # Build Khatri-Rao product of V and W: K_VW has shape (m*p)*(p*n) x rank
        # K_VW[b*(p*n)+c, r] = V[b,r]*W[c,r]
        K_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m*p*p*n, rank)
        
        # For each a: T[a,:,:].flatten() = U[a,:] @ K_VW.T
        # So U[a,:] = T[a,:,:].flatten() @ (K_VW.T @ K_VW)^{-1} @ K_VW
        # = T[a,:,:].flatten() @ K_VW.T  if K_VW has orthonormal columns (use lstsq)
        
        # Solve: U[a,:] @ K_VW.T = T[a,:,:].flatten()
        # i.e., K_VW @ U[a,:]^T = T[a,:,:].flatten()
        
        T_flat = target.reshape(n*m, -1)  # (n*m, m*p*p*n)
        
        # Solve K_VW @ U^T = T_flat^T  =>  U^T = K_VW^+ @ T_flat^T
        # Use lstsq for stability
        try:
            U_new, _, _, _ = np.linalg.lstsq(K_VW, T_flat.T, rcond=None)
            U = U_new.T  # (n*m, rank)
        except np.linalg.LinAlgError:
            break
        
        # Solve for V: for each b:
        #   T[:,b,c] over a,c: sum_r V[b,r] * (U[a,r]*W[c,r])
        #   T[:,b,c].flatten() = V[b,:] @ (K_UW[:, :]).T  where K_UW[a*(p*n)+c, r] = U[a,r]*W[c,r]
        K_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n*m*p*n, rank)
        
        # T[:,b,:] has shape (n*m, p*n), flatten to (n*m*p*n,)
        # For each b: T[:,b,:].flatten() = V[b,:] @ K_UW.T
        # T reshaped: (n*m, m*p, p*n) -> for fixed b, T[:,b,:] -> (n*m, p*n)
        T_V = np.zeros((m*p, n*m*p*n))
        for b in range(m*p):
            T_V[b, :] = target[:, b, :].flatten()
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(K_UW, T_V.T, rcond=None)
            V = V_new.T  # (m*p, rank)
        except np.linalg.LinAlgError:
            break
        
        # Solve for W: for each c:
        #   T[a,b,c] over a,b: sum_r W[c,r] * (U[a,r]*V[b,r])
        K_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n*m*m*p, rank)
        
        T_W = np.zeros((p*n, n*m*m*p))
        for c in range(p*n):
            T_W[c, :] = target[:, :, c].flatten()
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(K_UV, T_W.T, rcond=None)
            W = W_new.T  # (p*n, rank)
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        if it % 10 == 0 or it == max_iters - 1:
            err, _ = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e4)
            # Just compute max error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        val = np.dot(U[a,:], np.multiply(V[b,:], W[c,:]))
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < tol:
                break
    
    # Convert to lists
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank(rank, time_limit=100.0):
    """Try to find a decomposition of given rank."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        U, V, W = _als_decomposition(n, m, p, rank, max_iters=100, seed=attempt)
        ok, err = _verify(U, V, W, n, m, p, tol=1e-8)
        if ok:
            return U, V, W
        # Also try with different max_iters
        if time.time() - start_time > time_limit:
            break
        U, V, W = _als_decomposition(n, m, p, rank, max_iters=300, seed=1000+attempt)
        ok, err = _verify(U, V, W, n, m, p, tol=1e-8)
        if ok:
            return U, V, W
    
    return None


def _try_lower_rank_bruteforce(rank, time_limit=30.0):
    """Try to find decomposition by random search with refinement."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    for seed in range(100):
        if time.time() - start_time > time_limit:
            break
        U, V, W = _als_decomposition(n, m, p, rank, max_iters=50, seed=seed)
        ok, err = _verify(U, V, W, n, m, p, tol=1e-6)
        if ok:
            return U, V, W
        # Refine with more iterations
        if time.time() - start_time > time_limit:
            break
        U2, V2, W2 = _als_decomposition(n, m, p, rank, max_iters=200, seed=seed)
        ok, err = _verify(U2, V2, W2, n, m, p, tol=1e-8)
        if ok:
            return U2, V2, W2
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 140.0  # safety margin
    
    # Start with schoolbook (rank 54) as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to reduce rank
    # Known lower bounds: for 3x3x6, the rank is at least 54? 
    # Actually for 2x2x2 it's 7 (Strassen), for 3x3x3 it's 19-23.
    # For 3x3x6, let's try ranks below 54.
    
    # Try ALS with various ranks
    for rank in range(53, 40, -1):
        if time.time() - start > deadline:
            break
        time_left = deadline - (time.time() - start)
        if time_left < 5:
            break
        result = _try_rank(rank, time_limit=min(time_left - 1, 100))
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    # If we found a lower rank, try to go even lower
    if best_rank < 54:
        for rank in range(best_rank - 1, max(20, best_rank - 5), -1):
            if time.time() - start > deadline:
                break
            time_left = deadline - (time.time() - start)
            if time_left < 5:
                break
            result = _try_rank(rank, time_limit=min(time_left - 1, 100))
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best

# EVOLVE-BLOCK-END
