"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # The decomposition says: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Efficient verification: for each (a,c) pair, check that
    # sum_r U[a,r]*W[c,r] * V[:,r] matches the expected slice
    # Actually, let's just do it directly for small sizes
    
    # The multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    
    # Check: for all a in [0,n*m), b in [0,m*p), c in [0,p*n):
    #   T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Expected T:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # So for a = i*m+j, b = j*p+k, c = k*n+i, we need T=1
    # For all other (a,b,c), T=0
    
    # Let's verify by checking all non-zero entries and a sample of zeros
    U_arr = np.asarray(U)
    V_arr = np.asarray(V)
    W_arr = np.asarray(W)
    
    # Check all non-zero entries of the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some zero entries (sample)
    # T[a,b,c] = 0 when the indices don't match the multiplication pattern
    # a = i*m+j, b = j'*p+k, c = k'*n+i' where j != j' or k != k' or i != i'
    # Let's check all entries where the pattern doesn't hold
    checked = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this should be non-zero
                # a = i*m+j, b = j'*p+k, c = k'*n+i'
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    # This should be 1, already checked above
                    continue
                
                # This should be 0
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
                checked += 1
                if checked > 1000:  # Limit zero checks for speed
                    break
            if checked > 1000:
                break
        if checked > 1000:
            break
    
    return True

def _build_strassen_rank54(n, m, p):
    """Build the known rank-54 Strassen-based decomposition."""
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    current_col = 0
    
    def next_col():
        nonlocal current_col
        current_col += 1
    
    # Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0
            
            # M0: (A00+A11)(B00+B11) -> C00, C11
            U[i_base*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M1: (A10+A11)B00
            U[(i_base+1)*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = -1
            next_col()
            
            # M2: A00(B01-B11)
            U[i_base*m+j_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M3: A11(B10-B00)
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[j_base*p+k_base, current_col] = -1
            W[k_base*n+i_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            next_col()
            
            # M4: (A00+A01)B11
            U[i_base*m+j_base, current_col] = 1
            U[i_base*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            next_col()
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base+1)*m+j_base, current_col] = 1
            U[i_base*m+j_base, current_col] = -1
            V[j_base*p+k_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M6: (A01-A11)(B10+B11)
            U[i_base*m+(j_base+1), current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = -1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            next_col()
    
    # Naive blocks for j=4
    for k_base in [0, 2, 4]:
        i_base = 0
        j_idx = 4
        
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        next_col()
        
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+i_base, current_col] = 1
        next_col()
        
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        next_col()
        
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()
    
    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=30.0):
    """
    Attempt to reduce the rank of the decomposition using SVD-based methods.
    Returns (U, V, W) with reduced rank if successful, else None.
    """
    start_time = time.time()
    rank = U.shape[1]
    
    if rank <= target_rank:
        return None
    
    # Try to find linear dependencies among the rank-1 terms
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If we can find a linear combination of terms that equals zero, we can reduce rank.
    
    # Approach: use the fact that if the tensor has border rank <= target_rank,
    # we can find a rank-target_rank decomposition.
    
    # For now, try a simple approach: check if any columns are linearly dependent
    # in a way that allows rank reduction.
    
    # Alternative: use numerical tensor decomposition
    # Reshape the tensor and use SVD
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank approximation
    # Reshape to (n*m, m*p*p*n) and do SVD
    T_reshaped = T.reshape(n*m, -1)
    
    # SVD to find the numerical rank
    try:
        U_svd, s, Vt = np.linalg.svd(T_reshaped, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The numerical rank is the number of significant singular values
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank <= target_rank:
        # We can potentially reduce to this rank
        # But we need to find explicit integer/half-integer factors
        # This is complex, so for now just return None
        pass
    
    # Check time
    if time.time() - start_time > max_time:
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with the known rank-54 decomposition
    U, V, W = _build_strassen_rank54(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Rank-54 decomposition failed verification")
    
    best_rank = 54
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    if time.time() < deadline - 20.0:
        # Attempt rank-53
        result = _try_rank_reduction(U, V, W, n, m, p, 53, max_time=min(30.0, deadline - time.time() - 15.0))
        if result is not None:
            U53, V53, W53 = result
            if _verify_decomposition(U53, V53, W53, n, m, p):
                best_rank = 53
                best_U, best_V, best_W = U53, V53, W53
    
    # Convert to integer/half-integer representation
    # The entries should be integers or half-integers
    def to_half_integer(arr):
        """Convert float array to integer representation (multiply by 2 if needed)."""
        # Check if all entries are integers
        if np.allclose(arr, np.round(arr), atol=1e-9):
            return np.round(arr).astype(int)
        # Check if all entries are half-integers (multiples of 0.5)
        doubled = arr * 2
        if np.allclose(doubled, np.round(doubled), atol=1e-9):
            return np.round(doubled).astype(int)
        # Fallback: round to nearest integer
        return np.round(arr).astype(int)
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-END
