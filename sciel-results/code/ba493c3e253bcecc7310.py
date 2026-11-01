"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication target tensor."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = _build_target_tensor(n, m, p)
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _block_decomposition(n, m, p):
    """
    Construct a rank-26 decomposition by splitting 3x3 into 2x2 + 1x1 blocks
    and applying Strassen's algorithm to the 2x2 part.
    
    The 3x3 matrix multiplication can be viewed as:
    - Top-left 2x2 block: uses Strassen (7 multiplications)
    - Cross terms and bottom-right: use standard multiplication
    
    This gives rank 26 instead of 27.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen's 7 products for the 2x2 top-left block
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Indexing: a = i*m+j, b = j*p+k, c = k*n+i
    # For 3x3: i,j,k in {0,1,2}
    
    # M1 = (a11+a22)(b11+b22)
    # a11: i=0,j=0; a22: i=1,j=1; b11: j=0,k=0; b22: j=1,k=1
    # This product contributes to:
    # C11 (k=0,i=0): +1
    # C22 (k=1,i=1): +1
    # But we need to track all contributions carefully
    
    # Let me use a different approach: define the 7 Strassen products explicitly
    # and track their contributions to the 2x2 block of C
    
    # Product 1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0*3+0, col] = 1  # A[0,0]
    U[1*3+1, col] = 1  # A[1,1]
    V[0*3+0, col] = 1  # B[0,0]
    V[1*3+1, col] = 1  # B[1,1]
    # Contributes to C[0,0] with +1, C[1,1] with +1
    W[0*3+0, col] = 1  # C[0,0]
    W[1*3+1, col] = 1  # C[1,1]
    col += 1
    
    # Product 2: (A[1,0]+A[1,1]) * B[0,0]
    U[1*3+0, col] = 1  # A[1,0]
    U[1*3+1, col] = 1  # A[1,1]
    V[0*3+0, col] = 1  # B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    W[0*3+1, col] = 1  # C[1,0]
    W[1*3+1, col] = -1 # C[1,1]
    col += 1
    
    # Product 3: A[0,0] * (B[0,1]-B[1,1])
    U[0*3+0, col] = 1  # A[0,0]
    V[0*3+1, col] = 1  # B[0,1]
    V[1*3+1, col] = -1 # B[1,1]
    # Contributes to C[0,1] with +1, C[1,1] with +1
    W[1*3+0, col] = 1  # C[0,1]
    W[1*3+1, col] = 1  # C[1,1]
    col += 1
    
    # Product 4: A[1,1] * (B[1,0]-B[0,0])
    U[1*3+1, col] = 1  # A[1,1]
    V[1*3+0, col] = 1  # B[1,0]
    V[0*3+0, col] = -1 # B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    W[0*3+1, col] = 1  # C[1,0]
    W[1*3+1, col] = -1 # C[1,1]
    col += 1
    
    # Product 5: (A[0,0]+A[0,1]) * B[1,1]
    U[0*3+0, col] = 1  # A[0,0]
    U[0*3+1, col] = 1  # A[0,1]
    V[1*3+1, col] = 1  # B[1,1]
    # Contributes to C[0,1] with +1, C[1,1] with -1
    W[1*3+0, col] = 1  # C[0,1]
    W[1*3+1, col] = -1 # C[1,1]
    col += 1
    
    # Product 6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*3+0, col] = 1  # A[1,0]
    U[0*3+0, col] = -1 # A[0,0]
    V[0*3+0, col] = 1  # B[0,0]
    V[0*3+1, col] = 1  # B[0,1]
    # Contributes to C[1,1] with +1
    W[1*3+1, col] = 1  # C[1,1]
    col += 1
    
    # Product 7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*3+1, col] = 1  # A[0,1]
    U[1*3+1, col] = -1 # A[1,1]
    V[1*3+0, col] = 1  # B[1,0]
    V[1*3+1, col] = 1  # B[1,1]
    # Contributes to C[0,0] with +1, C[1,1] with +1
    W[0*3+0, col] = 1  # C[0,0]
    W[1*3+1, col] = 1  # C[1,1]
    col += 1
    
    # Now handle the remaining terms involving index 2 (the third row/column)
    # We need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 15 more terms, but we already used 7, so total would be 22.
    # Wait, let me recount. The 2x2 block has 4 entries, each needing 2 products = 8, 
    # but Strassen reduces to 7. The cross terms:
    # C[0,2]: 3 terms
    # C[1,2]: 3 terms  
    # C[2,0]: 3 terms
    # C[2,1]: 3 terms
    # C[2,2]: 3 terms
    # Total cross: 15 terms
    # Total: 7 + 15 = 22? That doesn't seem right.
    
    # Let me reconsider. The full 3x3 multiplication has 27 terms.
    # The 2x2 block (indices 0,1) has 4*2*2 = 16 terms in schoolbook.
    # Strassen reduces these 16 to 7, saving 9.
    # So total rank = 27 - 9 = 18? No, that's not right either.
    
    # Actually, the 2x2 sub-block uses indices i,j,k in {0,1}.
    # Terms: for i in {0,1}, j in {0,1}, k in {0,1}: 8 terms.
    # Strassen reduces 8 to 7, saving 1.
    # So total rank = 27 - 1 = 26. That matches!
    
    # So I need to add the remaining 19 terms (27 - 7 - 1 = 19? No, 27 - 7 = 20)
    # Wait: total terms = 27. Strassen handles 8 of them (the 2x2x2 sub-tensor) with 7 products.
    # Remaining: 27 - 8 = 19 terms.
    # Total rank: 7 + 19 = 26. Correct.
    
    # Let me list which terms are NOT in the 2x2x2 sub-tensor:
    # Terms where at least one of i,j,k is 2.
    # Total terms with i,j,k in {0,1}: 2*2*2 = 8
    # Terms with at least one index = 2: 27 - 8 = 19
    
    # Add these 19 terms:
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    # Check if this term is already handled by Strassen
                    # Strassen only handles i,j,k all in {0,1}
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _als_decompose(T, rank, n, m, p, max_iters=500, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
            
    prev_error = float('inf')
    stall_count = 0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            if best_error < 1e-6:
                return best_U, best_V, best_W
            return None
            
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if abs(prev_error - error) < tol:
            stall_count += 1
            if stall_count >= 3:
                if error < 1e-7:
                    return U, V, W
                else:
                    return _polish_pass(U, V, W, T, rank, n, m, p, deadline)
        else:
            stall_count = 0
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    if error < 1e-4:
        return _polish_pass(U, V, W, T, rank, n, m, p, deadline)
    return None

def _polish_pass(U, V, W, T, rank, n, m, p, deadline, max_iters=200, tol=1e-12):
    """High-precision polish pass."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-8:
                return U, V, W
            else:
                return None
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-8:
        return U, V, W
    return None

def _try_rank(target_rank, n, m, p, deadline, max_attempts=50, seed=42):
    """
    Attempt to find a decomposition of given rank using ALS with multiple strategies.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    
    U0, V0, W0 = _block_decomposition(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        # Rank is same as or higher than block decomposition
        if target_rank == U0.shape[1]:
            return U0, V0, W0
        # Need to reduce further
        pass
    
    if target_rank > U0.shape[1]:
        # Can't increase rank with this approach
        return None
    
    rng = random.Random(seed)
    
    # Try ALS from the block decomposition
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        init_U = U0.copy()
        init_V = V0.copy()
        init_W = W0.copy()
        
        # Add perturbation
        perturb_scale = rng.uniform(0.01, 0.1)
        for r_idx in range(target_rank):
            mask_u = init_U[:, r_idx] != 0
            mask_v = init_V[:, r_idx] != 0
            mask_w = init_W[:, r_idx] != 0
            if mask_u.any():
                init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_v.any():
                init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_w.any():
                init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            
        result = _als_decompose(T, target_rank, n, m, p, max_iters=150, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # First, try the structural block decomposition (rank 26)
    U, V, W = _block_decomposition(n, m, p)
    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
        # Try to reduce further
        for target_rank in [23, 24, 25]:
            if time.time() > deadline:
                break
            result = _try_rank(target_rank, n, m, p, deadline, max_attempts=30, seed=42)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Return rank 26
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If block decomposition failed verification, fall back to schoolbook
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

# EVOLVE-BLOCK-END
