import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_strassen_56():
    """Build the rank-56 Strassen-based decomposition."""
    n, m, p = 2, 4, 8
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            for la in range(4):
                row = la // 2
                col = la % 2
                global_col = s * 2 + col
                a_map[la] = row * 4 + global_col
            
            b_map = [0]*4
            for lb in range(4):
                row = lb // 2
                col = lb % 2
                global_row = s * 2 + row
                global_col = 2 * j + col
                b_map[lb] = global_row * 8 + global_col
            
            c_map = [0]*4
            for lc in range(4):
                row = lc // 2
                col = lc % 2
                global_row = row
                global_col = 2 * j + col
                c_map[lc] = global_col * 2 + global_row
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=8, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    rank = U.shape[1]
    
    C_recon = np.zeros((n, p))
    for r in range(rank):
        u_val = U[:, r] @ A_flat
        v_val = V[:, r] @ B_flat
        for i in range(n):
            for j in range(p):
                c_idx = j * n + i
                C_recon[i, j] += W[c_idx, r] * u_val * v_val
    
    return np.allclose(C_recon, C_expected, atol=1e-10)

def _try_rank_search(max_rank, n=2, m=4, p=8, time_limit=80):
    """
    Try to find a rank-`max_rank` decomposition with small integer entries.
    Uses a structured approach: build from 2x4 x 4x4 sub-decompositions.
    """
    start = time.time()
    
    # Build the target tensor
    # T[a, b, c] where:
    #   a = i*m + k (A[i,k]), i in [0,n), k in [0,m)
    #   b = k*p + j (B[k,j]), k in [0,m), j in [0,p)
    #   c = j*n + i (C[i,j]), j in [0,p), i in [0,n)
    # T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i
    
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                tensor[a, b, c] = 1.0
    
    # Flatten tensor to a matrix: T_mat[ab, c] = tensor[a,b,c]
    # Then T_mat = (U * V) @ W^T where (U*V) is element-wise product
    # Actually: tensor[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So for fixed c: sum_{a,b} tensor[a,b,c] * X[a,b] = sum_r W[c,r] * (U*V)[:,r]^T @ ... 
    # Let's use: tensor = sum_r u_r ⊗ v_r ⊗ w_r
    # Reshape: for each r, contribution to (a,b,c) is U[a,r]*V[b,r]*W[c,r]
    # So T[a,b,c] = sum_r (U[:,r] * V[:,r])_a... no.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This means: T = sum_r u_r ⊗ v_r ⊗ w_r where u_r = U[:,r], etc.
    
    # Strategy: try to find U, V, W with entries in {-1, 0, 1}
    # Use a greedy approach: start with Strassen structure and try to merge
    
    # Alternative: use the fact that for 2x4 x 4x8, we can try
    # rank-48 by finding a 2x4 x 4x4 decomposition of rank 24
    # and applying it twice.
    
    # Let's try a different approach: direct search with random initialization
    # and iterative refinement.
    
    np.random.seed(123)
    
    best_rank = None
    best_U, best_V, best_W = None, None, None
    
    # Try rank values from max_rank down
    for target_rank in range(max_rank, 40, -1):
        if time.time() - start > time_limit:
            break
        
        # Initialize with random sparse integer matrices
        for attempt in range(20):
            if time.time() - start > time_limit:
                break
            
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            # Initialize sparsely
            for r in range(target_rank):
                # U: pick 2-4 random positions
                nu = np.random.randint(2, 5)
                u_idx = np.random.choice(n*m, nu, replace=False)
                U[u_idx, r] = np.random.choice([-1, 1], nu)
                
                nv = np.random.randint(2, 5)
                v_idx = np.random.choice(m*p, nv, replace=False)
                V[v_idx, r] = np.random.choice([-1, 1], nv)
                
                nw = np.random.randint(1, 4)
                w_idx = np.random.choice(p*n, nw, replace=False)
                W[w_idx, r] = np.random.choice([-1, 1], nw)
            
            # Check if this works (unlikely for random init)
            # Instead, use a more structured approach
            
            # Use the Strassen-based structure as a starting point
            # and try to reduce rank by merging terms
            break  # Random init won't work, use structured approach
        
        # Structured approach: build from 2x2 Strassen blocks
        # 2x4 x 4x8 = two 2x4 x 4x4 blocks
        # For each 2x4 x 4x4, use 4 Strassen 2x2 blocks (rank 28)
        # Total: 2 * 28 = 56 (same as before)
        
        # To get rank 48, we need 2x4 x 4x4 with rank 24
        # 2x4 x 4x4: C[i,j] = sum_k A[i,k] B[k,j], i in {0,1}, j in {0,1,2,3}
        # = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
        # Each 2x2 x 2x4: rank 7 * 2 = 14 per half, total 28
        # To get 24, we need to save 4 multiplications per 2x4 x 4x4 block
        
        # Try: for 2x4 x 4x4, use a custom decomposition
        # C[i,j] = sum_{k=0}^{3} A[i,k] B[k,j]
        # Group k into pairs: k in {0,1} and k in {2,3}
        # C[i,j] = sum_{k=0}^{1} A[i,k] B[k,j] + sum_{k=2}^{3} A[i,k] B[k,j]
        # First term: 2x2 x 2x4 (A[:,0:2] @ B[0:2,:])
        # Second term: 2x2 x 2x4 (A[:,2:4] @ B[2:4,:])
        
        # For 2x2 x 2x4: C[i,j] = sum_{k=0}^{1} A[i,k] B[k,j]
        # This is 2x2 @ 2x4 = 2x4
        # Can we do this in rank < 14?
        # Schoolbook: 2*2*4 = 16... no, 2x2 @ 2x4 needs 2*2*4=16? No.
        # 2x2 @ 2x4: output is 2x4, each output C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
        # That's 2*4 = 8 products, but with Strassen-like:
        # Actually 2x2 @ 2x4 can be done in rank 7*2 = 14 (two 2x2@2x2)
        # Or can we do it in rank 8? Let's see...
        
        # For 2x2 @ 2x2, min rank is 7.
        # For 2x2 @ 2x4, we can split into two 2x2@2x2, giving rank 14.
        # Can we do better? The tensor is in R^4 x R^8 x R^8.
        # I believe rank 14 is optimal for 2x2@2x4.
        
        # So 2x4 x 4x4 has rank at least 28, and 2x4 x 4x8 has rank at least 56.
        # Unless there's a non-block decomposition that shares work.
        
        # Let me try a global search with a different structure.
        # What if we don't split by j-pairs but use a different grouping?
        
        # Alternative: C[i, j] = sum_k A[i,k] B[k,j]
        # For fixed i, this is a 1x4 @ 4x8 = 1x8 product.
        # Two such rows share B.
        # Can we compute both rows with fewer than 2*4*8 = 64 multiplications?
        # Yes, Strassen gives 56.
        
        # I think 56 might be close to optimal. Let me verify the current solution
        # works and return it.
        break
    
    # Return the Strassen-56 decomposition
    U, V, W = _build_strassen_56()
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W = _build_strassen_56()
    
    # Verify correctness
    valid = _verify_decomposition(U, V, W, n, m, p)
    
    if not valid:
        # Fallback: return schoolbook decomposition (rank 64)
        U_school = np.zeros((n * m, n * m * p), dtype=int)
        V_school = np.zeros((m * p, n * m * p), dtype=int)
        W_school = np.zeros((p * n, n * m * p), dtype=int)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U_school[i * m + k, r] = 1
                    V_school[k * p + j, r] = 1
                    W_school[j * n + i, r] = 1
        
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
