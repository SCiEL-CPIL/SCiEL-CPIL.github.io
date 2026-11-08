"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 6
    rank_schoolbook = n * m * p  # 48

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 otherwise
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook solution as fallback
    def schoolbook():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()

    def to_list(U, V, W):
        return {
            "U": [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in U],
            "V": [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in V],
            "W": [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in W]
        }

    def validate(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor T."""
        r = U.shape[1]
        for c in range(r):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            # Compute outer product u_col ⊗ v_col ⊗ w_col and check against T
            # This is expensive, so do it efficiently:
            # For each (i,j,k), T[i*m+j, j*p+k, k*n+i] should equal
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We can check by reconstructing the full tensor
        # Reconstruct and compare
        recon = np.zeros_like(T)
        for c in range(r):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        # Only check entries that should be 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = recon[i*m+j, j*p+k, k*n+i]
                    err = abs(actual - expected)
                    if err > max_err:
                        max_err = err
                    if max_err > tol:
                        return False, max_err
        # Also check that "wrong" entries are 0
        # Actually, the tensor T is sparse - only (i*m+j, j*p+k, k*n+i) entries are 1
        # But the reconstruction might have non-zero entries elsewhere
        # Let's check a few random "wrong" entries
        # Actually, for correctness we need ALL entries to match
        # But that's 8*24*12 = 2304 entries, which is fine
        for idx_a in range(n*m):
            for idx_b in range(m*p):
                for idx_c in range(p*n):
                    expected = T[idx_a, idx_b, idx_c]
                    actual = recon[idx_a, idx_b, idx_c]
                    if abs(actual - expected) > tol:
                        return False, abs(actual - expected)
        return True, 0.0

    # Try Strassen-like block decomposition
    # Split A (2x4) into two 2x2 blocks, B (4x6) into two 2x3 blocks
    # Use Strassen's 7-multiplication scheme for 2x2 * 2x2
    # Then extend to handle the 2x3 case

    # Actually, let's try a more direct approach:
    # Use the structure of the problem to find a low-rank decomposition
    
    # Try ALS (Alternating Least Squares) to find low-rank decomposition
    # Minimize ||T - sum_c u_c ⊗ v_c ⊗ w_c||_F
    
    def als_decompose(target, r, n_iters=200, tol=1e-10, seed=42):
        """ALS tensor decomposition."""
        np.random.seed(seed)
        # Initialize with random vectors
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each row a, U[a,:] = (sum_b T[a,b,:] * V[b,:] ⊙ W.T) / (sum_b (V[b,:] ⊙ W.T)^2)
            # This is complex, let's use a simpler approach
            
            # Actually, ALS for tensor is:
            # Fix V, W, solve for U:
            # T = sum_c u_c ⊗ v_c ⊗ w_c
            # For fixed V, W: T reshaped = U @ (V ⊙ W) in some sense
            # Let's use the unfolding approach
            
            # Unfold T along mode 1: T_(1) is (n*m) x (m*p * p*n)
            # T_(1)[a, (b,c)] = T[a,b,c]
            # T_(1) = U @ (W ⊙ V) where ⊙ is Khatri-Rao product
            # (W ⊙ V) has shape (p*n * m*p, r) where each column is w_c ⊗ v_c
            
            # Build Khatri-Rao product
            # For each c, the column is w_c ⊗ v_c (size (p*n)*(m*p))
            # This is too large: 12 * 24 = 288 columns of size 288
            
            # Alternative: iterate over rows of U
            # For row a: U[a,:] satisfies
            # sum_c U[a,c] * (v_c ⊗ w_c) = T[a, :, :] reshaped
            # Let T_a = T[a, :, :].reshape(-1) (size m*p*p*n = 24*12 = 288)
            # Let K = [v_1 ⊗ w_1, v_2 ⊗ w_2, ..., v_r ⊗ w_r] (size 288 x r)
            # Then U[a,:] = (K^T K)^{-1} K^T T_a
            
            # But building K is expensive. Let's use a different approach.
            
            # Simpler: use gradient descent or random restarts
            
            pass
        
        return best_U, best_V, best_W, best_err

    # Let's try a different strategy: use the block structure
    # A is 2x4, split into A1 (2x2) and A2 (2x2)
    # B is 4x6, split into B1 (2x3) and B2 (2x3)
    # C = A1*B1 + A2*B2 (each product is 2x3)
    
    # For each 2x2 * 2x3 product, we can try to find a low-rank decomposition
    # 2x2 * 2x3 schoolbook: 2*2*3 = 12 multiplications
    # Can we do better?
    
    # Actually, let's just try to verify if rank 42 works (from inspiration 1)
    # and also try to find even lower ranks
    
    # First, let's try the Strassen-based construction from inspiration 1
    # It claims rank 42
    
    def strassen_block(n, m, p, deadline):
        """Construct a Strassen-like decomposition for n x m * m x p."""
        # For n=2, m=4, p=6:
        # Split A into 2 blocks of 2x2
        # Split B into 2 blocks of 2x3
        # Split C into 2 blocks of 2x3
        
        # For each pair of blocks (A_i, B_j), compute C_{i,j} = A_i * B_j
        # But we can use Strassen's trick if both are square
        
        # Since B blocks are 2x3 (not square), let's handle differently
        # Split B into three 2x2 blocks? No, 4x6 = 2x3 blocks of 2x2
        # Actually, 4 rows, 6 cols: split into 2 row-blocks of 2 rows, and 3 col-blocks of 2 cols
        # So B has 2*3 = 6 blocks of 2x2
        
        # A has 2 row-blocks of 2 rows (since n=2, m=4: 1 row-block of 2 rows, 2 col-blocks of 2 cols)
        # Wait, A is 2x4: 1 row-block of 2 rows, 2 col-blocks of 2 cols
        # So A has 1*2 = 2 blocks of 2x2
        
        # C is 2x6: 1 row-block of 2 rows, 3 col-blocks of 2 cols
        # So C has 1*3 = 3 blocks of 2x2
        
        # C = A * B where A is 2x4, B is 4x6
        # In blocks: C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
        # But A has 1 row-block, B has 2 row-blocks, so:
        # C_block[i,j] = A_block[0,0] * B_block[0,j] + A_block[0,1] * B_block[1,j]
        # for i=0, j=0,1,2
        
        # So we have 3 output blocks, each computed from 2 products of 2x2 matrices
        # For each output block, we need to compute:
        # C_j = A1 * B1j + A2 * B2j
        # where A1, A2 are 2x2 and B1j, B2j are 2x2
        
        # This is a 2x2 * 2x2 addition problem
        # We can use Strassen's 7-multiplication scheme for the sum
        
        # Actually, let's just use schoolbook for each 2x2*2x2 product
        # That gives 4 multiplications per product, 2 products per output block, 3 output blocks
        # Total: 3 * 2 * 4 = 24 multiplications
        # But we can share some work
        
        # Let's try a different approach: just use the direct schoolbook
        # and see if we can reduce
        
        # For now, return the schoolbook solution
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Let's try to find a lower-rank decomposition using random search + refinement
    # Start with rank 40 and try to find a valid decomposition
    
    best_rank = rank_schoolbook
    best_result = to_list(U_sb, V_sb, W_sb)
    
    # Try ranks from 40 down to 30
    for target_rank in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 12345 + target_rank)
            
            # Initialize with random vectors
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize
            for c in range(target_rank):
                U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-10)
                V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-10)
                W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-10)
            
            # Try to refine using alternating least squares
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for a in range(n*m):
                    # T[a, :, :] = sum_c U[a,c] * (V[:,c] ⊗ W[:,c])
                    # Let T_a = T[a, :, :].reshape(-1)
                    T_a = T[a, :, :].reshape(-1)
                    # Build the design matrix: each column c is V[:,c] ⊗ W[:,c]
                    # This is size (m*p*p*n) x target_rank = 288 x target_rank
                    K = np.zeros((m*p*p*n, target_rank))
                    for c in range(target_rank):
                        K[:, c] = np.kron(V[:, c], W[:, c])
                    # Solve least squares
                    try:
                        U[a, :] = np.linalg.lstsq(K, T_a, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    T_b = T[:, b, :].reshape(-1)
                    K = np.zeros((n*m*p*n, target_rank))
                    for c in range(target_rank):
                        K[:, c] = np.kron(U[:, c], W[:, c])
                    try:
                        V[b, :] = np.linalg.lstsq(K, T_b, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c_idx in range(p*n):
                    T_c = T[:, :, c_idx].reshape(-1)
                    K = np.zeros((n*m*m*p, target_rank))
                    for c in range(target_rank):
                        K[:, c] = np.kron(U[:, c], V[:, c])
                    try:
                        W[c_idx, :] = np.linalg.lstsq(K, T_c, rcond=None)[0]
                    except:
                        pass
            
            # Check if the decomposition is valid
            valid, err = validate(U, V, W, tol=1e-6)
            if valid:
                found = True
                best_rank = target_rank
                best_result = to_list(U, V, W)
                break
        
        if found:
            break
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return to_list(U_sb, V_sb, W_sb)
# EVOLVE-BLOCK-END
