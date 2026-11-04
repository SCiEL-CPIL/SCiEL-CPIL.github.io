"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_decomp_fast(U, V, W, tol=1e-9):
        """Vectorized verification using numpy"""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        # We can compute this as: for each r, outer product of U_col_r, V_col_r, W_col_r
        # Then sum over r and check the tensor T[i,j,k] = sum_r ...
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's compute this efficiently
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        
        # For each r, we have vectors u_r (n*m), v_r (m*p), w_r (p*n)
        # Contribution to T[i,j,k] from r: u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        
        # Build the full tensor
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_r = U_arr[:, r]  # (n*m,)
            v_r = V_arr[:, r]  # (m*p,)
            w_r = W_arr[:, r]  # (p*n,)
            
            # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
            u_mat = u_r.reshape(n, m)
            v_mat = v_r.reshape(m, p)
            w_mat = w_r.reshape(p, n)
            
            # For each (i,j,k): u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is like a trace-like operation
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # We can compute: for fixed i,j: sum_k v_mat[j,k]*w_mat[k,i] = (v_mat @ w_mat)[j,i]
            # Then T[i,j,k] isn't separable that way...
            
            # Actually: T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # For fixed i: T[i,j,k] = u_mat[i,j] * (v_mat @ w_mat)[j,k] ... no
            # (v_mat @ w_mat) is (m, n), entry [j,i] = sum_k v_mat[j,k]*w_mat[k,i]
            # That's not what we want.
            
            # Let me just do it directly with broadcasting
            # u_mat: (n,m), v_mat: (m,p), w_mat: (p,n)
            # T[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            
            # Create tensors:
            # u_tensor: (n, m, 1)
            # v_tensor: (1, m, p)  
            # w_tensor: (n, 1, p) but indexed as w_mat[k,i] so we need w_mat.T which is (n,p), then (n,1,p)
            
            u_t = u_mat[:, :, np.newaxis]  # (n, m, 1)
            v_t = v_mat[np.newaxis, :, :]  # (1, m, p)
            w_t = w_mat.T[np.newaxis, np.newaxis, :]  # (1, 1, p) ... wait
            # w_mat[k,i] -> we need shape (n, m, p) where entry [i,j,k] = w_mat[k,i]
            # w_mat is (p,n), so w_mat.T is (n,p)
            # We need to broadcast: for each (i,j,k), value = w_mat[k,i]
            # So w_broadcast[i,j,k] = w_mat.T[i,k] = w_mat[k,i]
            # Shape: (n, 1, p) broadcast to (n, m, p)
            w_t = w_mat.T[:, np.newaxis, :]  # (n, 1, p)
            
            T += u_t * v_t * w_t
        
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        return True
    
    # Build schoolbook decomposition as baseline
    U_school = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0.0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Strategy 1: Greedy elimination - try to remove columns
    def try_eliminate(U, V, W, max_rounds=20):
        """Try to eliminate redundant terms by zeroing out columns and checking validity"""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        eliminated = 0
        
        for _round in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            found = False
            for r in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Save state
                U_save = [row[:] for row in U]
                V_save = [row[:] for row in V]
                W_save = [row[:] for row in W]
                
                # Zero out column r
                for row in U:
                    row[r] = 0.0
                for row in V:
                    row[r] = 0.0
                for row in W:
                    row[r] = 0.0
                
                if verify_decomp(U, V, W):
                    # Remove column r
                    U = [row[:r] + row[r+1:] for row in U]
                    V = [row[:r] + row[r+1:] for row in V]
                    W = [row[:r] + row[r+1:] for row in W]
                    rank -= 1
                    eliminated += 1
                    found = True
                    break
                else:
                    U, V, W = U_save, V_save, W_save
            
            if not found:
                break
        
        return U, V, W, eliminated
    
    # Try elimination on schoolbook
    U_elim, V_elim, W_elim, elim_count = try_eliminate(U_school, V_school, W_school)
    
    if elim_count > 0:
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = len(U_elim[0])
    
    # Strategy 2: Use Strassen-like decomposition for sub-blocks
    # Partition 4x5 * 5x6 into blocks
    # A (4x5) = [A11 (2x2) | A12 (2x3); A21 (2x2) | A22 (2x3)]
    # B (5x6) = [B11 (2x2) | B12 (2x4); B21 (3x2) | B22 (3x4)]
    # C (4x6) = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    # For each block multiplication, we can use optimized decompositions
    # 2x2 * 2x2: Strassen gives rank 7 (vs schoolbook 8)
    # 2x2 * 2x4: can be seen as 2x2 * [2x2 | 2x2], rank 7+7=14 or 16 schoolbook
    # 2x3 * 3x2: rank 12 schoolbook, possibly better
    # 2x3 * 3x4: rank 24 schoolbook
    # 2x2 * 2x4: rank 16 schoolbook
    # 2x3 * 3x4: rank 24 schoolbook
    
    # Let's try a different approach: use known Strassen for 2x2*2x2 blocks
    # and build up from there
    
    # Actually, let's try a more systematic approach:
    # Use the fact that we can combine multiple Strassen-like tricks
    
    # Strategy 3: Randomized search for lower-rank decompositions
    # Start from schoolbook and try to merge/combine terms
    
    # Strategy 4: Use the structure of the problem more directly
    # For 4x5*5x6, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form in A and B with output C
    
    # The tensor is T[i,j,k] = e_i^T A e_j * e_j^T B e_k
    # In tensor form: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (A_{ij} * B_{jk})
    
    # A rank-r decomposition means:
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # And the constraint is that for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a gradient-based or iterative approach
    # Initialize with random low-rank and optimize
    
    def random_decomposition(rank):
        """Generate a random decomposition of given rank"""
        U_rand = np.random.randn(n * m, rank)
        V_rand = np.random.randn(m * p, rank)
        W_rand = np.random.randn(p * n, rank)
        
        # Normalize to make the sum closer to 1
        # For each (i,j,k), we want sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is hard to satisfy randomly, so let's just return raw random
        
        return (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
    
    # Try to find a decomposition by starting from schoolbook and perturbing
    # Then using gradient descent to reduce rank
    
    # Actually, let's try a different strategy:
    # Use the block structure and known results for smaller matrices
    
    # For 4x4*4x4, the optimal rank is 21 (Landsberg 2011)
    # We can use this as a building block
    
    # Let's construct a decomposition using 4x4*4x4 blocks
    # 4x5*5x6 = [4x4 | 4x1] * [[4x4],[4x2]]... 
    # Wait, B is 5x6, so partitioning B as [[4x4],[1x4]] doesn't work dimensionally for the multiplication
    
    # A (4x5) * B (5x6):
    # Partition A = [A1 (4x4) | A2 (4x1)]
    # Partition B = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x6
    # A2*B2: 4x1 * 1x6 (outer product, rank 24)
    
    # For 4x4*4x6:
    # Partition B1 = [B11 (4x4) | B12 (4x2)]
    # A1*B1 = A1*B11 + A1*B12
    # A1*B11: 4x4*4x4, rank 21 optimal
    # A1*B12: 4x4*4x2, rank 32 schoolbook
    
    # Total: 21 + 32 + 24 = 77
    
    # But we need to actually construct the rank-21 decomposition for 4x4*4x4
    # This is complex. Let's use a simpler approach.
    
    # Strategy 5: Use the "border rank" or approximate methods, then round
    # Or use known algebraic identities
    
    # For now, let's try a more aggressive elimination with different orderings
    
    # Try elimination in different orders
    for order in range(5):
        if time.time() > deadline - 10:
            break
        
        U_try = [row[:] for row in best_U]
        V_try = [row[:] for row in best_V]
        W_try = [row[:] for row in best_W]
        
        # Shuffle columns before elimination
        np.random.seed(order)
        perm = np.random.permutation(len(U_try[0]))
        U_try = [row[perm] for row in U_try]
        V_try = [row[perm] for row in V_try]
        W_try = [row[perm] for row in W_try]
        
        U_new, V_new, W_new, elim = try_eliminate(U_try, V_try, W_try, max_rounds=10)
        
        if len(U_new[0]) < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = len(U_new[0])
    
    # Strategy 6: Try to construct a decomposition using Kronecker product structure
    # 4x5*5x6 can be related to smaller multiplications
    
    # Let's try: view 4x5*5x6 as (2x2 blocks of 2x2 and 2x3)
    # A = [[A11, A12], [A21, A22]] where A11,A21 are 2x2, A12,A22 are 2x3
    # B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2... wait
    # B is 5x6, so partition into 2x2 blocks: rows 0-1, 2-3, 4; cols 0-1, 2-3, 4-5
    # This gives a 3x3 block structure which is messy
    
    # Let's just return the best we have
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
