"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 48).
    2. Use a greedy rank-reduction approach: try to replace pairs of schoolbook
       terms with a single term that captures their combined contribution, or
       identify linear dependencies among the factor vectors.
    3. Verify each candidate decomposition before accepting it.
    
    The key insight is that for small matrix sizes, there may be non-trivial
    tensor rank reductions below the schoolbook bound. We use a local search
    that attempts to merge terms by checking if two rank-1 components can be
    expressed as a single rank-1 component (which is generally impossible),
    or by finding linear combinations that allow cancellation.
    
    A more practical approach: use the known Strassen-style decomposition
    for the 2x2 block structure. Partition the 3x4 * 4x4 multiplication into
    blocks and apply rank-7 (Strassen) to the 2x2 sub-problems where possible.
    
    For 3x4 * 4x4:
    - Split B (4x4) into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    - Split A (3x4) into two 3x2 blocks: A = [A0, A1]
    - C = A * B = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    - Each term like A0*B00 is (3x2)*(2x2) -> 3x2 output
    
    For (3x2)*(2x2): this is 3 rows, each a (1x2)*(2x2) = 1x2 output.
    The (1x2)*(2x2) has rank 2 (schoolbook: C[0,0]=A0*B0+A1*B1, C[0,1]=A0*B2+A1*B3)
    So (3x2)*(2x2) has rank 3*2 = 6 (schoolbook).
    
    Total schoolbook: 2 + 2 + 2 + 2 blocks, each contributing 6 = 24? No wait.
    Actually schoolbook for 3x4*4x4 is 3*4*4 = 48.
    
    Let me think differently. The schoolbook has 48 terms. Each term corresponds
    to a unique (i,j,k) triple. The question is whether we can do better.
    
    Known results: The tensor rank of matrix multiplication M_{n,m,p} is at most
    n*m*p (schoolbook). For small cases, the exact minimal rank is often unknown
    but bounded below by various measures.
    
    For our purposes, let's implement a practical search:
    1. Generate schoolbook solution.
    2. Try to find a lower-rank solution by:
       a. Randomly initializing a rank-r decomposition and using gradient-free
          optimization to minimize the error.
       b. If we find an exact solution (error < 1e-10), accept it.
    
    Given the 150s time limit, we'll be conservative and try a few strategies.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def _schoolbook():
        """Generate the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return U, V, W
    
    def _verify(U, V, W, tol=1e-9):
        """
        Verify that the decomposition is correct.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        should equal 1 if (i,j,k) corresponds to a valid multiplication term,
        which it always does for schoolbook. For general decompositions,
        we need to check that the reconstructed tensor matches the standard
        matrix multiplication tensor.
        
        The standard tensor for M_{n,m,p} has T[i,j,k] = 1 for all valid (i,j,k)
        where i in [0,n), j in [0,m), k in [0,p). Wait, that's not right.
        
        Actually, the matrix multiplication tensor is defined as:
        T[i, j, k] = delta_{i,j,k} where the tensor represents the bilinear map.
        
        Let me reconsider. The tensor T in M_{n,m,p} is a 3-tensor of size
        (n*m) x (m*p) x (p*n) where T[a,b,c] = 1 if the indices correspond
        to a valid term in the multiplication, 0 otherwise.
        
        More precisely, for matrix multiplication C = A*B where A is n x m,
        B is m x p, C is n x p:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
        T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
        
        So in flattened form:
        U index: i*m + j (for A element at row i, col j)
        V index: j*p + k (for B element at row j, col k)  
        W index: k*n + i (for C element at row i, col k) -- using the given convention W[k*n+i, r]
        
        T[a, b, c] = 1 if there exist i,j,k such that:
          a = i*m + j, b = j*p + k, c = k*n + i
        and the same j is used in both a and b.
        
        So T[a,b,c] = 1 iff:
          a = i*m + j for some i,j
          b = j'*p + k for some j',k
          c = k'*n + i' for some k',i'
          AND j = j' AND k = k' AND i = i'
        
        Equivalently: T[a,b,c] = 1 iff the j-index of a equals the j-index of b,
        the k-index of b equals the k-index of c, and the i-index of c equals
        the i-index of a.
        """
        rank = U.shape[1]
        # Reconstruct the tensor
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need to check this equals the standard tensor.
        
        # Build the standard tensor as a sparse representation
        # Standard tensor: T_std[a,b,c] = 1 if (a,b,c) is a valid triple
        # a = i*m + j, b = j*p + k, c = k*n + i
        
        # For efficiency, check a random sample or all entries
        # For small sizes, check all
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    error += (val - 1.0) ** 2
        
        # Also check that T[a,b,c] = 0 for invalid triples
        # This is expensive, so check a sample
        num_samples = min(1000, n*m*p*10)
        for _ in range(num_samples):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            # Check if this is a valid triple
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            is_valid = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if is_valid:
                error += (val - 1.0) ** 2
            else:
                error += val ** 2
        
        return error < tol
    
    def _verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy operations."""
        rank = U.shape[1]
        # Use einsum to compute the tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        # But that's rank-many 3D arrays. Instead, use einsum.
        
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be memory intensive for large rank, but our sizes are small.
        
        # For our sizes: n*m=12, m*p=16, p*n=12, rank up to 48
        # T is 12 x 16 x 12 = 2304 entries, manageable.
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build standard tensor
        T_std = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_std[a, b, c] = 1.0
        
        error = np.sum((T - T_std) ** 2)
        return error < tol, error
    
    # Generate schoolbook solution
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p
    best_error = 0.0
    
    # Verify schoolbook
    valid, err = _verify_fast(best_U, best_V, best_W)
    if not valid:
        # Should not happen, but if it does, we have a problem
        pass
    best_error = err
    
    # Try to find a lower-rank decomposition
    # Strategy 1: Try ranks from best_rank-1 down to some minimum
    # For each rank, try random initialization and local search
    
    def _random_init(rank, seed=None):
        """Randomly initialize factor matrices."""
        if seed is not None:
            random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        # Scale to make entries O(1)
        U /= np.sqrt(rank) + 1e-10
        V /= np.sqrt(rank) + 1e-10
        W /= np.sqrt(rank) + 1e-10
        return U, V, W
    
    def _local_search(U, V, W, max_iters=100, tol=1e-10, time_limit=10.0):
        """
        Local search to minimize the error of the decomposition.
        Uses a simple gradient-free approach: perturb one factor at a time.
        """
        start = time.time()
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        
        current_error, _ = _verify_fast(U, V, W)
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            if current_error < tol:
                break
            
            # Try perturbing each factor
            improved = False
            for factor_name in ['U', 'V', 'W']:
                if factor_name == 'U':
                    F = U
                elif factor_name == 'V':
                    F = V
                else:
                    F = W
                
                rows, cols = F.shape
                # Try random perturbations
                for _ in range(5):
                    F_new = F.copy()
                    # Perturb a random entry
                    r_idx = random.randint(0, rows - 1)
                    c_idx = random.randint(0, cols - 1)
                    F_new[r_idx, c_idx] += random.gauss(0, 0.1)
                    
                    if factor_name == 'U':
                        test_U, test_V, test_W = F_new, V, W
                    elif factor_name == 'V':
                        test_U, test_V, test_W = U, F_new, W
                    else:
                        test_U, test_V, test_W = U, V, F_new
                    
                    new_error, _ = _verify_fast(test_U, test_V, test_W)
                    if new_error < current_error - 1e-12:
                        if factor_name == 'U':
                            U = F_new
                        elif factor_name == 'V':
                            V = F_new
                        else:
                            W = F_new
                        current_error = new_error
                        improved = True
                        break
            if not improved:
                break
        
        return U, V, W, current_error
    
    # Try to find lower-rank solutions
    # We'll try a few different strategies:
    # 1. Start from schoolbook and try to zero out some columns
    # 2. Random initialization with local search
    
    # Strategy 1: Try to reduce rank by setting some columns to zero
    # and seeing if the remaining columns still span the tensor
    # This is essentially checking if the schoolbook has redundant terms.
    
    # For now, let's try random initialization for lower ranks
    min_rank_to_try = max(1, best_rank - 20)  # Try up to 20 ranks below schoolbook
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U_rand, V_rand, W_rand = _random_init(target_rank, seed=random.randint(0, 1000000))
            U_ls, V_ls, W_ls, err_ls = _local_search(U_rand, V_rand, W_ls if False else W_rand, 
                                                       max_iters=50, time_limit=5.0)
            
            if err_ls < 1e-9:
                # Found a valid lower-rank decomposition!
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                best_rank = target_rank
                best_error = err_ls
                break
        
        if best_rank < best_rank:  # This won't happen, but for clarity
            break
    
    # Convert to lists for output
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
