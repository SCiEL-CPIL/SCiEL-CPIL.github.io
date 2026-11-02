"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Trivial schoolbook decomposition (rank = n*m*p = 105)
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((mp, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((pn, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    def winograd_decompose(n_dim, m_dim, p_dim, alpha, beta, gamma):
        """
        Winograd-style decomposition for matrix multiplication.
        Embeds into larger problem and uses structured factors.
        """
        # Create the base matrices for Winograd
        # For n×m × m×p, we want to reduce the rank
        # Using the 1D Winograd algorithm on the m dimension
        
        # Pad to power of 2 if needed
        def next_pow2(x):
            return 1 << (x - 1).bit_length()
        
        M = next_pow2(m_dim)
        N = next_pow2(n_dim)
        P = next_pow2(p_dim)
        
        # Create DFT-like transform matrices
        def dft_matrix(size):
            W = np.zeros((size, size), dtype=np.complex128)
            for i in range(size):
                for j in range(size):
                    W[i, j] = np.exp(-2j * np.pi * i * j / size)
            return W
        
        # For Winograd, we use specific transform matrices
        # Simplified: use Hadamard-like transforms
        
        def hadamard_matrix(size):
            if size == 1:
                return np.array([[1]], dtype=np.float64)
            H_half = hadamard_matrix(size // 2)
            H = np.zeros((size, size), dtype=np.float64)
            H[:size//2, :size//2] = H_half
            H[:size//2, size//2:] = H_half
            H[size//2:, :size//2] = H_half
            H[size//2:, size//2:] = -H_half
            return H
        
        H_M = hadamard_matrix(M) / np.sqrt(M)
        H_N = hadamard_matrix(N) / np.sqrt(N)
        H_P = hadamard_matrix(P) / np.sqrt(P)
        
        # Build the decomposition
        # The tensor is sum over j of e_i ⊗ e_j ⊗ e_k
        # In Winograd form, we factor through the transform space
        
        rank = M * N * P  # This is still the trivial rank, but let's see if we can do better
        # Actually, for Winograd to help, we need the structure
        
        # Let's try a different approach: direct structured decomposition
        # For each pair (i, k), the j-dimension gives a rank-1 contribution
        # Total rank = n * p * m = 105 (same as trivial)
        
        # Try a mixed approach: group by i first
        # For each i, we have a m×p sub-problem
        # Each such sub-problem has rank m*p, but they share structure
        
        # Better: use the fact that the tensor has a Kronecker-like structure
        # T[i,j,k] = 1 if the indices match the matrix multiplication pattern
        
        # Let's try a rank-n*p decomposition for each "block" and see if we can merge
        # For fixed i, the tensor entries are T[i*m+j, j*p+k, k*n+i]
        # This is a permutation of the m×p block
        
        # Actually, let's try to use the Strassen-like idea
        # For 3×5 × 5×7, we can potentially use:
        # - Split 3 = 2+1, 5 = 4+1, 7 = 4+3
        # - Apply rank-7 to 2×2 and rank-2 to 1×1 parts
        
        # This is getting complex. Let's return the trivial for now
        return None
    
    # Try to find lower-rank decompositions using structured search
    # The key insight: the matrix multiplication tensor for m×n has rank at most m*n
    # but the bilinear complexity can be less
    
    # For our specific case (3,5,7), let's try:
    # 1. A direct structured decomposition based on the index structure
    # 2. Random search with smarter initialization
    
    rng = np.random.default_rng(42)
    
    # Try a structured decomposition based on the Kronecker product structure
    # The tensor T can be written as sum over j of (sum over i of e_i) ⊗ e_j ⊗ (sum over k of e_k)
    # but with the specific index mapping
    
    # Let's try a rank-n*p = 21 decomposition (one per (i,k) pair)
    # For each (i,k), we need to handle the j-dimension
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # This means for fixed (i,k), the entries are at positions:
    # (i*m+j, j*p+k, k*n+i) for j=0,...,m-1
    # This is a rank-m vector in the "j" direction
    
    # So the total rank would be n*p*m = 105 again
    
    # The only way to reduce rank is to find linear dependencies among these rank-1 terms
    
    # Let's try a different structured approach:
    # Use the fact that we can combine terms across different j values
    
    # For the 3×5 case:
    # Row indices: 0,1,2 (for i=0,1,2) times 5 + j
    # So rows: 0-4 (i=0), 5-9 (i=1), 10-14 (i=2)
    # Col indices: j*7 + k, so cols: 0-6 (j=0), 7-13 (j=1), 14-20 (j=2), 21-27 (j=3), 28-34 (j=4)
    # Out indices: k*3 + i, so out: 0-2 (k=0), 3-5 (k=1), 6-8 (k=2), 9-11 (k=3), 12-14 (k=4), 15-17 (k=5), 18-20 (k=6)
    
    # Let's try to find a decomposition of rank < 105
    # by exploiting the block structure
    
    # A rank-35 decomposition might be possible if we can merge 3-fold in one dimension
    
    # Try: for each j, use a rank-7 factorization of the n×p = 3×7 sub-tensor
    # But that gives rank 5*7 = 35
    
    # The 3×7 sub-tensor for fixed j has entries at:
    # (i, k) -> position (i, k) in the sub-space
    # This is a permutation matrix, which has rank 21
    
    # So 5 * 21 = 105, no improvement
    
    # The only hope is to find cross-term cancellations
    
    # Let's try a computational approach: set up the system of equations
    # and try to solve for a low-rank decomposition
    
    # For rank r, we need to find U (15×r), V (35×r), W (21×r)
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # This is a system of 15*35*21 = 11025 equations in r*(15+35+21) = 71r unknowns
    
    # For r < 105, this is underdetermined only if 71r > 11025, i.e., r > 155
    
    # So for r <= 105, the system is overdetermined!
    # This means rank 105 might be minimal, or we need a different approach
    
    # Wait, that's not quite right. The tensor has many zero entries.
    # The actual number of non-zero entries is n*m*p = 105
    
    # So we have 105 equations in 71r unknowns
    # For r = 2, we have 142 unknowns > 105 equations, so it's underdetermined
    
    # Let's try to solve the linear system for small r
    
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import lsqr
    
    def try_rank(r, max_iter=1000):
        """Try to find a rank-r decomposition by solving the linear system."""
        if time.time() > deadline - 10:
            return None
        
        # Build the system of equations
        # For each non-zero entry (a,b,c) of T_target, we need:
        # sum_s U[a,s]*V[b,s]*W[c,s] = T_target[a,b,c]
        
        # This is a bilinear system in the variables
        # To linearize, we can fix two of the three matrices
        
        # Fix U and V, solve for W (linear in W)
        # Or use alternating minimization
        
        # Let's try alternating minimization
        # Initialize with random small values
        U = rng.standard_normal((nm, r)) * 0.1
        V = rng.standard_normal((mp, r)) * 0.1
        W = rng.standard_normal((pn, r)) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Update U: for each a, solve U[a,:] using current V, W
            # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
            # For fixed a: sum_s U[a,s] * (sum_c' V[b,s]*W[c',s] where c' = c)
            # This is still bilinear
            
            # Actually, let's use a simpler approach: gradient descent
            # on the loss L = ||T_computed - T_target||^2
            
            # Compute current tensor
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((T_comp - T_target)**2)
            
            if loss < 1e-12:
                return U, V, W
            
            # Compute gradients
            # dL/dU[a,s] = 2 * sum_{b,c} (T_comp[a,b,c] - T_target[a,b,c]) * V[b,s] * W[c,s]
            # This is expensive. Let's use a simpler random search instead.
            
            # Perturb one matrix
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                idx = rng.integers(0, nm)
                U[idx, :] += rng.standard_normal(r) * 0.01
            elif mat == 1:
                idx = rng.integers(0, mp)
                V[idx, :] += rng.standard_normal(r) * 0.01
            else:
                idx = rng.integers(0, pn)
                W[idx, :] += rng.standard_normal(r) * 0.01
        
        return None
    
    # Try various ranks with the alternating approach
    candidate_ranks = [104, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, max_iter=5000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # If we haven't found anything better, also try some more structured approaches
    # based on the specific dimensions
    
    # For 3×5×7, try a decomposition that exploits the fact that 5 = 3+2
    # and 7 = 3+2+2
    
    # Try a block decomposition
    if best_rank == rank_trivial and time.time() < deadline - 20:
        # Try rank = n*p + m*p + n*m = 21 + 35 + 15 = 71
        # This corresponds to a Strassen-like decomposition
        
        # For 2×2 matrix multiplication, Strassen uses rank 7
        # For our case, we can try to embed into a 4×4 problem
        # and use Strassen's algorithm, then project back
        
        # This is complex. Let's just try random search with more iterations
        # for ranks 70-104
        
        for target_rank in range(70, 105):
            if time.time() > deadline - 15:
                break
            
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                
                U = rng.integers(-2, 3, size=(nm, target_rank)).astype(np.float64)
                V = rng.integers(-2, 3, size=(mp, target_rank)).astype(np.float64)
                W = rng.integers(-2, 3, size=(pn, target_rank)).astype(np.float64)
                
                if verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
            
            if best_rank < rank_trivial:
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
