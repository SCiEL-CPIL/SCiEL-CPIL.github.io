"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    rank = n * m * p
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Strategy: try to find linear dependencies among columns
    # If column c can be written as sum of alpha_l * column_l for l != c,
    # we can remove column c and adjust others.
    # 
    # For matrix multiplication tensor, the columns are indexed by (i,j,k).
    # U column (i,j,k) = e_{i,j}, V column = e_{j,k}, W column = e_{k,i}
    #
    # A dependency: sum_{(i,j,k) in S} alpha_{i,j,k} * (e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}) = 0
    # This means: for each (a,b,c), sum_{(i,j,k) in S: i=a,j=b,k=c} ... 
    # Actually the tensor is sum of rank-1 terms, and a dependency means
    # sum alpha_t * u_t ⊗ v_t ⊗ w_t = 0
    #
    # Let's try: for a fixed k, can we relate columns with same k?
    # Or for fixed (i,j), relate across k?
    #
    # Better approach: use the fact that for matrix multiplication,
    # the tensor has structure. Let's try random linear combinations.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find dependencies by checking if columns are linearly dependent
    # in the tensor space. The tensor lives in R^{(n*m) x (m*p) x (p*n)}
    # which has dimension n*m*p = 96. Each rank-1 term is a vector in this space.
    # So we have 96 vectors in a 96-dimensional space - they form a basis!
    # Wait, that means rank is exactly 96 for the schoolbook... 
    #
    # Actually no. The multiplication tensor is a specific tensor, not just any
    # set of 96 vectors. The rank is the minimum number of rank-1 tensors needed.
    # The schoolbook gives 96, but the true rank might be less.
    #
    # For 2x2x2, rank is 7 (Strassen). For 3x3x3, rank is at most 23.
    # For 3x4x8, we need to search.
    
    # Approach: randomized search for lower rank
    # Start with schoolbook, try to "absorb" one column into others
    
    def try_reduce(U_arr, V_arr, W_arr, current_rank, seed):
        """Try to reduce rank by 1 by finding a dependency."""
        np.random.seed(seed)
        n_rm, m_rm, p_rm = n, m, p
        
        # Try each column as the one to remove
        for target in range(current_rank):
            # Get the target column's (i,j,k) index
            # Column index c corresponds to (i,j,k) where c = i*m*p + j*p + k
            i_t = target // (m_rm * p_rm)
            j_t = (target // p_rm) % m_rm
            k_t = target % p_rm
            
            # We want: u_t ⊗ v_t ⊗ w_t = sum_{l != t} alpha_l * u_l ⊗ v_l ⊗ w_l
            # This is a system of n*m*p equations in current_rank-1 unknowns
            # 
            # u_t = e_{i_t, j_t} (one-hot in R^{n*m})
            # v_t = e_{j_t, k_t} (one-hot in R^{m*p})
            # w_t = e_{k_t, i_t} (one-hot in R^{p*n})
            #
            # The tensor equation: for each (a, b, c) in [n*m] x [m*p] x [p*n]:
            #   [a == i_t*m + j_t] * [b == j_t*p + k_t] * [c == k_t*n + i_t]
            #     = sum_{l != t} alpha_l * U[a,l] * V[b,l] * W[c,l]
            #
            # Since U, V, W are one-hot in the schoolbook, U[a,l] = 1 iff a == i_l*m + j_l
            # Similarly for V and W.
            #
            # So the RHS is nonzero only when a = i_l*m + j_l, b = j_l*p + k_l, c = k_l*n + i_l
            # for some l. And in that case, the value is alpha_l.
            #
            # The LHS is 1 only when (a,b,c) = (i_t*m+j_t, j_t*p+k_t, k_t*n+i_t), else 0.
            #
            # So we need: alpha_l = 1 if (i_l,j_l,k_l) = (i_t,j_t,k_t) and l != t... 
            # But that's impossible since each (i,j,k) appears exactly once.
            #
            # Wait, this means in the schoolbook basis, no single column is a linear
            # combination of others! The 96 vectors are linearly independent.
            #
            # To get a lower rank, we need to use a DIFFERENT set of rank-1 tensors,
            # not just the schoolbook ones. We need to find new vectors.
        
        return None
    
    # Alternative: use known results or structured search
    # For 3x4x8, let's try to use block structure
    # 
    # Actually, let me try a different approach: use the fact that we can
    # represent the multiplication using fewer multiplications if we
    # exploit algebraic identities.
    #
    # For now, let's try a randomized approach where we perturb the
    # schoolbook decomposition and look for lower-rank approximations
    # that are still exact.
    
    # Let's try: can we use rank < 96?
    # The border rank might be lower, but we need exact rank.
    
    # Try a structured approach: group by k values
    # For each k, we have n*m = 12 columns. Total 8*12 = 96.
    # Can we share some structure across k values?
    
    # Let me try: for the 3x4 part, use Strassen-like tricks
    # 3x4 = 3x2 + 3x2, and 4x8 = 2x4 + 2x4
    # 
    # Actually, let's just try to verify if rank 96 is minimal by
    # checking if we can do better with a small search.
    
    # Given time constraints, let's return the schoolbook if we can't
    # find something better quickly.
    
    # Try: use the identity that (A*B) can be computed with
    # fewer than n*m*p multiplications for some dimensions.
    
    # For 3x4x8, let's try rank 88 or so by removing 8 columns
    # if they're dependent in some extended sense.
    
    # Actually, the cleanest approach: try random linear combinations
    # of the 96 schoolbook terms to see if we can get a valid
    # decomposition with fewer terms.
    
    # Let's use a different strategy: 
    # The tensor T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # We want T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r with R < 96
    #
    # This is equivalent to finding a rank-R decomposition.
    # 
    # One approach: fix some of the u_r, v_r, w_r to have structure
    # and solve for the rest.
    
    # Given the complexity, let's just return schoolbook for now
    # but format it properly.
    
    # Convert to list of lists with integer entries
    def to_list(arr):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in arr]
    
    # Actually, let's try one more thing: check if we can reduce by
    # using the fact that for fixed i, the m*p entries have structure.
    
    # For a 3x4x8 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as: for each i, C[i,:] = A[i,:] * B
    # A[i,:] is a 1x4 row, B is 4x8.
    # 
    # Using Strassen for 1x4 * 4x8? That's just a matrix-vector product.
    # 1x4 * 4x8 requires 4*8 = 32 multiplications schoolbook.
    # Can we do better? For a row times a matrix, it's just 32 mults.
    # 
    # Total: 3 * 32 = 96. So schoolbook is 96.
    # 
    # The question is whether we can share multiplications across i values.
    # 
    # Let's try: compute some linear combinations of rows of A first,
    # multiply by B, then combine.
    
    # Example: Let S1 = A[0,:] + A[1,:], S2 = A[0,:] - A[1,:], etc.
    # Then C[0,k] = (S1[k] + S2[k]) / 2 where S1[k] = sum_j (A[0,j]+A[1,j])*B[j,k]
    # This uses 2 multiplications per k instead of 4, but we need to
    # also get C[1,k] = (S1[k] - S2[k]) / 2
    # So for i=0,1 together: we use 2*8 = 16 mults instead of 2*4*8 = 64?
    # No wait: S1[k] = sum_j (A[0,j]+A[1,j])*B[j,k] requires 4 mults per k
    # (one for each j), so 4*8 = 32 mults for S1, same for S2, total 64.
    # That's not better.
    
    # The issue is that each "combined" row still needs 4*8 mults.
    # 
    # To actually save, we need to share the B[j,k] values across i.
    # 
    # Let me think differently. The tensor rank for 3x4x8...
    # 
    # Known: rank(2x2x2) = 7, rank(3x3x3) <= 23, rank(4x4x4) <= 49
    # For rectangular: rank(2x3x3) <= ?, etc.
    # 
    # For 3x4x8, I'll try to see if we can get below 96.
    # 
    # Let's just return the schoolbook decomposition properly formatted.
    
    result = {
        "U": [[int(U[i,j]) for j in range(rank)] for i in range(n*m)],
        "V": [[int(V[i,j]) for j in range(rank)] for i in range(m*p)],
        "W": [[int(W[i,j]) for j in range(rank)] for i in range(p*n)]
    }
    
    return result

# EVOLVE-BLOCK-END
