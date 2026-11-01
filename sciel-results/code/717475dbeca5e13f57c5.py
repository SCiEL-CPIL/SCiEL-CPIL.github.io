"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # We'll try to find a low-rank decomposition
    # Start with a guess and use iterative methods
    
    # First, let's try known constructions
    # For M_{2,2,p}, try rank = 2*p + 2 or similar
    
    # Let me try a systematic approach:
    # The tensor has 56 non-zero entries.
    # Each rank-1 term can cover at most one non-zero entry if they're "independent"
    # but cleverly chosen terms can cover multiple entries.
    
    # Actually, a rank-1 tensor u⊗v⊗w has u[a]*v[b]*w[c] non-zero where all three are non-zero.
    # For it to match T, we need u[a]*v[b]*w[c] = T[a,b,c] for all a,b,c.
    # Since T is 0-1, and we want exact decomposition, we need the sum to equal T exactly.
    
    # Let me try a different construction.
    # For each (i,k), the contributions come from j=0 and j=1:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Think of it as: for fixed i, we have a vector of length 2 (over j) for A,
    # and for each k, a vector of length 2 (over j) for B.
    
    # Let me try rank = 2*p = 28:
    # For each k and i, use one term? That's 2*14=28 terms.
    # Term (i,k): u has 1 at position i*2+0 and i*2+1? No, that doesn't work.
    
    # Let me try: for each j and k, use one term. That's 2*14=28 terms.
    # Term (j,k): u[j*2... no, u is indexed by (i,j).
    
    # Term for (j,k): u[i,j] = 1 for all i? No...
    
    # Let me think again. A rank-1 term is u⊗v⊗w.
    # u has length 4 (indices (i,j)), v has length 28 (indices (j,k)), w has length 28 (indices (k,i)).
    
    # For a term to contribute to T[i,j,k] = 1, we need u[i*2+j]*v[j*14+k]*w[k*2+i] = 1 (or sum to 1).
    
    # Idea: For each (i,k), create a term that handles both j=0 and j=1.
    # But a single rank-1 term can't handle two different j values unless u and v are structured.
    
    # Alternative: Use the fact that for fixed j, the sub-tensor over (i,k) is a 2×14×14 tensor
    # where T[i*2+j, j*14+k, k*2+i] = 1.
    # For j=0: T[2i, k, 2k+i] = 1
    # For j=1: T[2i+1, 14+k, 2k+i] = 1
    
    # Each of these is a 2×14×14 sub-tensor (in the relevant indices).
    # The rank of each such sub-tensor... 
    
    # For j=0: the non-zero entries are at (2i, k, 2k+i) for i∈{0,1}, k∈{0,...,13}
    # That's 28 entries. Can we decompose this with fewer than 28 rank-1 terms?
    
    # A rank-1 term in this sub-tensor: u[i]*v[k]*w[i',k'] where it's non-zero only at specific (i,k).
    # Actually, in the full tensor, a rank-1 term u⊗v⊗w restricted to j=0 slice:
    # u[2i]*v[k]*w[2k+i] for i∈{0,1}, k∈{0,...,13}
    
    # This is a 2×14 matrix (over i,k) where entry (i,k) = u[2i]*v[k]*w[2k+i].
    # This is not a general rank-1 matrix; it's a structured one.
    
    # This is getting too complex analytically. Let me just do a numerical search.
    
    # Use a simple approach: start with random initialization and use ALS or similar.
    # But we need exact integer/half-integer solutions.
    
    # Let me try known small ranks and check if they work.
    
    # Try rank 28:
    # Construction: For each (j,k), define a term.
    # u[i*2+j] = 1 for all i (so u has 1s at positions j and 2+j... wait)
    # u is length 4: positions 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1)
    # For fixed j: u[0*2+j] and u[1*2+j] should be non-zero.
    # If j=0: u[0] and u[2] non-zero
    # If j=1: u[1] and u[3] non-zero
    
    # v[j*14+k] = 1, others 0
    # w[k*2+i] = 1 for all i? No, w should be 1 at position k*2+i for the specific i.
    
    # Hmm, this doesn't quite work for a single term.
    
    # Let me just try to verify what rank is achievable by searching.
    
    # Given time constraints, let me return the schoolbook for now but try a quick search.
    
    # Actually, let me try a specific construction for rank 2p = 28.
    # For each k (0 to 13), we need to compute:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is 2 outputs, each a dot product of 2-vectors.
    # For each k, this is a 2×2×1 "matrix multiplication" which has rank 4 (schoolbook) 
    # but actually the rank of the 2×2×1 tensor is 2 (since it's just two independent dot products... no).
    
    # Wait, the tensor for computing [A[0,0]*B[0,k] + A[0,1]*B[1,k], A[1,0]*B[0,k] + A[1,1]*B[1,k]]
    # has 4 non-zero entries: (0,0,0,k), (0,1,0,k), (1,0,0,k), (1,1,0,k) in (i,j,k) coords.
    # Actually in our indexing: (0*2+0, 0*14+k, k*2+0), (0*2+1, 1*14+k, k*2+0),
    # (1*2+0, 0*14+k, k*2+1), (1*2+1, 1*14+k, k*2+1)
    # = (0, k, 2k), (1, 14+k, 2k), (2, k, 2k+1), (3, 14+k, 2k+1)
    
    # Can we decompose these 4 entries with 2 rank-1 terms?
    # Term 1: covers (0,k,2k) and (2,k,2k+1)? 
    #   u[0]*v[k]*w[2k] = 1, u[2]*v[k]*w[2k+1] = 1, others 0.
    #   So u[0]=a, u[2]=b, v[k]=c, w[2k]=d, w[2k+1]=e with a*c*d=1, b*c*e=1.
    #   And u[1]=u[3]=0, v[other]=0, w[other]=0.
    #   This works! With a=b=c=d=e=1.
    
    # Term 2: covers (1,14+k,2k) and (3,14+k,2k+1)?
    #   u[1]*v[14+k]*w[2k] = 1, u[3]*v[14+k]*w[2k+1] = 1, others 0.
    #   So u[1]=a, u[3]=b, v[14+k]=c, w[2k]=d, w[2k+1]=e with a*c*d=1, b*c*e=1.
    #   With a=b=c=d=e=1, this works!
    
    # So for each k, we need 2 terms, total 2*14 = 28 terms!
    
    # Let me verify:
    # Term for (k, type=0): u[0]=1, u[2]=1, v[k]=1, w[2k]=1, w[2k+1]=1, all else 0.
    #   This contributes to T[0,k,2k] += 1*1*1 = 1 ✓
    #   T[2,k,2k+1] += 1*1*1 = 1 ✓
    #   All other entries get 0 since either u, v, or w is 0.
    
    # Term for (k, type=1): u[1]=1, u[3]=1, v[14+k]=1, w[2k]=1, w[2k+1]=1, all else 0.
    #   This contributes to T[1,14+k,2k] += 1*1*1 = 1 ✓
    #   T[3,14+k,2k+1] += 1*1*1 = 1 ✓
    
    # Total: 28 terms, each with integer entries (0 or 1).
    # This gives rank 28!
    
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 0: covers j=0 contributions for both i values
        col = k  # first term for this k
        # u[0]=1 (i=0,j=0), u[2]=1 (i=1,j=0)
        U[0][col] = 1
        U[2][col] = 1
        # v[k]=1 (j=0,k)
        V[0 * p + k][col] = 1
        # w[2k]=1 (k,i=0), w[2k+1]=1 (k,i=1)
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        
        # Term 1: covers j=1 contributions for both i values
        col = p + k  # second term for this k
        # u[1]=1 (i=0,j=1), u[3]=1 (i=1,j=1)
        U[1][col] = 1
        U[3][col] = 1
        # v[14+k]=1 (j=1,k)
        V[1 * p + k][col] = 1
        # w[2k]=1 (k,i=0), w[2k+1]=1 (k,i=1)
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
    
    # Verify the decomposition
    # Check that sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx] for all a,b,c_idx
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute the reconstructed tensor
    # T_recon[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
    # Use einsum for efficiency
    T_recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    # Build the true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1
    
    if not np.array_equal(T_recon, T_true):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
