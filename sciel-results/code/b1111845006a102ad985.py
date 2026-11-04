import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 130
    
    # We'll use a deterministic approach: start with trivial decomposition
    # and try to find a rank-reduction pattern
    
    # The trivial decomposition has rank 144. We need to find 14 terms
    # that can be eliminated by expressing them as linear combinations.
    
    # Strategy: Use a known rank reduction for matrix multiplication.
    # For 4x4x9, we can save rank by exploiting the structure.
    
    # Let's try a constructive approach: generate 130 rank-1 terms
    # that exactly decompose the multiplication tensor.
    
    # We'll use a systematic enumeration with pruning.
    # Start with all 144 trivial terms, then find 14 that are linearly dependent.
    
    # First, build the trivial decomposition
    trivial_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                trivial_terms.append((i, j, k))
    
    # Now we need to find a subset of 130 terms (with possibly modified coefficients)
    # that still decompose the tensor.
    
    # Alternative approach: Use a randomized search to find a valid rank-130 decomposition.
    # Initialize U, V, W with random values and project onto the constraint manifold.
    
    # Actually, let's use a more reliable method: start with the trivial decomposition
    # and attempt to nullify 14 columns by finding linear dependencies.
    
    U_full = np.zeros((n * m, n * m * p), dtype=float)
    V_full = np.zeros((m * p, n * m * p), dtype=float)
    W_full = np.zeros((p * n, n * m * p), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_full[u_idx, col] = 1.0
                V_full[v_idx, col] = 1.0
                W_full[w_idx, col] = 1.0
                col += 1
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this is a valid multiplication term, else 0.
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this case? No...)
    
    # Wait, I need to reconsider. The matrix multiplication tensor is:
    # M[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the standard formulation is that we want:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be such that:
    # T[i,j,k] = 1 if j is the summation index linking A and B
    # But in the standard decomposition, each rank-1 term corresponds to one (i,j,k) triple.
    
    # The correct tensor for matrix multiplication C = AB where A is n×m, B is m×p, C is n×p is:
    # T[i,k,j] = 1 (where i indexes rows of C/A, k indexes columns of C/B, j is the inner dimension)
    # But in our convention, W[k*n+i, r] suggests the output is indexed by (k,i) where k is col, i is row.
    
    # Let me verify: the output C has shape (n, p), with C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T should satisfy: T[i,j,k] = 1 for all valid (i,j,k)
    # Because each product A[i,j]*B[j,k] contributes to C[i,k].
    
    # So the target tensor is all ones of shape (n, m, p) when flattened appropriately.
    
    # Actually no. The tensor decomposition is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # And we want T[i,j,k] = 1 for all i,j,k (since every combination contributes).
    
    # Wait, that's not right either. Let me think again.
    
    # For matrix multiplication, the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as a tensor contraction. The tensor T has entries:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k in range.
    
    # Yes, the tensor is all-ones of shape (n, m, p).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # With the trivial decomposition, each term has U=V=W=1 at the corresponding position.
    
    # Now, to reduce rank, we need to find 130 vectors such that the sum still equals 1 everywhere.
    
    # This is a hard problem. Let me try a different approach: use a known result.
    
    # For 4x4x9, the minimum rank is believed to be around 130-135.
    # Let me try to construct one using a greedy method.
    
    # Approach: Start with 130 random rank-1 terms and solve for the coefficients.
    
    np.random.seed(42)
    
    # Generate 130 random rank-1 terms
    U_rand = np.random.randn(n * m, target_rank)
    V_rand = np.random.randn(m * p, target_rank)
    W_rand = np.random.randn(p * n, target_rank)
    
    # The constraint is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p = 144 equations with 130*3 = 390 unknowns.
    # But it's nonlinear (cubic). We can linearize by fixing two of the three matrices.
    
    # Fix U and V, solve for W. For each (k,i), we have:
    # sum_r W[k*n+i, r] * (sum over j of U[i*m+j, r] * V[j*p+k, r]) = 1
    # Wait, that's not right. Let me expand:
    # For fixed (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Group by W index: for fixed (k,i), sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
    # But j varies, so this doesn't decouple cleanly.
    
    # Better: For fixed (i,k), we have multiple j values:
    # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system where W[k*n+i, :] is shared across all j.
    
    # Let's set up the linear system. For each (i,k) pair, and for each j, we have one equation.
    # Total equations: n * p * m = 4 * 9 * 4 = 144
    # Unknowns: for each (k,i), we have target_rank unknowns in W. Total: p * n * target_rank = 9 * 4 * 130 = 4680
    # This is underdetermined, so we can find a solution.
    
    # But we also need U and V to be consistent. This is getting complicated.
    
    # Let me try a simpler approach: use the trivial decomposition and find 14 columns
    # that are linear combinations of the others, then eliminate them.
    
    # Build the full 144-column decomposition
    U_144 = np.zeros((n * m, n * m * p), dtype=float)
    V_144 = np.zeros((m * p, n * m * p), dtype=float)
    W_144 = np.zeros((p * n, n * m * p), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_144[u_idx, col] = 1.0
                V_144[v_idx, col] = 1.0
                W_144[w_idx, col] = 1.0
                col += 1
    
    # Now, try to find 14 columns that can be removed.
    # A column r can be removed if its contribution can be expressed as a linear combination
    # of other columns.
    
    # For column r to be removable, we need to find coefficients c_s for s != r such that:
    # U[:, r] = sum_{s != r} c_s * U[:, s]
    # V[:, r] = sum_{s != r} c_s * V[:, s]  
    # W[:, r] = sum_{s != r} c_s * W[:, s]
    
    # But this is too restrictive. Instead, we need the tensor contribution to be preserved:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # If we remove column r, we need to adjust other columns to compensate.
    
    # This is a complex optimization problem. Let me try a pragmatic approach:
    # Return a rank-130 decomposition found by a simple heuristic.
    
    # Heuristic: Remove 14 columns that have the smallest "importance" and renormalize.
    
    # Actually, let me just return the trivial decomposition for now, since finding
    # a valid rank-130 decomposition is non-trivial.
    
    # But the task asks for rank-minimal. Let me try one more approach.
    
    # Use a fixed pattern: for 4x4x9, we can save rank by sharing terms across
    # different (i,k) pairs when j is the same.
    
    # Let me construct a rank-130 decomposition using a structured approach.
    
    # I'll use the following pattern: for each j, group the terms by j.
    # There are m=4 values of j, each with n*p=36 terms. Total 144.
    # If we can save 3-4 terms per j-group, we get to 130.
    
    # For a fixed j, the terms are (i,k) pairs. The constraint is:
    # For each (i,k): sum over terms in this j-group of U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # (assuming other j-groups don't contribute to this (i,k) - but they do in general)
    
    # This is still complex. Let me just return the trivial rank-144 decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
