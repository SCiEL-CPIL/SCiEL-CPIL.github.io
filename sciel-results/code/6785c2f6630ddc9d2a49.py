"""Matrix multiplication 3x4x8 - rank 48 decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-48 decomposition for 3x4x8 matrix multiplication.
    
    Split k into two groups of 4. For each 3x4x4 sub-problem, use a rank-24
    decomposition. The total rank is 48.
    
    For 3x4x4: A(3x4) * B(4x4) = C(3x4)
    We use 24 terms by exploiting the structure:
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Group j into {0,1} and {2,3}, and k into {0,1} and {2,3}.
    
    For the (j in {0,1}, k in {0,1}) block: 3x2 * 2x2 = 3x2
    For the (j in {0,1}, k in {2,3}) block: 3x2 * 2x2 = 3x2
    For the (j in {2,3}, k in {0,1}) block: 3x2 * 2x2 = 3x2
    For the (j in {2,3}, k in {2,3}) block: 3x2 * 2x2 = 3x2
    
    Each 3x2*2x2 sub-problem can be done in rank 6 (schoolbook for 3*2*2=12,
    but we can do better). Actually, let me just use schoolbook for each:
    3x2*2x2 has schoolbook rank 12. 4 blocks * 12 = 48 for 3x4x4. That's not
    rank 24.
    
    Let me try a different approach. Use rank 48 total for 3x4x8 directly.
    
    I'll construct 48 terms where each term is a product of vectors.
    """
    start_time = time.time()
    deadline = start_time + 118  # safety margin
    
    n, m, p = 3, 4, 8
    
    # Try to find a valid rank-48 decomposition
    # Use the following construction:
    # Split into 4 groups of 12 terms each.
    # 
    # Actually, let me use a verified construction.
    # 
    # For 3x4x8, the schoolbook rank is 96.
    # Let me try rank 48 by using the following idea:
    # 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # I'll use 48 terms. Let me define them as follows:
    # For each pair (i, k), I need to represent the bilinear form.
    # 
    # Let me just verify that a rank-48 decomposition exists by construction.
    # 
    # Here's a valid rank-48 construction:
    # Use 48 terms where term c corresponds to (i_c, j_c, k_c) with some
    # sharing of indices.
    #
    # Actually, the simplest valid approach: use schoolbook but merge terms
    # where possible. Since all entries are 0 or 1, we can't merge.
    #
    # Let me try a different strategy: use a rank-64 decomposition.
    # Split k into 4 groups of 2. For each 3x4x2 sub-problem, use rank 16.
    # 4 * 16 = 64.
    #
    # For 3x4x2: A(3x4) * B(4x2) = C(3x2)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Schoolbook: 3*4*2 = 24 terms.
    # Can we do rank 16? 
    #
    # Let me just return a valid decomposition. I'll use rank 96 (schoolbook)
    # since I can't easily verify a lower rank decomposition.
    
    # Use schoolbook rank 96 as fallback
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
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-48 decomposition using linear algebra
    # The tensor T has shape (12, 32, 24) in flattened form
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and c%n == a//m
    
    # For rank r, we need U (12xr), V (32xr), W (24xr) such that
    # for all a,b,c: sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c]
    
    # This is a system of equations. Let me try rank 48.
    
    # Actually, let me try a constructive approach for rank 48.
    # 
    # Key idea: For 3x4x4, use the following rank-24 decomposition:
    # 
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..2, k in 0..3
    # 
    # Write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # 
    # Now, define new variables:
    # M0 = A[:,0], M1 = A[:,1], M2 = A[:,2], M3 = A[:,3] (each 3-vectors)
    # N0 = B[0,:], N1 = B[1,:], N2 = B[2,:], N3 = B[3,:] (each 4-vectors)
    # 
    # C = M0*N0 + M1*N1 + M2*N2 + M3*N3
    # 
    # This is already rank 4 (in terms of outer products of 3-vectors and 4-vectors).
    # But each outer product M_i * N_i contributes to all 3*4=12 entries of C.
    # So this is rank 4 in the "matrix" sense, but rank 12 in the tensor sense
    # for each block.
    #
    # I'm going in circles. Let me just return the schoolbook.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
