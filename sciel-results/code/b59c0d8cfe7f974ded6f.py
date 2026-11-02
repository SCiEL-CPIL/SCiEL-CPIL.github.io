"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank-56 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # MPS/SVD-based rank reduction:
    # The multiplication tensor T[i,j,k] = delta can be viewed as a matrix
    # For each fixed k, we get a matrix A_k[i,j] that should equal U[:,k] * V[:,k]^T contribution
    # We try to reduce rank by finding linear dependencies in the tensor slices
    
    # Strategy: Try to find a lower-rank decomposition by SVD on reshaped tensor
    # Reshape U, V, W to exploit structure
    
    # The tensor T has shape (n*m, m*p, p*n) with T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # This is a very sparse structure. Let's try to find a compressed representation.
    
    # Approach: Use the fact that for each j, the coupling between (i,k) pairs is simple.
    # For fixed j: U[i*m+j, :] has nonzeros only where the column corresponds to (i,j,k) for some k
    # V[j*p+k, :] has nonzeros only where column corresponds to (i,j,k) for some i
    # W[k*n+i, :] has nonzeros only where column corresponds to (i,j,k) for some j
    
    # Try Strassen-like block decomposition for the 2x4 and 4x7 structure
    # Split into 2x2 blocks where possible
    
    # n=2, m=4, p=7
    # Split m=4 into two blocks of 2: m1=2, m2=2
    # For each block, we can use Strassen on 2x2 * 2x7 subproblems
    
    # Actually, let's try a direct SVD-based approach on the tensor
    # Reshape the tensor to (n*m, m*p) for each k-slice, but that's not quite right.
    
    # Better: The tensor T[i,j,k] can be seen as a 3D array.
    # Flatten to 2D: T2D[i*m+j, k*n+i] for fixed j? No, that's not a clean factorization.
    
    # Let's try: for each pair (i,j), the vector over k is U[i*m+j,:] * V[j*p:,:]*W[:,k*n+i]
    # This is complex. Let's use a different approach.
    
    # Try random linear combinations to find rank reductions
    # If columns c1 and c2 are linearly dependent in a certain sense, we can merge them.
    
    # For the schoolbook, column (i,j,k) has:
    # U: 1 at position i*m+j
    # V: 1 at position j*p+k  
    # W: 1 at position k*n+i
    
    # Two columns (i1,j1,k1) and (i2,j2,k2) are "compatible" for merging if
    # we can find scalars a,b such that a*U1+b*U2, a*V1+b*V2, a*W1+b*W2 still
    # gives valid contributions. This is generally not possible for arbitrary pairs.
    
    # However, if i1=i2, j1=j2, k1≠k2: then U1=U2, but V1≠V2 and W1≠W2
    # We can't simply merge these.
    
    # Let's try a computational approach: build the full tensor and use SVD
    # The tensor T has shape (n*m, m*p, p*n) = (8, 28, 14)
    # Total size: 8*28*14 = 3136, which is manageable.
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try SVD on different unfoldings to find rank reductions
    # Unfold along mode 1: T1 shape (n*m, m*p*p*n)
    T1 = T.reshape(n*m, m*p*p*n)
    U_svd, s, Vt = np.linalg.svd(T1, full_matrices=False)
    rank1 = np.sum(s > 1e-10)
    
    # Unfold along mode 2: T2 shape (m*p, n*m*p*n)
    T2 = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
    U_svd2, s2, Vt2 = np.linalg.svd(T2, full_matrices=False)
    rank2 = np.sum(s2 > 1e-10)
    
    # Unfold along mode 3: T3 shape (p*n, n*m*m*p)
    T3 = T.transpose(2,0,1).reshape(p*n, n*m*m*p)
    U_svd3, s3, Vt3 = np.linalg.svd(T3, full_matrices=False)
    rank3 = np.sum(s3 > 1e-10)
    
    # The minimum of these gives a lower bound on the rank
    min_unfold_rank = min(rank1, rank2, rank3)
    
    # If we can achieve this rank with a valid decomposition, great.
    # But SVD on unfolding doesn't directly give a valid tensor decomposition.
    # We need to verify.
    
    # For now, let's try to construct a decomposition of rank min_unfold_rank
    # using the SVD factors appropriately.
    
    # Actually, the SVD of a single unfolding gives the best rank-r approximation
    # in Frobenius norm, but not necessarily an exact decomposition.
    
    # Since our tensor has 0/1 entries and is very sparse, let's check if
    # the rank can be reduced below 56.
    
    # Known result: rank of matrix multiplication tensor [n,m,p] is at most
    # n*m*p, and for small cases often much less.
    # For [2,4,7], literature suggests rank around 30-40 might be possible.
    
    # Let's try a more systematic approach: use the SVD to find a candidate
    # rank-r decomposition and verify it exactly.
    
    # For each candidate rank r from min_unfold_rank to 56, try to find
    # U, V, W of rank r that exactly decompose T.
    
    # This is a hard problem in general, but for this small case we can try.
    
    # Alternative: use the fact that the tensor is a sum of 56 rank-1 terms
    # and try to find linear dependencies among them.
    
    # The rank-1 terms are e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Two terms are linearly dependent only if they're identical, which they're not.
    # But combinations might be dependent.
    
    # Let's just return the schoolbook for now if we can't find better quickly.
    # But first, let's check if a simple block decomposition helps.
    
    # Split m=4 into m1=2, m2=2
    # For block 1 (j=0,1): multiply 2x2 by 2x7 -> rank 2*2*7=28
    # For block 2 (j=2,3): multiply 2x2 by 2x7 -> rank 2*2*7=28
    # Total: 56, same as before. No gain.
    
    # Try splitting p=7 into p1=3, p2=4
    # For block 1 (k=0,1,2): 2x4 * 4x3 -> rank 2*4*3=24
    # For block 2 (k=3,4,5,6): 2x4 * 4x4 -> rank 2*4*4=32
    # Total: 56. Still same.
    
    # The schoolbook is already optimal for this splitting strategy.
    
    # Let's try to use the SVD result to see if we can do better.
    # If rank1 < 56, there might be a way.
    
    if min_unfold_rank < best_rank and time.time() < deadline - 10:
        # Try to construct a decomposition of rank min_unfold_rank
        # This is non-trivial, so let's just report the potential improvement
        # and return schoolbook if we can't verify.
        pass
    
    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
