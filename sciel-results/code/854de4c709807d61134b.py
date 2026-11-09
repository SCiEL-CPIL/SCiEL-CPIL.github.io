"""Matrix multiplication 4x4x8 - structured low-rank tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (all-ones tensor)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Strategy: Use block structure.
    # Partition indices to exploit the all-ones structure.
    # 
    # For the all-ones tensor of size n x m x p, we can use the following:
    # Group the p dimension into blocks of size b, and use rank-b decompositions.
    #
    # Key insight: The all-ones matrix of size a x b has rank 1 (outer product of ones).
    # So if we can factor the tensor as a product of all-ones blocks, we get low rank.
    #
    # Approach: Use the fact that for the all-ones tensor T[i,j,k]=1:
    # We can write 1 = sum_c f_c(i) * g_c(j) * h_c(k)
    # where we need to find low-rank factorizations.
    #
    # The minimum rank of the all-ones tensor n x m x p is max(n,m,p) when one dimension
    # can be factored. But for rectangular cases, we can do better with structured approaches.
    #
    # For n=4, m=4, p=8:
    # One approach: group j into pairs and k into pairs.
    #
    # Let's use a direct construction:
    # Since T[i,j,k] = 1 for all i,j,k, we can use:
    # 1 = sum_{c=1}^{R} u_c(i,j) * v_c(j,k) * w_c(k,i)
    #
    # Simple construction using the all-ones structure:
    # For each "diagonal" pattern, use rank-1 terms.
    #
    # Actually, the simplest low-rank approach for all-ones tensor:
    # T[i,j,k] = sum_{c=1}^{n} (1/n) * [1 for all i] * [1 for all j] * [1 for all k]
    # No, that gives rank 1 but T[i,j,k] = 1/n, not 1.
    #
    # Let me think differently. We need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Since the tensor is all-ones, we can use the following structured decomposition:
    # Partition into blocks based on i and k (both size 4).
    #
    # For the 4x4 subproblem in (i,k) with j as the "free" index:
    # T[i,j,k] = 1 for all i,j,k
    #
    # Method: Use rank-1 terms where each term covers a subset of (i,j,k).
    # 
    # Direct construction:
    # Since all entries are 1, we can use the following:
    # Let R = n * p = 32 (or less with optimization)
    # For each (i0, k0) pair, create a term that contributes to all j:
    #   U[i*m+j, c] = 1 if i == i0 else 0
    #   V[j*p+k, c] = 1 for all j,k  (but this doesn't work as V depends on j,k)
    #
    # Better approach: exploit that V[j*p+k] can be 1 for all j,k in a group.
    #
    # Let's use the following construction with rank = n * p = 32:
    # For each (i, k) pair (16 pairs), and for each j... no, that's too many.
    #
    # Alternative: Use the fact that we can separate the j index.
    # T[i,j,k] = 1 = (1/m) * sum_{j'} 1  (trivially)
    #
    # Let me try a different structured approach:
    # Rank = n * p = 32
    # For c = i * p + k (i in 0..n-1, k in 0..p-1):
    #   U[i'*m+j', c] = 1 if i' == i else 0
    #   V[j'*p+k', c] = 1 if k' == k else 0  
    #   W[k''*n+i'', c] = m if k'' == k and i'' == i else 0
    #
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Only c = i*p+k contributes: 1 * 1 * m = m = 4, not 1.
    # Scale by 1/m: W = 1 if match, then sum = 1. 
    #
    # So with rank = n*p = 32:
    # For c = i*p + k:
    #   U[i'*m+j', c] = 1 if i' == i else 0
    #   V[j'*p+k', c] = 1 if k' == k else 0
    #   W[k''*n+i'', c] = 1.0/m if k'' == k and i'' == i else 0
    #
    # This gives rank 32. Can we do better?
    
    # Even better: use the all-ones structure more aggressively.
    # Since V[j*p+k] doesn't depend on i, and U[i*m+j] doesn't depend on k,
    # we can potentially reduce rank.
    #
    # Try rank = max(n, p) = 8:
    # For c = 0..7:
    #   U[i*m+j, c] = 1 for all i,j  (constant in i,j)
    #   V[j*p+k, c] = 1 if k == c else 0
    #   W[k*n+i, c] = 1/(n) for all k,i  (constant)
    # Check: sum_c U * V * W = sum_c 1 * (1 if k==c else 0) * 1/n = 1/n = 1/4. Not 1.
    # Scale W by n: W = 1 for all, then sum = 1. But V only nonzero for one k.
    # sum_c 1 * delta(k,c) * 1 = 1. Yes!
    #
    # So rank = p = 8:
    # For c = 0..p-1:
    #   U[i*m+j, c] = 1 for all i, j
    #   V[j*p+k, c] = 1 if k == c else 0
    #   W[k*n+i, c] = 1 for all k, i
    #
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c 1 * (1 if k==c else 0) * 1
    # = 1 (only c=k contributes)
    # YES! This works with rank = p = 8!
    
    # Let's verify this construction:
    rank = p  # 8
    U = np.ones((n * m, rank))  # All ones: U[i*m+j, c] = 1
    V = np.zeros((m * p, rank))
    for j in range(m):
        for k in range(p):
            V[j * p + k, k] = 1  # V[j*p+k, c] = 1 if c == k else 0
    W = np.ones((p * n, rank))  # All ones: W[k*n+i, c] = 1
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
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
    
    # Try to reduce rank further if time permits
    if time.time() < deadline - 10:
        # Try rank = 4 by exploiting more structure
        # Can we do rank = 4?
        # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # With rank 4, we need 4 rank-1 tensors whose product sums to all-ones.
        # This is equivalent to finding a rank-4 decomposition of the all-ones tensor 4x4x8.
        # The all-ones tensor of size a x b x c has tensor rank = max(a,b,c) in general,
        # but can be less with structured approaches.
        # For 4x4x8, the rank is at most 8 (as shown). Can it be less?
        # The border rank is at least max(n,m,p) = 8 for generic tensors,
        # but the all-ones tensor is special.
        # Actually, the all-ones tensor has rank = min(n,m,p) if we allow real entries?
        # No, let me just try to verify rank 4 doesn't work by checking if we can find such a decomposition.
        # For now, rank 8 is our best structured answer.
        pass
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    # Round to clean values
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx] - round(row[idx])) < 1e-8:
                    row[idx] = round(row[idx])
    
    return {"U": U_list, "V": V_list, "W": W_list}
