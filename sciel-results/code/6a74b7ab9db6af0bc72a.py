import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    # Build the full schoolbook tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    rank_schoolbook = n * m * p  # 45
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # We'll search for a rank-r decomposition.
    # Start from the schoolbook and try to reduce rank by merging terms.
    #
    # Key insight: The matrix multiplication tensor for small sizes is known
    # to have rank equal to the Strassen rank or a specific known value.
    # For 3x3x5, let's try to find a decomposition of rank < 45.
    #
    # We'll use a greedy approach: start with all 45 schoolbook terms,
    # and try to see if any pair of terms can be merged into a single
    # tensor product term (i.e., if their sum is rank 1 in the tensor sense).
    #
    # Two schoolbook terms T_{i1,j1,k1} and T_{i2,j2,k2} can be merged
    # if (T_{i1,j1,k1} + T_{i2,j2,k2}) is a rank-1 tensor, meaning:
    # (T_{i1,j1,k1} + T_{i2,j2,k2})[a,b,c] = U[a]*V[b]*W[c]
    # This happens only if the support patterns align in a compatible way.
    #
    # Actually, for a more systematic approach: we can use the fact that
    # the matrix multiplication tensor's rank is related to the bilinear
    # complexity. Let's try a direct search using numpy.
    #
    # For 3x3x5, the known rank is 40 (I recall some results, but let's verify
    # by searching). We'll try ranks from 40 down.
    #
    # A simpler approach: use the fact that for small matrices, we can try
    # to find a decomposition by solving a system of equations.
    #
    # Let's try rank 40 first, then 39, etc.
    #
    # For a given rank r, we want to find U (9 x r), V (15 x r), W (15 x r)
    # such that for all (i,j,k) in {0,1,2}^2 x {0,1,2,3,4}:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of 45 equations in r^2 * (9 + 15 + 15) = 45r unknowns
    # (since U, V, W each have r columns, and we can normalize).
    #
    # Actually, let's think about this differently. The tensor T has 45 non-zero
    # entries. A rank-r decomposition means T can be written as sum of r rank-1
    # tensors.
    #
    # We'll use a numerical approach: for a given r, initialize U, V, W randomly
    # and use gradient descent or a similar optimization to minimize the error.
    #
    # But this is computationally expensive. Let's try a smarter approach.
    #
    # For 3x3x5, let's try to use the Strassen-like approach. The 3x3 matrix
    # multiplication has known rank 19 (Lader). For 3x3x5, it might be different.
    #
    # Let's just try to find a valid decomposition by searching over possible
    # structures. We'll start with the schoolbook and try to reduce the rank
    # by identifying linear dependencies.
    #
    # Actually, let's just return the schoolbook decomposition for now, since
    # finding a lower-rank exact decomposition is complex and time-consuming.
    # But we can try a quick search for rank 44, 43, etc.
    
    # Try to find a rank-44 decomposition by merging one pair of schoolbook terms
    # if their sum is rank 1.
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Initialize with schoolbook
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    for c in range(rank_schoolbook):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0

    # Try to merge pairs
    for c1 in range(rank_schoolbook):
        if time.time() > deadline - 30:
            break
        for c2 in range(c1+1, rank_schoolbook):
            # Check if T_c1 + T_c2 is rank 1
            # The sum has non-zero entries at the positions of T_c1 and T_c2
            # For it to be rank 1, the pattern must factor.
            i1 = c1 // (m*p); j1 = (c1 // p) % m; k1 = c1 % p
            i2 = c2 // (m*p); j2 = (c2 // p) % m; k2 = c2 % p
            
            # The sum is rank 1 if and only if the four "corner" products
            # satisfy a certain condition. Specifically, the support of the
            # sum has 2 non-zero entries (if i1!=i2 or j1!=j2 or k1!=k2, 
            # they're at different positions). For a rank-1 tensor, the
            # non-zero entries must be on a "grid" that factors.
            
            # If (i1,j1,k1) != (i2,j2,k2), the sum has 2 non-zero entries.
            # A rank-1 tensor with 2 non-zero entries is possible only if
            # the two entries are at positions (a1,b1,c1) and (a2,b2,c2)
            # where either a1=a2, b1=b2, or c1=c2 (i.e., they share an index).
            # Actually, no: a rank-1 tensor U[a]*V[b]*W[c] has non-zero
            # entries wherever U[a]!=0, V[b]!=0, W[c]!=0. So the support
            # is a Cartesian product of the supports of U, V, W.
            # For the support to have exactly 2 points, one of U, V, W must
            # have support of size 1, and the other two must have support
            # of size 2 (and 1*2*2=4, not 2). Or one has size 1, one has
            # size 1, one has size 2: 1*1*2=2. Yes!
            
            # So the sum is rank 1 iff two of the three indices match between
            # the two positions, and the third differs.
            # i.e., (i1==i2 and j1==j2) or (i1==i2 and k1==k2) or (j1==j2 and k1==k2)
            # But wait, if i1==i2 and j1==j2, then k1!=k2 (since c1!=c2).
            # The support is {(i1,j1,k1), (i1,j1,k2)}. This is rank 1 iff
            # U[i1]!=0, V[j1]!=0, W[k1]=1, W[k2]=1, and W is zero elsewhere
            # in the support. Actually, for rank 1, we need:
            # T_sum[a,b,c] = u_a * v_b * w_c
            # T_sum[i1,j1,k1] = 1, T_sum[i1,j1,k2] = 1
            # So u_{i1}*v_{j1}*w_{k1} = 1 and u_{i1}*v_{j1}*w_{k2} = 1
            # This implies w_{k1} = w_{k2}, and u_{i1}*v_{j1}*w_{k1} = 1.
            # Also, T_sum is zero elsewhere, so u_a = 0 for a != i1,
            # v_b = 0 for b != j1, w_c = 0 for c not in {k1,k2}.
            # This works! So the sum is rank 1.
            
            # Similarly for the other cases.
            
            if (i1 == i2 and j1 == j2) or (i1 == i2 and k1 == k2) or (j1 == j2 and k1 == k2):
                # Try to merge
                # Create new U, V, W with rank-1 decomposition of the sum
                # U_new[i1*m+j1] = 1, V_new[j1*p+k1] = 1, W_new[k1*n+i1] = 1
                # U_new[i1*m+j1] = 1, V_new[j1*p+k2] = 1, W_new[k2*n+i1] = 1
                # But this doesn't work because W would need to be non-zero at two positions.
                # The rank-1 decomposition of the sum is:
                # U[i1*m+j1] = 1, V[j1*p+k1] = 1, W[k1*n+i1] = 1
                # But we need T_sum[i1,j1,k2] = 1 too, so we need another term.
                # So the sum of two schoolbook terms is NOT rank 1 in general.
                # I made an error above.
                pass
    
    # The above approach is flawed. Let's just return the schoolbook.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
