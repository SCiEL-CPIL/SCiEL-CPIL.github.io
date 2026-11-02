"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110  # Leave 40s safety margin
    
    # Trivial decomposition of rank n*m*p
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank_trivial
    
    # Build target tensor for verification
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition using integer arithmetic."""
        U_arr = np.array(U_mat, dtype=np.int64)
        V_arr = np.array(V_mat, dtype=np.int64)
        W_arr = np.array(W_mat, dtype=np.int64)
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.array_equal(T_comp, T_target)
    
    # Try structured block-diagonal decompositions
    # For each row i of A, the sub-problem is 1xm * mxp -> 1xp
    # Tensor entries: T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-1 term can cover entries where U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] are all nonzero
    
    # Attempt: group by (i,k) pairs. For fixed i and k, j varies over 0..m-1
    # T[i*m+j, j*p+k, k*n+i] = 1 for all j
    # Can we cover these m entries with one rank-1 term?
    # Need U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j, and 0 elsewhere
    # If W[k*n+i,r] = 1, then need U[i*m+j,r]*V[j*p+k,r] = 1 for all j
    # and U[i*m+j',r]*V[j'*p+k',r] = 0 for all other (j',k')
    # This requires U[i*m+j,r] = 1 for all j and V[j*p+k,r] = 1 for all j
    # But then for k' != k, V[j*p+k',r] must be 0, which is fine if V only has entries at k
    # But U[i*m+j,r] = 1 for all j means for other i', U[i'*m+j,r] must be 0
    # So: U[i*m+j,r] = 1 for fixed i, all j; V[j*p+k,r] = 1 for fixed k, all j; W[k*n+i,r] = 1
    # Check: for (i', j', k') != (i, j, k):
    #   If i' != i: U[i'*m+j',r] = 0, so product is 0. Good.
    #   If i' == i but k' != k: V[j'*p+k',r] = 0 (since V only nonzero at k), product 0. Good.
    #   If i' == i, k' == k but j' != j: U[i*m+j',r] = 1, V[j'*p+k,r] = 1, W[k*n+i,r] = 1
    #     Product = 1, but T[i*m+j', j'*p+k, k*n+i] = 1 only if j' is consistent.
    #     Actually T[i*m+j', j'*p+k, k*n+i] = 1 for ALL j'! So this is correct!
    
    # So for each (i,k) pair, we can use ONE rank-1 term to cover all m entries!
    # Total rank = n * p = 3 * 7 = 21
    
    # Let's verify this construction:
    # r = i * p + k
    # U[i*m+j, i*p+k] = 1 for all j
    # V[j*p+k, i*p+k] = 1 for all j  
    # W[k*n+i, i*p+k] = 1
    
    # Check T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # a = i*m+j, b = j'*p+k', c = k''*n+i'
    # sum_{i,k} U[i*m+j, i*p+k] * V[j'*p+k', i*p+k] * W[k''*n+i', i*p+k]
    # U[i*m+j, i*p+k] = 1 iff i matches (the row index in a)
    # V[j'*p+k', i*p+k] = 1 iff k' == k (the col index in b matches the k in r)
    # W[k''*n+i', i*p+k] = 1 iff i' == i and k'' == k
    # So the sum is 1 iff i' == i, k' == k, and k'' == k, i.e., i==i', k==k', k==k''
    # But we also need j == j' for T to be nonzero!
    # The sum gives 1 when i==i', k==k', k==k'' but regardless of j,j'
    # T[i*m+j, j'*p+k, k*n+i] = 1 iff j == j' (and i==i', k==k)
    # So this overcounts: it gives 1 for all j,j' when i==i', k==k', but T only has 1 when j==j'
    
    # This doesn't work either. The issue is that U and V both have 1s for all j, so they don't
    # enforce j == j'.
    
    # The fundamental issue: to enforce j == j', we need the product U[i*m+j,r] * V[j'*p+k,r]
    # to be 1 when j == j' and 0 when j != j'. This requires the vectors U[:,r] and V[:,r]
    # to be "orthogonal" in a multiplicative sense, which for {0,1} values means they must
    # have disjoint support or one must be a scaled version... but with 0/1, disjoint support
    # means product is always 0.
    
    # So with 0/1 entries, we can't enforce j == j' unless each rank-1 term covers exactly
    # one j value. This means we need at least m rank-1 terms per (i,k) pair, giving
    # rank >= n * m * p = 105.
    
    # But what if we use non-0/1 entries? E.g., U[i*m+j,r] = alpha_j, V[j*p+k,r] = beta_j
    # Then product = alpha_j * beta_j. For this to be 1 when j=j' and 0 when j!=j',
    # we'd need alpha_j * beta_{j'} = 0 for j != j', which means alpha_j or beta_{j'} is 0.
    # Same issue.
    
    # What about using multiple rank-1 terms per (i,k) with cancellations?
    # E.g., for (i,k), use terms r1, r2 with:
    # U[i*m+0,r1]=1, U[i*m+1,r1]=0, U[i*m+0,r2]=0, U[i*m+1,r2]=1
    # V[0*p+k,r1]=1, V[1*p+k,r1]=0, V[0*p+k,r2]=0, V[1*p+k,r2]=1
    # Then for j=0: U*V = 1*1 + 0*0 = 1 (for r1), 0*0 + 1*0 = 0 (for r2)
    # For j=1: U*V = 0*0 + 1*1 = 1 (for r2), 1*0 + 0*1 = 0 (for r1)
    # This works! But it uses 2 terms for m=2, i.e., m terms total.
    
    # So the trivial decomposition is optimal for 0/1 entries. With integer entries,
    # we might do better with cancellations, but it's complex.
    
    # Let's try a different approach: use the known Strassen-like ideas or
    # structured random search with better initialization.
    
    # First, verify the trivial decomposition
    assert verify(U, V, W), "Trivial decomposition failed verification!"
    
    # Try random search with structured initialization
    np.random.seed(42)
    
    # Try ranks from 104 down to 90 with random sparse decompositions
    for target_rank in range(best_rank - 1, max(best_rank - 15, 85), -1):
        if time.time() > deadline:
            break
        
        # Try a few random integer decompositions with sparse structure
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Generate random sparse integer matrices with density ~0.3
            U_rand = np.zeros((n*m, target_rank), dtype=np.int64)
            V_rand = np.zeros((m*p, target_rank), dtype=np.int64)
            W_rand = np.zeros((p*n, target_rank), dtype=np.int64)
            
            for r in range(target_rank):
                # Sparse random with values in {-1, 0, 1}
                U_rand[:, r] = np.random.randint(-1, 2, n*m)
                V_rand[:, r] = np.random.randint(-1, 2, m*p)
                W_rand[:, r] = np.random.randint(-1, 2, p*n)
            
            if verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                best = {'U': U_rand.tolist(), 'V': V_rand.tolist(), 'W': W_rand.tolist()}
                best_rank = target_rank
                break  # Move to next lower rank
        
        if time.time() > deadline:
            break
    
    return best
# EVOLVE-BLOCK-END
