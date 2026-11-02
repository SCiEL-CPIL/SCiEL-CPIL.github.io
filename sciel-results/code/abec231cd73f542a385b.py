"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    def verify_fast(U, V, W):
        """
        Fast verification using tensor structure.
        T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        where a=i*m+j, b=j'*p+k', c=k''*n+i'
        Expected: T=1 iff i==i', j==j', k==k''
        """
        U = np.asarray(U, dtype=np.int64)
        V = np.asarray(V, dtype=np.int64)
        W = np.asarray(W, dtype=np.int64)
        r = U.shape[1]
        
        # Reshape to exploit structure
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)  
        # W: (p*n, r) -> (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # For each (i, j, k), check:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        # For all other (i2, j2, k2) != (i,j,k), the corresponding tensor entry should be 0
        
        # The tensor T has indices (i2, j2, k2) where:
        # a = i2*m + j2, b = j*p + k2, c = k*n + i
        # T[i2,j2,k2] = sum_r U[i2,j2,r] * V[j2,k2,r] * W[k2,i,r]
        
        # We need T[i,j,k]=1 for all i,j,k and T[i2,j2,k2]=0 otherwise
        
        # Check all 105 ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U3[i, j] * V3[j, k] * W3[k, i])
                    if val != 1:
                        return False
        
        # Check zeros: for each (i2,j2,k2) with (i2,j2,k2) != (i,j,k) for all valid (i,j,k)
        # Actually we need: T[i2,j2,k2] = 0 unless i2==j2's row matches... 
        # The condition is: T[i2,j2,k2]=1 iff there exists i,j,k such that i2==i, j2==j, k2==k
        # But i2,j2,k2 are independent indices. The tensor entry T[i2,j2,k2] 
        # corresponds to the (i2,j2,k2) position in the 3D tensor.
        # It should be 1 only when i2==j2 is "compatible" - actually no.
        
        # Let me re-read: a = i*m+j (input index), b = j'*p+k' (output index), c = k''*n+i' (output index)
        # T[a,b,c] = 1 iff i==i', j==j', k==k''
        # So for fixed (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i -> T=1
        # For any other combination of (a,b,c) not matching this pattern -> T=0
        
        # So we need to check that for all (a,b,c) where the implied (i,j,k) from a,
        # (j',k') from b, (k'',i') from c don't satisfy i==i', j==j', k==k'', T=0
        
        # Efficient check: for each (i,j,k), the "one" position is (a=i*m+j, b=j*p+k, c=k*n+i)
        # All other positions should be zero.
        
        # Total positions: n*m * m*p * p*n = 105*35*21 = 77175
        # Ones: 105
        # Zeros: 77070
        
        # This is too many to check individually. Let's use a smarter approach.
        # Compute the full tensor and check.
        
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U * (V^T * W^T)^T)[a,c] ... no
        
        # Actually: for fixed a,c: T[a,b,c] as function of b is U[a,:] * (V*W^T)[b,:]... 
        # Let M[b,c] = sum_r V[b,r]*W[c,r] = (V @ W.T)[b,c]
        # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = sum_b' ... no
        # T[a,b,c] = U[a,:] dot (V[b,:] * W[c,:]) element-wise
        
        # Alternative: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For fixed a: T[a,:,c] = sum_r U[a,r]*V[:,r]*W[c,r] = (U[a,:] * (V * W[c,:].reshape(1,-1))).T... 
        # Actually: T[a,b,c] = sum_r U[a,r] * (V[:,r] * W[c,r])
        # = U[a,:] dot (V * W[c].T)[:,b]... 
        # Let Q = V * W.T  # (m*p, p*n) where Q[b,c] = sum_r V[b,r]*W[c,r]
        # Then T[a,b,c] is NOT simply related to Q because of the U factor.
        
        # Better: T = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
        # This is a rank-r decomposition of the 3D tensor.
        
        # For verification, compute T for all non-trivial entries.
        # Given time constraints, let's just check the 105 ones and a random sample of zeros.
        
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if val != 1:
                        return False
        
        # Check some zeros (sample)
        rng = np.random.RandomState(42)
        num_zero_checks = min(2000, (n*m)*(m*p)*(p*n) - 105)
        for _ in range(num_zero_checks):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue
            val = np.sum(U[a] * V[b] * W[c])
            if val != 0:
                return False
        
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def verify_full(U, V, W):
        """Full verification - check all tensor entries."""
        U = np.asarray(U, dtype=np.int64)
        V = np.asarray(V, dtype=np.int64)
        W = np.asarray(W, dtype=np.int64)
        r = U.shape[1]
        
        # Check all 105 ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if val != 1:
                        return False
        
        # Check all zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                val = np.sum(U[a] * V[b] * W[c])
                                if val != 0:
                                    return False
        return True

    # Start with schoolbook
    U_sb, V_sb, W_sb, base_rank = schoolbook()
    current_best = (base_rank, U_sb, V_sb, W_sb)
    
    # Strategy: Try to find lower rank by exploiting structure
    # Key insight: The multiplication tensor for (n,m,p) has rank at most n*m*p
    # but can often be reduced.
    
    # For small cases, we can try systematic search with constraint propagation
    
    def try_rank_reduction(start_rank, max_attempts=500):
        """
        Try to find a decomposition of rank r by starting from schoolbook
        and trying to merge columns.
        """
        if time.time() > deadline - 20:
            return None
        
        best = None
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            r = start_rank
            # Start with schoolbook and try to perturb
            U = np.zeros((n * m, r), dtype=np.int64)
            V = np.zeros((m * p, r), dtype=np.int64)
            W = np.zeros((p * n, r), dtype=np.int64)
            
            # Initialize with a subset of schoolbook columns
            cols_used = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if cols_used >= r:
                            break
                        U[i * m + j, cols_used] = 1
                        V[j * p + k, cols_used] = 1
                        W[k * n + i, cols_used] = 1
                        cols_used += 1
                    if cols_used >= r:
                        break
                if cols_used >= r:
                    break
            
            # Fill remaining columns with small random values
            for col in range(cols_used, r):
                U[:, col] = np.random.randint(-1, 2)
                V[:, col] = np.random.randint(-1, 2)
                W[:, col] = np.random.randint(-1, 2)
            
            if verify_fast(U, V, W):
                if verify_full(U, V, W):
                    if best is None or r < best[0]:
                        best = (r, U.copy(), V.copy(), W.copy())
        
        return best

    # Try various ranks below schoolbook
    for r in range(base_rank - 1, max(base_rank - 30, 70), -1):
        if time.time() > deadline - 30:
            break
        found = try_rank_reduction(r, max_attempts=100)
        if found is not None:
            if r < current_best[0]:
                current_best = found
                break
    
    # Also try a different approach: use the fact that for rectangular matrices,
    # there might be structural savings
    
    def structured_search(rank, num_trials=200):
        """More structured random search."""
        if time.time() > deadline - 20:
            return None
        
        best = None
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            
            # Use a mix of structured and random initialization
            for col in range(rank):
                # Some columns are structured (like schoolbook)
                if col < 50:
                    # Map to a (i,j,k) triple
                    idx = col
                    k = idx % p
                    idx //= p
                    j = idx % m
                    idx //= m
                    i = idx
                    if i < n and j < m and k < p:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                else:
                    # Random small integers
                    U[:, col] = np.random.randint(-1, 2)
                    V[:, col] = np.random.randint(-1, 2)
                    W[:, col] = np.random.randint(-1, 2)
            
            if verify_fast(U, V, W):
                if verify_full(U, V, W):
                    if best is None or rank < best[0]:
                        best = (rank, U.copy(), V.copy(), W.copy())
        
        return best

    # Try structured search for lower ranks
    for r in range(base_rank - 1, max(base_rank - 25, 75), -1):
        if time.time() > deadline - 25:
            break
        found = structured_search(r, num_trials=50)
        if found is not None:
            if r < current_best[0]:
                current_best = found
                break
    
    rank, U, V, W = current_best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
