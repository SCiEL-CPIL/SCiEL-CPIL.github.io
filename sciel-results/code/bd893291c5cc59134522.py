import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    deadline = time.time() + 130  # Safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        M = np.einsum('ar,cr->acr', U, W)
        T_approx = np.einsum('acr,br->abc', M, V)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
    
    def try_round_and_verify(U, V, W):
        """Try rounding to half-integers and verify"""
        for scale in [2, 4, 8, 16, 32]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None
    
    def rank_of(U, V, W):
        """Return the number of non-zero rank-1 terms"""
        # Count columns that are non-zero in all three
        nz = 0
        for r in range(U.shape[1]):
            if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
                nz += 1
        return nz
    
    # Strategy 1: Block structure with Strassen-like additions
    # For 2x5x5, think of it as: A is 2x5, B is 5x5
    # Group columns of A into blocks, group rows of B into blocks
    # Use the fact that (A1+A2)(B1+B2) = A1B1 + A1B2 + A2B1 + A2B2
    # can be done in 3 multiplications: (A1+A2)(B1+B2), A1B1, A2B2
    
    # For our specific tensor, let's try structured patterns
    
    # Strategy 2: Search over small integer/half-integer coefficient patterns
    # The tensor has a very specific structure. Let's exploit it.
    
    # Key observation: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # So for fixed j: T[i*m+j, j*p+k, k*n+i] = 1 for all i,k
    # This means each "j-slice" is a 2x5x5 tensor
    
    # Let's try to find decompositions where each rank-1 term has
    # a specific sparsity pattern
    
    # Strategy 3: Use the fact that we can combine terms
    # If we have two terms that differ only in one index, we might combine them
    
    # Let's do a systematic search over structured families
    
    # Family 1: Terms grouped by j (column of A, row of B)
    # For each j, we need to cover the 2x5 sub-tensor
    # The j-th slice: T[i*m+j, j*p+k, k*n+i] for i in 0..1, k in 0..4
    
    # For a fixed j, the sub-tensor has rank at most 2*5 = 10 (schoolbook for that slice)
    # But we can share across j's
    
    # Let's try: for each j, use a 2x5x5 decomposition
    # 2x5x5 schoolbook is rank 10, but maybe we can do better
    
    # Actually, let's think about this differently.
    # The full tensor T has a specific structure. Let's look at its slices.
    
    # For fixed i (row of A): T[i*m+j, j*p+k, k*n+i] = 1
    # This is a 5x5 matrix (j,k) with all 1s - rank 1!
    
    # For fixed k (col of B): T[i*m+j, j*p+k, k*n+i] = 1
    # This is a 10x10 matrix (a,c) where a=i*m+j, c=k*n+i
    # For fixed k: entries are 1 when c=k*n+i and a=i*m+j for any j
    # So it's a permutation-like structure
    
    # Let's try a different approach: use the matrix multiplication interpretation
    # C = A * B where A is 2x5, B is 5x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is T[i,j,k] = A[i,j] * B[j,k] * C[k,i] (in tensor form)
    
    # We want to write this as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Key: if we can find a factorization where the j-dependence is separable,
    # we can reduce the rank.
    
    # Let's try: U[i,j,r] = sum over some structure
    # V[j,k,r] = ...
    # W[k,i,r] = ...
    
    # Approach: Try rank-r decompositions where r < 50
    # Use a structured initialization based on the tensor's properties
    
    # For each candidate rank, try multiple structured initializations
    
    def structured_search(target_rank, max_tries=50):
        """Search for a decomposition of given rank using structured patterns"""
        for attempt in range(max_tries):
            if time.time() > deadline - 15:
                return None
                
            # Different structured initializations
            if attempt % 5 == 0:
                # Pattern: group by j
                # For each j, try to use fewer terms
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Distribute rank across j values
                terms_per_j = target_rank // m
                rem = target_rank % m
                
                col = 0
                for j in range(m):
                    t = terms_per_j + (1 if j < rem else 0)
                    for r in range(t):
                        # For this j, cover some i,k pairs
                        # Use a pattern where we combine i and k
                        i1 = (r * 2) % n
                        i2 = (r * 2 + 1) % n
                        k1 = (r * 2) % p
                        k2 = (r * 2 + 1) % p
                        
                        # Try different coefficient patterns
                        if r % 3 == 0:
                            U[i1*m+j, col] = 1
                            U[i2*m+j, col] = 1
                            V[j*p+k1, col] = 1
                            V[j*p+k2, col] = 1
                            W[k1*n+i1, col] = 1
                            W[k2*n+i2, col] = 1
                        elif r % 3 == 1:
                            U[i1*m+j, col] = 1
                            V[j*p+k1, col] = 1
                            W[k1*n+i1, col] = 1
                            U[i2*m+j, col] = -1
                            V[j*p+k2, col] = -1
                            W[k2*n+i2, col] = -1
                        else:
                            U[i1*m+j, col] = 0.5
                            U[i2*m+j, col] = 0.5
                            V[j*p+k1, col] = 0.5
                            V[j*p+k2, col] = 0.5
                            W[k1*n+i1, col] = 0.5
                            W[k2*n+i2, col] = 0.5
                        col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 5 == 1:
                # Pattern: Strassen-like on blocks
                # Split 5 into 2+3
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Block 1: first 2 columns/rows
                # Block 2: last 3 columns/rows
                # Use Strassen: M1=(A1+A2)(B1+B2), M2=A1(B2-B1), M3=A2B1
                # C1 = M1 + M2 - M3, C2 = M1 + M3
                
                col = 0
                # M1 = (A1+A2)(B1+B2)
                for i in range(n):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        # A1+A2: sum of first 2 and last 3 columns
                        for j in range(2):
                            U[i*m+j, col] = 1
                        for j in range(2, 5):
                            U[i*m+j, col] = 1
                        for j in range(2):
                            V[j*p+k, col] = 1
                        for j in range(2, 5):
                            V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
                
                # M2 = A1*(B2-B1)
                if col < target_rank:
                    for i in range(n):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            for j in range(2):
                                U[i*m+j, col] = 1
                            for j in range(2):
                                V[j*p+k, col] = -1
                            for j in range(2, 5):
                                V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                
                # M3 = A2*B1
                if col < target_rank:
                    for i in range(n):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            for j in range(2, 5):
                                U[i*m+j, col] = 1
                            for j in range(2):
                                V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 5 == 2:
                # Random with structure
                np.random.seed(42 + attempt)
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                for r in range(target_rank):
                    # Pick a random j
                    j = np.random.randint(0, m)
                    # Pick i and k patterns
                    i_pat = np.random.randint(0, 2, size=n)
                    k_pat = np.random.randint(0, p, size=p)
                    
                    for i in range(n):
                        U[i*m+j, r] = float(np.random.choice([-1, 0, 1]))
                    for k in range(p):
                        V[j*p+k, r] = float(np.random.choice([-1, 0, 1]))
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = float(np.random.choice([-1, 0, 1]))
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 5 == 3:
                # Try to combine schoolbook terms
                # If two schoolbook terms can be merged
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Start with schoolbook and try to merge
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            # Try to combine with previous terms
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                # Now try to merge: if U[:,a] + U[:,b] has a simpler pattern
                # This is heuristic - just check if we can use fewer terms
                # by having some terms be sums
                
                if verify(U, V, W):
                    return U, V, W
                    
            else:
                # Gradient-based from structured init
                np.random.seed(100 + attempt)
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
                
                # Add schoolbook structure
                if target_rank <= rank_sb:
                    idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                    U += U_sb[:, idx] * 0.5
                    V += V_sb[:, idx] * 0.5
                    W += W_sb[:, idx] * 0.5
                
                # Quick gradient descent
                lr = 0.05
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    M = np.einsum('ar,cr->acr', U, W)
                    T_approx = np.einsum('acr,br->abc', M, V)
                    R = T - T_approx
                    res = np.sum(R**2)
                    if res < 1e-10:
                        break
                    
                    grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                    grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                    grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                rounded = try_round_and_verify(U, V, W)
                if rounded is not None:
                    return rounded
                
                if verify(U, V, W):
                    return U, V, W
        
        return None
    
    # Try ranks from 49 down, but focus on structured searches
    # The minimum possible rank is at least max(nm, mp, pn) = 25
    # But realistically for this structure, it's probably higher
    
    # Try a range of ranks
    for target_rank in range(rank_sb - 1, 24, -1):
        if time.time() > deadline - 25:
            break
            
        result = structured_search(target_rank, max_tries=30)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = target_rank
                continue
    
    # Final: if we found something better, return it; else schoolbook
    return best_result
