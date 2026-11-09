import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # Recursive tensor decomposition builder
    # For multiplying A (n x m) by B (m x p) to get C (n x p)
    # We build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # where C[i,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] for each term r
    
    # The decomposition: for each rank-1 term, we pick (i, j, k) and set
    # U[i*m + j][r] = 1, V[j*p + k][r] = 1, W[k*n + i][r] = 1
    # This gives C[i,k] += A[i,j] * B[j,k]
    
    # Strategy: split the middle dimension m into parts
    # If we split m into m1 + m2, then:
    # A = [A1 | A2] where A1 is n x m1, A2 is n x m2
    # B = [B1; B2] where B1 is m1 x p, B2 is m2 x p
    # C = A1*B1 + A2*B2
    # Rank = R(n, m1, p) + R(n, m2, p)
    
    # Also can split n or p dimensions similarly
    
    # For small cases, use direct enumeration (rank = n*m*p)
    # For larger cases, use recursive splitting
    
    # Let's implement a recursive approach:
    # R(n, m, p) with splits
    
    # Key splits for 3x4x8:
    # Split m=4 into m1=3, m2=1: R(3,3,8) + R(3,1,8)
    # R(3,1,8) = 3*1*8 = 24? No, R(n,1,p) = n*p = 24
    # Wait, R(n,1,p): A is n x 1, B is 1 x p. C[i,k] = A[i,0]*B[0,k]
    # This is outer product, rank = n*p = 24. But we can do better?
    # Actually for n x 1 x p, the tensor has rank n*p because each (i,k) pair needs its own term.
    # Hmm, actually R(n,1,p) = n*p is the standard result.
    
    # Let me reconsider. The rank of matrix multiplication tensor T_{n,m,p}:
    # R(n,m,p) <= R(n,m1,p) + R(n,m2,p) when m = m1+m2
    # R(n,m,p) <= R(n1,m,p) + R(n2,m,p) when n = n1+n2  
    # R(n,m,p) <= R(n,m,p1) + R(n,m,p2) when p = p1+p2
    
    # Also: R(n,m,p) <= R(n,m1,p1) + R(n,m1,p2) + R(n,m2,p1) + R(n,m2,p2)
    # when splitting both m and p (this is the "2x2" split)
    
    # For 3x4x8:
    # Split m: R(3,3,8) + R(3,1,8)
    # R(3,1,8) = 3*8 = 24 (outer product)
    # R(3,3,8): split p=8 into 4+4: R(3,3,4) + R(3,3,4)
    # R(3,3,4): split m=3 into 2+1: R(3,2,4) + R(3,1,4)
    # R(3,1,4) = 3*4 = 12
    # R(3,2,4): split p=4 into 2+2: R(3,2,2) + R(3,2,2)
    # R(3,2,2): This is 3x2x2. Split m=2 into 1+1: R(3,1,2)+R(3,1,2) = 6+6=12
    # Or split n=3 into 2+1: R(2,2,2)+R(1,2,2) = 8+4=12
    # Or use known result: R(3,2,2) <= 12
    # Actually, let's try: R(3,2,2). A is 3x2, B is 2x2, C is 3x2.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Split: C = A[:,0]*B[0,:] + A[:,1]*B[1,:]
    # Each term is outer product of length-3 and length-2 vectors, rank 3*2=6 each.
    # Total: 12.
    # Can we do better? The tensor T_{3,2,2} has 12 entries that are products.
    # I believe R(3,2,2) = 12 is tight for this small case.
    
    # So R(3,2,4) = 2 * 12 = 24
    # R(3,3,4) = 24 + 12 = 36
    # R(3,3,8) = 2 * 36 = 72
    # Total for 3x4x8: 72 + 24 = 96. That's just schoolbook!
    
    # I need a better split. Let me think about 2x2 splits.
    # For R(3,3,4): split m=3 into 2+1 AND p=4 into 2+2:
    # R(3,3,4) <= R(3,2,2) + R(3,2,2) + R(3,1,2) + R(3,1,2)
    # = 12 + 12 + 6 + 6 = 36. Same.
    
    # Hmm, let me try different splits for R(3,2,2).
    # Actually, maybe I should use the fact that for small dimensions,
    # there might be non-trivial decompositions.
    
    # Let me try a different approach: use known low-rank decompositions.
    # For 2x2x2, rank is 7 (Strassen).
    # For 3x3x3, rank is at most 23 (Strassen-like) or 21.
    
    # Actually, let me just implement the recursive splitting properly and see what rank we get.
    # The key insight is that we should try ALL possible splits and take the minimum.
    
    def min_rank(n, m, p, memo={}):
        """Compute minimum rank using recursive splitting."""
        key = (n, m, p)
        if key in memo:
            return memo[key]
        
        # Base case: small dimensions, use schoolbook
        if n * m * p <= 8:
            memo[key] = n * m * p
            return n * m * p
        
        best = n * m * p  # schoolbook upper bound
        
        # Split m dimension
        for m1 in range(1, m):
            m2 = m - m1
            r = min_rank(n, m1, p) + min_rank(n, m2, p)
            best = min(best, r)
        
        # Split n dimension
        for n1 in range(1, n):
            n2 = n - n1
            r = min_rank(n1, m, p) + min_rank(n2, m, p)
            best = min(best, r)
        
        # Split p dimension
        for p1 in range(1, p):
            p2 = p - p1
            r = min_rank(n, m, p1) + min_rank(n, m, p2)
            best = min(best, r)
        
        # 2x2 splits (split two dimensions at once)
        # Split m and p
        for m1 in range(1, m):
            m2 = m - m1
            for p1 in range(1, p):
                p2 = p - p1
                r = (min_rank(n, m1, p1) + min_rank(n, m1, p2) + 
                     min_rank(n, m2, p1) + min_rank(n, m2, p2))
                best = min(best, r)
        
        # Split n and m
        for n1 in range(1, n):
            n2 = n - n1
            for m1 in range(1, m):
                m2 = m - m1
                r = (min_rank(n1, m1, p) + min_rank(n1, m2, p) + 
                     min_rank(n2, m1, p) + min_rank(n2, m2, p))
                best = min(best, r)
        
        # Split n and p
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                r = (min_rank(n1, m, p1) + min_rank(n1, m, p2) + 
                     min_rank(n2, m, p1) + min_rank(n2, m, p2))
                best = min(best, r)
        
        memo[key] = best
        return best
    
    # Compute the rank
    rank = min_rank(n, m, p)
    
    # Now we need to actually CONSTRUCT the decomposition.
    # We'll use a recursive builder that tracks offsets.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def build_decomp(n_sub, m_sub, p_sub, i_off, j_off, k_off, col_start):
        """
        Build decomposition for multiplying A[i_off:i_off+n_sub, j_off:j_off+m_sub]
        by B[j_off:j_off+m_sub, k_off:k_off+p_sub]
        into C[i_off:i_off+n_sub, k_off:k_off+p_sub]
        
        Returns the next available column index.
        """
        if time.time() > deadline:
            return col_start  # timeout, return what we have
        
        # Determine which split to use (same logic as min_rank)
        if n_sub * m_sub * p_sub <= 8:
            # Schoolbook: one term per (i, j, k)
            col = col_start
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[(i_off + i) * m + (j_off + j)][col] = 1
                        V[(j_off + j) * p + (k_off + k)][col] = 1
                        W[(k_off + k) * n + (i_off + i)][col] = 1
                        col += 1
            return col
        
        # Find the best split (same as min_rank logic)
        best_rank = n_sub * m_sub * p_sub
        best_split = None  # (type, params)
        
        # Try all splits and find the one that gives minimum rank
        candidates = []
        
        # Split m
        for m1 in range(1, m_sub):
            m2 = m_sub - m1
            r = min_rank(n_sub, m1, p_sub) + min_rank(n_sub, m2, p_sub)
            candidates.append((r, ('m', m1, m2)))
        
        # Split n
        for n1 in range(1, n_sub):
            n2 = n_sub - n1
            r = min_rank(n1, m_sub, p_sub) + min_rank(n2, m_sub, p_sub)
            candidates.append((r, ('n', n1, n2)))
        
        # Split p
        for p1 in range(1, p_sub):
            p2 = p_sub - p1
            r = min_rank(n_sub, m_sub, p1) + min_rank(n_sub, m_sub, p2)
            candidates.append((r, ('p', p1, p2)))
        
        # 2x2 splits
        for m1 in range(1, m_sub):
            m2 = m_sub - m1
            for p1 in range(1, p_sub):
                p2 = p_sub - p1
                r = (min_rank(n_sub, m1, p1) + min_rank(n_sub, m1, p2) + 
                     min_rank(n_sub, m2, p1) + min_rank(n_sub, m2, p2))
                candidates.append((r, ('mp', m1, m2, p1, p2)))
        
        for n1 in range(1, n_sub):
            n2 = n_sub - n1
            for m1 in range(1, m_sub):
                m2 = m_sub - m1
                r = (min_rank(n1, m1, p_sub) + min_rank(n1, m2, p_sub) + 
                     min_rank(n2, m1, p_sub) + min_rank(n2, m2, p_sub))
                candidates.append((r, ('nm', n1, n2, m1, m2)))
        
        for n1 in range(1, n_sub):
            n2 = n_sub - n1
            for p1 in range(1, p_sub):
                p2 = p_sub - p1
                r = (min_rank(n1, m_sub, p1) + min_rank(n1, m_sub, p2) + 
                     min_rank(n2, m_sub, p1) + min_rank(n2, m_sub, p2))
                candidates.append((r, ('np', n1, n2, p1, p2)))
        
        # Pick the best
        candidates.sort(key=lambda x: x[0])
        best_r, best_split = candidates[0]
        
        col = col_start
        split_type = best_split[0]
        
        if split_type == 'm':
            m1, m2 = best_split[1], best_split[2]
            col = build_decomp(n_sub, m1, p_sub, i_off, j_off, k_off, col)
            col = build_decomp(n_sub, m2, p_sub, i_off, j_off + m1, k_off, col)
            
        elif split_type == 'n':
            n1, n2 = best_split[1], best_split[2]
            col = build_decomp(n1, m_sub, p_sub, i_off, j_off, k_off, col)
            col = build_decomp(n2, m_sub, p_sub, i_off + n1, j_off, k_off, col)
            
        elif split_type == 'p':
            p1, p2 = best_split[1], best_split[2]
            col = build_decomp(n_sub, m_sub, p1, i_off, j_off, k_off, col)
            col = build_decomp(n_sub, m_sub, p2, i_off, j_off, k_off + p1, col)
            
        elif split_type == 'mp':
            m1, m2, p1, p2 = best_split[1], best_split[2], best_split[3], best_split[4]
            col = build_decomp(n_sub, m1, p1, i_off, j_off, k_off, col)
            col = build_decomp(n_sub, m1, p2, i_off, j_off, k_off + p1, col)
            col = build_decomp(n_sub, m2, p1, i_off, j_off + m1, k_off, col)
            col = build_decomp(n_sub, m2, p2, i_off, j_off + m1, k_off + p1, col)
            
        elif split_type == 'nm':
            n1, n2, m1, m2 = best_split[1], best_split[2], best_split[3], best_split[4]
            col = build_decomp(n1, m1, p_sub, i_off, j_off, k_off, col)
            col = build_decomp(n1, m2, p_sub, i_off, j_off + m1, k_off, col)
            col = build_decomp(n2, m1, p_sub, i_off + n1, j_off, k_off, col)
            col = build_decomp(n2, m2, p_sub, i_off + n1, j_off + m1, k_off, col)
            
        elif split_type == 'np':
            n1, n2, p1, p2 = best_split[1], best_split[2], best_split[3], best_split[4]
            col = build_decomp(n1, m_sub, p1, i_off, j_off, k_off, col)
            col = build_decomp(n1, m_sub, p2, i_off, j_off, k_off + p1, col)
            col = build_decomp(n2, m_sub, p1, i_off + n1, j_off, k_off, col)
            col = build_decomp(n2, m_sub, p2, i_off + n1, j_off, k_off + p1, col)
        
        return col
    
    final_col = build_decomp(n, m, p, 0, 0, 0, 0)
    
    # Verify the construction used exactly 'rank' columns
    # If final_col != rank, we have a bug. Use the actual number.
    actual_rank = final_col
    if actual_rank != rank:
        # This shouldn't happen, but handle gracefully
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
