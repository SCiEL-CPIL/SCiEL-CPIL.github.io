"""Rewritten program for 4x4 * 4x7 matrix multiplication tensor decomposition.

Strategy:
1. Start with schoolbook (rank 112) as a valid baseline.
2. Use randomized tensor decomposition search to find lower-rank exact decompositions.
   We search for integer/half-integer entries by scaling: if we find a rational solution
   with denominator dividing some small D, we can represent as integers by multiplying.
   
   Actually, the task says "integer/half-integer entries". This means each entry is in
   {k/2 : k in Z}. So we can search over entries that are multiples of 0.5.
   
   Key insight: For matrix multiplication tensor, the standard result is that the
   rank of C_{n,m,p} is at most n*m*p (schoolbook). For 4x4*4x7, the known upper
   bounds are not trivially below 112. However, there are known constructions using
   Winograd-style or Strassen-style reductions.
   
   For 2x2x2, rank is 7 (Strassen). For larger matrices, block Strassen can reduce
   the rank. Let's think about whether we can apply a block decomposition.
   
   Actually, let's think more carefully. The matrix multiplication tensor T_{n,m,p}
   has rank equal to the minimum number of bilinear multiplications needed.
   
   For n=4, m=4, p=7: We can potentially use a recursive/block approach.
   
   One approach: Use the fact that 4 = 2+2, and apply Strassen's algorithm on blocks.
   But Strassen works for square matrices of size 2^k. Here we have 4x4 * 4x7.
   
   Let me think about a direct search approach. The space of rank-r decompositions
   is large, but we can use numerical optimization to find low-rank approximations
   and then round to half-integers.
   
   Given the 150s time limit, a full exhaustive search is infeasible. Let's try:
   1. Use randomized iterative methods (like ALS - Alternating Least Squares) for
      tensor decomposition, starting from random initializations.
   2. After convergence, round to nearest half-integer and verify exactness.
   3. Try decreasing ranks until we find one that works.
   
   However, ALS typically finds approximate decompositions. For exact decompositions
   with half-integer entries, we need a more careful approach.
   
   Alternative: Use the known algebraic structure. The matrix multiplication tensor
   T_{n,m,p} = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with appropriate indexing).
   
   For our case, T = sum_{i=0}^{3} sum_{j=0}^{3} sum_{k=0}^{6} E_{i,j} ⊗ E_{j,k} ⊗ F_{k,i}
   where F_{k,i} is the output tensor factor with index k*n+i.
   
   Let's try a smarter search: use the fact that we can combine terms. For example,
   in Strassen's algorithm for 2x2, we combine 8 terms into 7 by using clever
   linear combinations.
   
   Given time constraints, let's implement:
   1. A baseline schoolbook decomposition.
   2. A randomized search that tries to find lower-rank decompositions by:
      a. Randomly initializing U, V, W with half-integer entries in a small range.
      b. Using gradient-free optimization (e.g., coordinate descent or random perturbation)
         to minimize the error ||T - sum_c U_c ⊗ V_c ⊗ W_c||.
      c. Checking for exact zero error.
   
   But this is unlikely to work well in 150s for finding a rank < 112.
   
   Let me reconsider. The known best upper bound for 4x4*4x4 is 49 (from Strassen-like
   constructions). For 4x4*4x7, it might be higher.
   
   Actually, I recall that the rank of C_{4,4,4} is at most 49, and this is achieved
   by a construction due to Strassen. For C_{4,4,7}, we can potentially use a similar
   approach.
   
   However, implementing such a construction from scratch in the time available is
   challenging. Let's try a different approach:
   
   Use the fact that we can pad the 7 to 8 (making it 4x4*4x8) and use a known
   decomposition for that, then restrict. But this doesn't directly help.
   
   Given the constraints, let's implement a practical search:
   1. Start with rank 112 (schoolbook).
   2. Try to find a rank < 112 decomposition using numerical methods.
   3. If we find one, return it; otherwise, return the schoolbook.
   
   For the numerical search, I'll use a simplified approach:
   - Represent the tensor T as a 3D array of shape (n*m, m*p, p*n) = (16, 28, 28).
   - We want to find U (16 x r), V (28 x r), W (28 x r) such that
     T[a, b, c] = sum_{l=0}^{r-1} U[a,l] * V[b,l] * W[c,l]
   - We'll use a randomized ALS-like approach, but with rounding to half-integers
     at each step.
   
   Given the difficulty, let's just return the schoolbook decomposition for now,
   but with a small optimization: check if we can merge any terms. In the schoolbook,
   each term has exactly one nonzero entry in each of U, V, W. There's no obvious
   merging possible without changing the structure.
   
   Let's just return the schoolbook. The score will be 112, which is the baseline.
   
   Wait, the task says "score is the rank" and "lower is better". The seed program
   returns rank 112. If I return rank 112, the score is 112. The search score is
   seed/value, so 112/112 = 1.0. I need to do better than 1.0.
   
   Let me think harder about whether there's a known rank < 112 for 4x4*4x7.
   
   Actually, I found that for C_{n,m,p}, if we can embed it in a known lower-rank
   tensor, we can get a bound. But I don't recall a specific bound for 4,4,7.
   
   Let's try a computational approach with a time limit. I'll use scipy to do a
   randomized search for low-rank decompositions.
"""

import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = make_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower-rank decomposition
    # Strategy: Use randomized ALS with rounding to half-integers
    
    def tensor_error(U, V, W):
        """Compute ||T - sum_l U[:,l] ⊗ V[:,l] ⊗ W[:,l]||_F"""
        approx = np.einsum('al,bl,cl->abc', U, V, W, optimize=True)
        return np.linalg.norm(T - approx)
    
    def round_to_halfint(x):
        """Round array to nearest half-integer"""
        return np.round(2 * x) / 2.0
    
    def is_halfinteger(x, tol=1e-9):
        """Check if all entries are half-integers"""
        scaled = 2 * x
        return np.allclose(scaled, np.round(scaled), atol=tol)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        approx = np.einsum('al,bl,cl->abc', U, V, W, optimize=True)
        return np.allclose(T, approx, atol=tol)
    
    # Try ranks from min_rank to rank_sb - 1
    # Lower bound: The tensor T has rank at least max(nm, mp, pn) = 28
    # But for matrix multiplication, the lower bound is higher.
    # Let's try a few ranks.
    
    min_rank_to_try = 50  # Start from a reasonable lower bound
    
    for target_rank in range(min_rank_to_try, rank_sb):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        num_tries = 5
        for trial in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with half-integer entries in [-2, 2]
            U = np.round(2 * np.random.uniform(-2, 2, (nm, target_rank))) / 2.0
            V = np.round(2 * np.random.uniform(-2, 2, (mp, target_rank))) / 2.0
            W = np.round(2 * np.random.uniform(-2, 2, (pn, target_rank))) / 2.0
            
            # ALS iterations
            for it in range(20):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                WV = np.einsum('bl,cl->bc', V, W)  # (mp, pn)
                for a in range(nm):
                    # U[a, l] = sum_{b,c} T[a,b,c] * V[b,l] * W[c,l] / sum_{b,c} V[b,l]^2 * W[c,l]^2
                    num = np.einsum('bc,l->', T[a, :, :], WV[:, :, :].reshape(mp, pn, target_rank).transpose(2, 0, 1).reshape(target_rank, mp * pn)[:, :, :].reshape(target_rank, -1).T @ T[a, :, :].reshape(-1))
                    # Simpler:
                    num = np.einsum('bc,bl,cl->l', T[a, :, :], V, W)
                    den = np.einsum('bl,cl,bl,cl->l', V, W, V, W)
                    den = np.einsum('bl,bl,cl,cl->l', V, V, W, W)
                    den = np.sum((V ** 2) * (W ** 2), axis=0)
                    if np.any(den > 1e-12):
                        U[a, :] = num / np.where(den > 1e-12, den, 1)
                
                # Update V
                UW = np.einsum('al,cl->ac', U, W)
                for b in range(mp):
                    num = np.einsum('ac,al,cl->l', T[:, b, :], U, W)
                    den = np.sum((U ** 2) * (W ** 2), axis=0)
                    if np.any(den > 1e-12):
                        V[b, :] = num / np.where(den > 1e-12, den, 1)
                
                # Update W
                UV = np.einsum('al,bl->ab', U, V)
                for c in range(pn):
                    num = np.einsum('ab,al,bl->l', T[:, :, c], U, V)
                    den = np.sum((U ** 2) * (V ** 2), axis=0)
                    if np.any(den > 1e-12):
                        W[c, :] = num / np.where(den > 1e-12, den, 1)
                
                # Round to half-integers
                U = round_to_halfint(U)
                V = round_to_halfint(V)
                W = round_to_halfint(W)
            
            # Check if we found an exact decomposition
            if verify_decomposition(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank < rank_sb:
            break
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
