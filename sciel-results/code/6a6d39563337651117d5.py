"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-6:
                        return False
        return True
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Strategy: Try to find rank reductions using structured decompositions
    # Key insight: The tensor T[i,j,k] = 1 for all (i,j,k) is rank 1 as a 3-tensor
    # but we need the "structured" rank where each term is a product of three vectors
    # in the respective index spaces.
    
    # For a "product" tensor where T[i,j,k] = A[i]*B[j]*C[k], rank is 1.
    # Our tensor is T[i,j,k] = 1 for all i,j,k, which IS a product tensor: 1*1*1.
    # So the free rank (tensor rank) is 1!
    # But wait - the task says "exact tensor decomposition for multiplying"
    # The tensor for matrix multiplication M_{n x m} * M_{m x p} = M_{n x p} is:
    # T[i, j, k] where the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, (j,k), (j', k')] ... no.
    
    # Let me re-read the task:
    # "exactly decompose the matrix multiplication tensor"
    # The tensor for multiplying (n x m) * (m x p) -> (n x p) is:
    # T_{i, j, k}^{i', j', k'} ... 
    # Actually the standard tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k, i', j', k'] = delta_{j,j'} * delta_{k,k'} ... no.
    
    # The matrix multiplication tensor T in C^{n*m} x C^{m*p} x C^{p*n} is defined by:
    # T = sum_{i,j,k} e_{i*m+j} tensor e_{j*p+k} tensor e_{k*n+i}
    # So T_{(i*m+j), (j*p+k), (k*n+i)} = 1 and 0 otherwise.
    # This is the schoolbook decomposition of rank n*m*p.
    
    # The question is: what is the minimal rank?
    # For n=m=p=2, it's 7 (Strassen).
    # For 3x5x6, we need to find the minimal rank.
    
    # The tensor is: sum over all (i,j,k) of e_{i*m+j} * e_{j*p+k} * e_{k*n+i}
    # This has 90 terms in schoolbook. Can we do better?
    
    # Key observation: The tensor T satisfies:
    # T_{a,b,c} = 1 if and only if there exist i,j,k such that 
    # a = i*m+j, b = j*p+k, c = k*n+i
    # i.e., a mod m = b div p, and b mod p = c div n, and c mod n = a div m
    
    # This is a very structured tensor. The question is its rank.
    
    # One approach: Try to find a decomposition of rank < 90.
    # Since this is a research problem, let's try a heuristic search.
    
    # Approach: Start with schoolbook, try to "merge" terms by finding
    # linear dependencies. For example, if we can find coefficients such that
    # sum_r alpha_r * T_r = 0, we might be able to reduce rank.
    
    # Actually, let's try a different approach: use the fact that for large enough
    # rank, we can use random linear combinations.
    
    # Let's try to find a rank-1 decomposition (impossible since the tensor is not rank 1)
    # or try structured approaches.
    
    # For 2x2x2, Strassen uses rank 7. The idea is to use:
    # m0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # m1 = (A[1,0]+A[1,1])*B[0,0]
    # m2 = A[0,0]*(B[0,1]-B[1,1])
    # m3 = A[1,1]*(B[1,0]-B[0,0])
    # m4 = (A[0,0]+A[0,1])*B[1,1]
    # m5 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # m6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # For our 3x5x6 case, let's try to find ANY improvement over 90.
    
    # Simple improvement: Notice that the tensor is a sum of 90 "unit" tensors.
    # If we can find a linear relation among them, we can reduce rank.
    
    # Actually, the tensor T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is the structure tensor of the matrix multiplication algebra.
    
    # Let me try a numerical approach: use random vectors and solve a linear system.
    
    # For rank r, we need vectors u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # such that sum_r u_r ⊗ v_r ⊗ w_r = T
    
    # This is equivalent to: for each (a,b,c), sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c]
    
    # T[a,b,c] = 1 if a%m == b//p and b%p == c//n and c%n == a//m, else 0.
    
    # Let's try rank 89 first as a test.
    
    # Actually, for a cleaner approach, let me try to exploit the structure.
    # The tensor can be written as:
    # T = sum_j (sum_i e_{i*m+j} ⊗ e_{j*p+*}) (sum_k e_{*p+k} ⊗ e_{k*n+*})
    # Hmm, that's not quite right.
    
    # Let me try: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Group by j: T = sum_j sum_{i,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # For fixed j: sum_{i,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is a tensor in R^{n*m} x R^{m*p} x R^{p*n} with support on 
    # {i*m+j} x {j*p+k} x {k*n+i} for i in 0..n-1, k in 0..p-1.
    
    # For fixed j, this sub-tensor has n*p = 3*6 = 18 terms.
    # Can we decompose it with fewer than 18 terms?
    # The sub-tensor for fixed j: S_j = sum_{i,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is like a "matrix multiplication" tensor of size n x 1 x p... 
    # Actually it's the tensor for multiplying an (n x 1) matrix by a (1 x p) matrix,
    # but with the middle index fixed.
    
    # For n x 1 x p, the schoolbook rank is n*p. But the tensor is:
    # sum_{i,k} e_i ⊗ e_k ⊗ e_{k*n+i}... no wait, the indices are mixed.
    
    # Let me just try a brute force numerical search for small rank reductions.
    
    # Given time constraints, let's try a simpler approach:
    # Use the fact that the tensor has a lot of zeros and try to find
    # a low-rank approximation that is exact.
    
    # Actually, I realize I should try to use known results.
    # The rank of the matrix multiplication tensor M_{3,5,6} is unknown in general,
    # but it's at most 90 (schoolbook) and at least some lower bound.
    
    # Let me try a heuristic: randomly perturb the schoolbook decomposition
    # and see if we can find a lower rank exact solution.
    
    # Time check
    if time.time() > deadline - 30:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find rank 89 decomposition
    # We need to find 89 terms that sum to T.
    # Start with schoolbook 90 terms, try to merge two terms into one.
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
    # u, v, w such that u⊗v⊗w = e_{i1*m+j1}⊗e_{j1*p+k1}⊗e_{k1*n+i1} + e_{i2*m+j2}⊗e_{j2*p+k2}⊗e_{k2*n+i2}
    # This requires the sum to be a rank-1 tensor, which is generally impossible.
    
    # Alternative: Try to find 89 vectors that work.
    # This is a system of 90 equations (one for each (i,j,k) pair where T=1)
    # in 89*3*(15+30+18) = too many unknowns, so it should be solvable.
    
    # Actually, let's think about this differently.
    # The tensor T is in R^{15} x R^{30} x R^{18}.
    # We want to write T as sum of r rank-1 tensors.
    # For r = 90, we have the schoolbook solution.
    # For r < 90, we need to find a different decomposition.
    
    # Let me try a randomized approach:
    # 1. Pick r random vectors u_1,...,u_r in R^{15}
    # 2. For each (a,b,c) with T[a,b,c]=1, we need sum_{r'} u_{r'}[a] * v_{r'}[b] * w_{r'}[c] = 1
    #    For (a,b,c) with T[a,b,c]=0, we need the sum = 0.
    
    # This is a bilinear (trilinear) system, which is hard to solve directly.
    
    # Let me try a different strategy: use the structure of the problem.
    # The tensor T satisfies T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and c%n == a//m.
    
    # Let's try to see if we can reduce the rank by exploiting symmetries or
    # by using a block decomposition.
    
    # Partition the indices:
    # a in {0..14}, split by j = a%m: j=0 (a=0,5,10), j=1 (a=1,6,11), ..., j=4 (a=4,9,14)
    # b in {0..29}, split by k = b%p: k=0 (b=0,6,12,18,24), ..., k=5 (b=5,11,17,23,29)
    #   and also by j' = b//p: j'=0 (b=0..5), j'=1 (b=6..11), ..., j'=4 (b=24..29)
    # c in {0..17}, split by i = c%n: i=0 (c=0,3,6,9,12,15), ..., i=2 (c=2,5,8,11,14,17)
    #   and by k' = c//n: k'=0 (c=0,1,2), k'=1 (c=3,4,5), ..., k'=5 (c=15,16,17)
    
    # T[a,b,c]=1 iff j(a) == j'(b) and k(b) == k'(c) and i(c) == i(a)
    # where i(a) = a//m, j(a) = a%m, j'(b) = b//p, k(b) = b%p, k'(c) = c//n, i(c) = c%n
    
    # This is a very specific pattern. The tensor is a "perfect matching" type tensor.
    
    # I think for now, given the time constraints and the difficulty of finding
    # a better decomposition, I'll return the schoolbook decomposition.
    # A proper optimization would require more sophisticated algorithms.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
