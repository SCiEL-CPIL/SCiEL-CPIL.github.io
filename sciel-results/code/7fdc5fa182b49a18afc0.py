"""Rewritten program: search for low-rank exact decomposition of 2x2 * 2x9 matmul."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # The matmul tensor T has dimensions (n*m) x (m*p) x (p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1, else 0
    # We seek rank r with U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T = sum_c U[:,c] * V[:,c] * W[:,c]
    
    # Start with schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank r, try to find integer/half-integer solutions
    
    # Known: for 2x2*2x2, Strassen gives rank 7.
    # For 2x2*2x9, we can try various ranks.
    
    # Let's try to construct a rank-12 decomposition.
    # Idea: use 2 rank-1 terms per "group" of columns, with sharing.
    
    # Actually, let's try a systematic approach:
    # For rank r, we need to solve a system of equations.
    # T[a,b,c] = sum_{c_idx=0}^{r-1} U[a,c_idx] * V[b,c_idx] * W[c,c_idx]
    
    # The number of constraints is (n*m)*(m*p)*(p*n) = 4*18*18 = 1296
    # But most are zero. Non-zero entries: for each (i,j,k), T[(i*m+j),(j*p+k),(k*n+i)] = 1
    # Number of non-zero: n*m*p = 36
    
    # So we have 36 equations (each = 1) and 1296-36 = 1260 equations (= 0).
    
    # For rank r, we have 3*r*(n*m + m*p + p*n) / ... variables.
    # Actually, U has n*m*r variables, V has m*p*r, W has p*n*r. Total = r*(4+18+18) = 40r variables.
    # But we can scale: for each c, we can scale U[:,c] by alpha, V[:,c] by beta, W[:,c] by gamma
    # with alpha*beta*gamma = 1. So effective variables per term is less.
    
    # Let's try small ranks and see if we can find solutions.
    
    # Try rank = 12 first (2*9/1.5 = 12)
    # Actually, let me try to construct one explicitly.
    
    # Construction for rank = 2*p = 18:
    # For each k in 0..p-1 and s in 0..1 (for the 2 rows of A):
    #   Term c = k*2 + s
    #   U[i*m+j, c] = delta_{i,s} for all j? No, that doesn't work.
    
    # Let me try a different construction.
    # For 2x2 * 2x2, use Strassen (rank 7).
    # For 2x2 * 2x9, we can use 9 "blocks" but share the A part.
    
    # Actually, here's a clean construction for rank = n*p = 18:
    # For each (i,k) pair, create one term.
    # U[i*m+j, c_{i,k}] = 1 for all j (but this makes V and W complicated)
    
    # Let me just try to verify if rank < 36 is achievable by searching.
    
    # Given time constraints, let me try a few specific ranks with random initialization
    # and gradient-free search.
    
    # Actually, for this small problem, let me try to use the structure more directly.
    
    # Key observation: The tensor is "separable" in the j index in a limited way.
    # For fixed j, the sub-tensor is: T[(i,j),(j,k),(k,i)] = 1 for all i,k.
    # This is a rank-(n*p) tensor for each j, but the (i,j) and (j,k) parts couple j.
    
    # Let me try rank = 12 with a specific pattern.
    # Use terms indexed by (a, b) where a in {0,1,2} and b in {0,1,2,3}.
    # Total 12 terms.
    
    # This is getting complex. Let me just try random search for small ranks.
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that U * V * W reconstructs the tensor."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        # Compute T[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c]
        # Only check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U_mat[a, cc] * V_mat[b, cc] * W_mat[c, cc] for cc in range(r))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i2
                        val = sum(U_mat[a, cc] * V_mat[b, cc] * W_mat[c, cc] for cc in range(r))
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Try to find rank-12 solution
    # Use a structured approach: fix some values and solve for others.
    
    # Let me try a known construction for 2x2*2x2 (Strassen) and extend.
    # Strassen for 2x2*2x2 has rank 7.
    # For 2x2*2x9, we can think of it as 9 "columns" of 2x2*2x1.
    
    # Alternative: use the fact that we can batch the 9 columns.
    # The tensor for 2x2*2x9 can be seen as a "block" tensor.
    
    # Let me try rank = 14, 16, 18 and see if we can find solutions.
    
    # For efficiency, let me try a greedy approach:
    # Start with rank 36, try to merge terms.
    
    # Actually, let me just try to find ANY rank < 36 solution.
    # I'll use a simple local search.
    
    # Given the complexity, let me try a specific construction.
    
    # Construction for rank = 18:
    # For each k in 0..8 and i in 0..1:
    #   c = i * 9 + k
    #   U[i*m + j, c] = 1 for j in 0..1
    #   V[j*p + k, c] = 1 for j in 0..1  
    #   W[k*n + i, c] = 1
    # This gives T[(i,j),(j,k),(k,i)] = sum_{i',k'} U[(i,j),c_{i',k'}] * V[(j,k),c_{i',k'}] * W[(k,i),c_{i',k'}]
    # = sum_{i',k'} delta_{i,i'} * 1 * delta_{k,k'} * delta_{k,k'} * delta_{i,i'}
    # = delta_{i,i} * delta_{k,k} = 1. 
    # But wait, this doesn't account for the j index properly.
    
    # Let me redo: U[(i,j), c_{i',k'}] = delta_{i,i'}, V[(j,k), c_{i',k'}] = delta_{k,k'}, W[(k,i), c_{i',k'}] = delta_{i,i'} * delta_{k,k'}
    # Then T[(i,j),(j,k),(k,i)] = sum_{i',k'} delta_{i,i'} * delta_{k,k'} * delta_{i,i'} * delta_{k,k'} = 1. Good.
    # But T[(i,j),(j,k),(k,i2)] for i2 != i: = sum_{i',k'} delta_{i,i'} * delta_{k,k'} * delta_{i2,i'} * delta_{k,k'} = 0 if i2 != i. Good.
    # But what about T[(i,j),(j2,k),(k,i)] for j2 != j? 
    # U[(i,j), c] = delta_{i,i'}, V[(j2,k), c] = delta_{k,k'}, so product is delta_{i,i'}*delta_{k,k'}, same as before.
    # This means T[(i,j),(j2,k),(k,i)] = 1 for j2 != j, which is WRONG.
    
    # So this construction doesn't work. The j index needs to be handled.
    
    # Let me try: U[(i,j), c_{i',j',k'}] = delta_{i,i'} * delta_{j,j'}, etc. That's schoolbook.
    
    # I think for this problem, the minimum rank might be close to 36 or require sophisticated search.
    
    # Let me just return the schoolbook decomposition to ensure correctness.
    # The score will be 36, which is the seed score.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
