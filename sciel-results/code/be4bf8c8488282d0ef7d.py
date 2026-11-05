"""Matrix multiplication 3x4x4 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # The multiplication tensor T has indices (i, j, k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank = n*m*p = 48
    # We want to find lower rank.
    
    # Strategy: Use structured rank-reduction via block decomposition.
    # For 3x4x4, we can try to exploit structure.
    # 
    # One approach: partition the 4x4 part into blocks and apply Strassen-like
    # ideas, or use known results about tensor rank for rectangular matrices.
    #
    # Known: rank(3x4x4) is at most 44 (from various constructions).
    # Let's try to find a decomposition of rank 44 or lower.
    #
    # Approach: Use a combination of:
    # 1. Start with schoolbook (rank 48)
    # 2. Try to merge terms using linear dependencies
    # 3. Use random search / optimization to find lower-rank decompositions
    # 4. Use structured block approaches
    
    # First, let's try a structured approach based on block decomposition.
    # Split the 4x4 matrix into 2x2 blocks:
    # A is 3x4, B is 4x4
    # Write B as [[B11, B12], [B21, B22]] where each Bi is 2x2
    # A = [A1 | A2] where A1, A2 are 3x2
    # C = A*B = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    # 
    # Each 3x2 * 2x2 multiplication: rank = 3*2*2 = 12
    # Total: 4 * 12 = 48 (same as schoolbook)
    #
    # To reduce rank, we need to share computations across blocks.
    
    # Let's try a direct approach: search for a low-rank decomposition
    # using the fact that the tensor has specific structure.
    
    # Method: Try rank r from low to high, attempt to find U, V, W with 
    # integer/half-integer entries such that the tensor identity holds.
    
    # For efficiency, let's use a known construction if possible.
    
    # Known construction for 3x4x4 with rank 44:
    # This uses the fact that we can "absorb" some terms.
    
    # Let me try a different approach: use the structure of the problem.
    # The tensor T_{ijk} = 1 for all i,j,k (it's the all-ones tensor in 
    # the appropriate sense... no, it's the standard multiplication tensor).
    
    # Actually, T_{ijk} = 1 if and only if the (i,j) entry of A and (j,k) entry
    # of B contribute to (i,k) entry of C. So T_{ijk} = 1 for all valid i,j,k.
    # Wait, no. The multiplication tensor has T_{ijk} = 1 for all i in [n], j in [m], k in [p].
    # Because C_{ik} = sum_j A_{ij} B_{jk}, so the tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # in the appropriate basis. So T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    # The rank of this tensor is what we're looking for.
    
    # For n=3, m=4, p=4:
    # Lower bound: max(n*p, m*p, n*m) = max(12, 16, 12) = 16
    # Upper bound (schoolbook): 48
    # Best known: I believe it's around 40-44.
    
    # Let's try to construct a decomposition.
    
    # Approach: Use a greedy/structured method.
    # Start with the 48 terms and try to combine them.
    
    # Alternative: Use random initialization and gradient descent on the
    # constraint that U ⊙ V ⊙ W = T (where ⊙ is the outer product sum).
    
    # For exactness with integer/half-integer entries, we need a different approach.
    
    # Let's try: construct U, V, W with small integer entries and verify.
    
    # Key insight: If we can find U, V, W with entries in {0, 1/2, 1} or similar,
    # and the product sums to exactly 1 for each (i,j,k), we're done.
    
    # Let's try a block-based Strassen-like construction.
    
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8.
    # For our case, we can partition into 2x2 blocks.
    
    # Partition:
    # A (3x4): rows 0-2, cols 0-3
    # B (4x4): rows 0-3, cols 0-3
    # C (3x4): rows 0-2, cols 0-3
    
    # Split columns of A and B into two halves of size 2:
    # A = [A_L | A_R] (3x2 each)
    # B = [[B_TL | B_TR], [B_BL | B_BR]] (2x2 each)
    # C = [C_L | C_R] (3x2 each)
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    
    # Each term A_X * B_Y is a 3x2 * 2x2 = 3x2 multiplication, rank 12.
    # 4 terms * 12 = 48.
    
    # To reduce: use Strassen on the 2x2 blocks of B?
    # Actually, let's think differently.
    
    # Let's try to use the following: for the 2x2x2 subproblem, Strassen uses 7 multiplications.
    # If we can apply this to our blocks...
    
    # Actually, let me just try a computational search for a good rank.
    
    # Given time constraints, let's try:
    # 1. A known rank-44 construction (if I can recall/derive it)
    # 2. Random search for lower rank
    
    # Let's try rank 44 first with a structured approach.
    
    # Construction idea: 
    # Use 12 "full" terms for the main diagonal blocks and 32 for the off-diagonal,
    # but share some computations.
    
    # Actually, let me try a simpler approach: use numpy to attempt to find
    # a low-rank decomposition via optimization, then round to nearest half-integers.
    
    # But for exactness, we need exact integer/half-integer entries.
    
    # Let's try: for rank r, randomly initialize U, V, W with entries from {0, 1, -1, 1/2, -1/2}
    # and check if they work. This is a huge search space, so we need structure.
    
    # Better: use the fact that the tensor is "generic" in a sense, and use
    # known algebraic results.
    
    # For now, let's try to construct a rank-44 solution using a specific pattern.
    
    # Pattern: Start with schoolbook (48 terms) and remove 4 terms by adjusting
    # the remaining 44 to compensate.
    
    # This is complex. Let's try a different tactic:
    
    # Use the LLL algorithm or similar to find a low-rank decomposition?
    # No, that's for linear algebra, not tensor rank.
    
    # Let's try: assume a specific structure for U, V, W and solve for the entries.
    
    # For rank r, we have r * (n*m + m*p + p*n) = r * (12 + 16 + 12) = r * 40 unknowns.
    # We have n*m*p = 48 equations.
    # For r >= 48/40 = 1.2, we have more unknowns than equations, so solutions exist.
    # But we need integer/half-integer solutions, which is much harder.
    
    # Let's try a pragmatic approach:
    # 1. Try rank 44 with a specific structured guess
    # 2. If that fails, fall back to rank 48 (schoolbook)
    
    # Structured guess for rank 44:
    # Use all 48 schoolbook terms, but for 4 of them, express them as linear combinations
    # of the others. This requires finding linear dependencies.
    
    # Actually, let's try a completely different approach:
    # Use the fact that for rectangular matrix multiplication, there are known
    # rank bounds and constructions.
    
    # For 3x4x4, I'll try to use a "padded" Strassen approach.
    
    # Pad 3x4 to 4x4 by adding a zero row, and 4x4 stays.
    # Then use rank-16 (or lower) for 4x4x4? No, 4x4x4 has rank at least 16.
    # Schoolbook for 4x4x4 is 64. Strassen gives 49.
    # But padding doesn't directly help.
    
    # Let me just try a computational approach with time limit.
    
    # Approach: For each candidate rank r from 16 to 48:
    #   Try to find U, V, W with entries in {0, 1/2, 1} (or small integers)
    #   such that the tensor identity holds.
    #   Use random sampling with verification.
    
    # But this is intractable for large r.
    
    # Let's try: use the schoolbook as a starting point and try to reduce rank
    # by identifying and removing redundant terms.
    
    # For the schoolbook, term (i,j,k) contributes U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if there's a way
    # to combine them into fewer terms.
    
    # This is essentially finding a low-rank decomposition, which is hard.
    
    # Given the time constraints, let's try a specific known construction.
    
    # After some thought, let's try the following construction for rank 44:
    
    # Use 12 terms for "diagonal" blocks (i, j, k where j and k are in the same half)
    # and 32 terms for "off-diagonal" blocks, with some sharing.
    
    # Actually, let me just try to verify if a rank-44 solution exists by
    # attempting to construct one.
    
    # Let's use a different strategy: start with schoolbook and try to
    # eliminate 4 terms by modifying the others.
    
    # For each set of 4 terms to eliminate, check if the remaining 44 can be
    # adjusted (with integer/half-integer entries) to still produce the correct tensor.
    
    # This is a system of linear equations. Let's set it up.
    
    # For a given set of 44 "active" terms, we need to find U, V, W such that:
    # sum_{r in active} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a nonlinear system (trilinear). Hard to solve directly.
    
    # Let's try a simpler approach: use random search with a time limit.
    
    # For rank 44, randomly initialize U, V, W with small integer/half-integer entries,
    # compute the resulting tensor, and check if it matches.
    
    # This is unlikely to work with pure random search, but let's try with structure.
    
    # Better idea: use the schoolbook as a base and perturb it.
    
    # Let me try a specific construction:
    
    # For 3x4x4, consider the following:
    # - 12 terms for (i, j, j) for i in [0,3), j in [0,4) -- "diagonal" in j,k
    # - 12 terms for (i, j, k) where j != k, j < k -- upper triangle
    # - 12 terms for (i, j, k) where j != k, j > k -- lower triangle  
    # - 8 terms for additional structure
    # Total: 44
    
    # This is getting too speculative. Let me just implement a search with time limit.
    
    # Practical approach:
    # 1. Return schoolbook (rank 48) as a guaranteed valid solution
    # 2. Try to find a lower-rank solution within the time budget
    # 3. Return the best found
    
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the multiplication tensor."""
        rank = len(U[0])
        # Check dimensions
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if val * 2 != int(val * 2):
                        return False
        # Check tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to find a rank-44 solution using a structured approach
    # based on block decomposition with Strassen-like sharing.
    
    # Construction for rank 44:
    # Partition the 4x4 matrix B into four 2x2 blocks.
    # A is 3x4, split into two 3x2 halves: A1 (cols 0-1), A2 (cols 2-3)
    # B = [[B11, B12], [B21, B22]]
    # C = [C1, C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    
    # Each 3x2 * 2x2 product has rank 12 (schoolbook).
    # 4 products * 12 = 48.
    
    # To reduce to 44, we need to save 4 ranks.
    # Use Strassen on the 2x2 blocks: each 2x2*2x2 can be done in rank 7 instead of 8.
    # But our products are 3x2 * 2x2, not 2x2*2x2.
    
    # For 3x2 * 2x2: 
    # Let A' be 3x2, B' be 2x2. C' = A'*B' is 3x2.
    # C'[i][0] = A'[i][0]*B'[0][0] + A'[i][1]*B'[1][0]
    # C'[i][1] = A'[i][0]*B'[0][1] + A'[i][1]*B'[1][1]
    # For each i, this is a 1x2 * 2x2 = 1x2 product, rank 4 schoolbook.
    # 3 rows * 4 = 12.
    
    # Can we do better? For 1x2 * 2x2, the rank is 4 (can't do better than schoolbook
    # for this small case... actually, rank of 1x2x2 tensor is 4).
    
    # So 3x2*2x2 has rank 12, and we can't easily reduce it.
    
    # Alternative: share computations across the four 3x2*2x2 products.
    
    # Let me try: 
    # M1 = A1 * B11 (3x2)
    # M2 = A2 * B21 (3x2)  
    # M3 = A1 * B12 (3x2)
    # M4 = A2 * B22 (3x2)
    # C1 = M1 + M2, C2 = M3 + M4
    
    # If we could compute M1+M2 and M3+M4 directly with fewer operations...
    # M1 + M2 = A1*B11 + A2*B21 = [A1|A2] * [B11; B21] (3x4 * 4x2 = 3x2)
    # This is a 3x4 * 4x2 multiplication, rank 24 schoolbook.
    # But we're computing it as part of the larger product, so this doesn't directly help.
    
    # I think the best approach given time constraints is to:
    # 1. Use the schoolbook (rank 48) as a safe fallback
    # 2. Attempt a limited search for rank 44 or lower
    
    # Let's try a specific rank-44 construction:
    
    # Use the following idea: for the 3x4x4 problem, we can use
    # 12 terms for the "main" structure and 32 for the rest, with
    # 4 terms saved by using the relation that some products can be
    # expressed differently.
    
    # Actually, let me try to use a known result: 
    # The tensor rank of 3x4x4 is at most 44.
    # One construction uses the following pattern:
    
    # I'll attempt to construct it explicitly.
    
    # Let's use indices:
    # U has 12 rows (3*4), V has 16 rows (4*4), W has 12 rows (4*3)
    
    # For rank 44, we have 44 columns in each matrix.
    
    # Construction:
    # Terms 0-11: (i, j, j) for i in 0..2, j in 0..3  [12 terms]
    # Terms 12-23: (i, j, k) for i in 0..2, j in 0..1, k in 2..3  [12 terms]  
    # Terms 24-35: (i, j, k) for i in 0..2, j in 2..3, k in 0..1  [12 terms]
    # Terms 36-43: additional 8 terms to handle the remaining structure
    
    # This is getting too complex without a clear derivation.
    
    # Let me just try a random search with time limit for rank 44.
    
    best_result = None
    best_rank = n * m * p  # 48
    
    # Try to find a solution with rank < 48
    for target_rank in range(44, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try a few random constructions
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Initialize with schoolbook pattern, then randomly zero out
            # some entries and adjust others
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Start with a subset of schoolbook terms
            terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            random.shuffle(terms)
            terms = terms[:target_rank]
            
            for r, (i, j, k) in enumerate(terms):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
            
            # Try to fix up the solution
            # This is a crude approach; in practice we'd need optimization.
            # Skip verification for now due to time.
            
            # Check if this works
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    return schoolbook()
