"""Program using randomized algebraic geometry approach for low-rank tensor decomposition."""
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the expected tensor
        # T[a,b,c] = 1 iff exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        # Precompute which (a,b,c) are valid
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_set.add((a, b, c))
        
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    expected = 1 if (a, b, c) in valid_set else 0
                    if total != expected:
                        return False
        return True
    
    def try_rank_41():
        """
        Try to find a rank-41 decomposition using structured rational search.
        
        Approach: We look for a decomposition where most terms are schoolbook-like
        but with some shared structure. Specifically, we try to find two (i,j,k)
        triples that can be covered by a single rank-1 term with non-trivial
        coefficients, or find a linear dependency among the 42 schoolbook terms.
        
        For rank 41, we need 41 rank-1 tensors whose sum equals the multiplication
        tensor. We try structured perturbations of the schoolbook decomposition.
        """
        random.seed(42)
        
        # Strategy: Try to find a rank-41 decomposition by solving a linear system.
        # Fix U and V to have a specific sparse structure with a few free parameters,
        # then solve linearly for W.
        
        # For each pair of schoolbook terms we try to merge, check if the residual
        # (the sum of the two basis tensors minus one rank-1 term) can be absorbed
        # into the remaining 40 terms.
        
        # More concretely: try all pairs of (i,j,k) triples and see if there exists
        # a rank-1 tensor R such that T - R has rank <= 40.
        
        # T - R where R = u ⊗ v ⊗ w. For T - R to have rank 40, we need the
        # remaining 40 schoolbook terms to span T - R.
        
        # This is equivalent to: does there exist a rank-1 tensor R such that
        # for all (a,b,c) not in the support of the 40 fixed terms, the equation
        # holds, and for the two "merged" positions, the equation also holds.
        
        # Let's try: fix 40 of the 42 schoolbook terms. The two remaining positions
        # (a1,b1,c1) and (a2,b2,c2) need to be covered by one rank-1 term plus
        # adjustments to the 40 fixed terms.
        
        # Actually, let's try a direct approach:
        # For a rank-41 decomposition, we have 41 terms. Let's parameterize:
        # - 40 terms are schoolbook for 40 specific (i,j,k) triples
        # - 1 term is a general rank-1 tensor u ⊗ v ⊗ w
        # The constraint is that the sum equals T.
        
        # For each of the 42 choose 2 ways to pick which two triples are "special":
        # The 40 standard terms handle 40 triples. The special term must handle
        # the remaining 2 triples AND not disturb the 40 standard ones.
        
        # This means: for each of the 40 standard (a,b,c), the special term must
        # contribute 0. And for the 2 special (a,b,c), the special term must
        # contribute 1.
        
        # So we need: u[a]*v[b]*w[c] = 0 for 40 positions, and = 1 for 2 positions.
        # This is generally impossible unless the 2 positions are "compatible."
        
        # Let's check: for two positions (a1,b1,c1) and (a2,b2,c2), can we find
        # u,v,w such that u[a1]*v[b1]*w[c1]=1, u[a2]*v[b2]*w[c2]=1, and
        # u[a]*v[b]*w[c]=0 for all other 40 positions?
        
        # The zero constraints mean: for each of the 40 positions, at least one of
        # u[a], v[b], w[c] must be 0.
        
        # Let's enumerate pairs and check feasibility.
        
        # Generate all 42 (i,j,k) triples
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # For each pair, check if a rank-1 tensor can satisfy the constraints
        for idx1 in range(len(triples)):
            for idx2 in range(idx1 + 1, len(triples)):
                if time.time() > deadline:
                    return None
                
                i1, j1, k1 = triples[idx1]
                i2, j2, k2 = triples[idx2]
                
                a1, b1, c1 = i1*m+j1, j1*p+k1, k1*n+i1
                a2, b2, c2 = i2*m+j2, j2*p+k2, k2*n+i2
                
                # The special term must be 1 at (a1,b1,c1) and (a2,b2,c2)
                # and 0 at all other 40 positions.
                
                # For the zero constraints: for each other (a,b,c), 
                # u[a]*v[b]*w[c] = 0.
                
                # This means: the set of a-values used by the 40 other positions
                # (excluding a1, a2) must have u=0, OR the b-values must have v=0,
                # OR the c-values must have w=0.
                
                # More precisely: for each of the 40 positions, at least one factor is 0.
                
                # Let's see what a-values are "covered" by the 40 other positions.
                other_a = set()
                other_b = set()
                other_c = set()
                for idx in range(len(triples)):
                    if idx == idx1 or idx == idx2:
                        continue
                    i, j, k = triples[idx]
                    other_a.add(i*m+j)
                    other_b.add(j*p+k)
                    other_c.add(k*n+i)
                
                # a1 and a2 might or might not be in other_a
                # For the special term to be 1 at (a1,b1,c1): u[a1]≠0, v[b1]≠0, w[c1]≠0
                # For the special term to be 1 at (a2,b2,c2): u[a2]≠0, v[b2]≠0, w[c2]≠0
                # For zero at other positions: for each (a,b,c) in the 40, 
                #   u[a]=0 or v[b]=0 or w[c]=0
                
                # Since u[a1]≠0 and u[a2]≠0, if a1 is in other_a, then for positions
                # with a=a1, we need v[b]=0 or w[c]=0.
                
                # This is getting complex. Let's just try to solve it directly
                # with a small number of free variables.
                
                # Simple check: if a1==a2, b1==b2, or c1==c2, it might be easier.
                # If a1==a2: u[a1]≠0, need v[b1]*w[c1]=1 and v[b2]*w[c2]=1,
                # and for all other positions with a=a1, v[b]*w[c]=0.
                
                # Let's just do a quick feasibility check by trying small rational values.
                # For efficiency, only try a few candidate pairs.
                
                # Quick necessary condition: the two positions must share at least
                # one index coordinate for a rank-1 tensor to be nonzero at both
                # while being zero elsewhere (in a simple pattern).
                
                # Actually, this condition is not necessary in general. Let's skip
                # to a more direct approach.
        
        # Direct linear algebra approach for rank 41:
        # Parameterize U and V with some structure, solve for W.
        # 
        # Try: U has 6x41 entries, V has 21x41 entries. Fix most to 0 or 1,
        # leave a few free, solve linear system for W (14x41).
        #
        # The system is: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is linear in W if U and V are fixed.
        
        # Let's try: fix U to be the first 41 columns of the schoolbook U (6x42),
        # and fix V similarly. Then solve for W.
        # But this gives W = schoolbook W with last column zeroed, which won't work.
        
        # Better: try random sparse U and V, solve for W, check.
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Create random sparse U and V
            U = [[0]*41 for _ in range(n*m)]
            V = [[0]*41 for _ in range(m*p)]
            
            # Initialize with random small rationals
            for a in range(n*m):
                for r in range(41):
                    if random.random() < 0.3:
                        U[a][r] = random.choice([-1, 1, 2, -2, 3, -3])
            
            for b in range(m*p):
                for r in range(41):
                    if random.random() < 0.3:
                        V[b][r] = random.choice([-1, 1, 2, -2, 3, -3])
            
            # Solve for W: for each c, the equation is:
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b)
            # This is a linear system in W[c,0], W[c,1], ..., W[c,40]
            # with 6*21 = 126 equations and 41 unknowns (overdetermined)
            
            # For each c, solve the linear system A_c * w_c = t_c
            # where A_c[a,b] = U[a,r]*V[b,r] (flattened) and t_c[a,b] = T[a,b,c]
            
            W = [[0]*41 for _ in range(p*n)]
            
            feasible = True
            for c in range(p*n):
                # Build the linear system
                # Unknowns: W[c,0], ..., W[c,40]
                # Equations: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                
                # T[a,b,c] = 1 iff exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
                t_c = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            cc = k*n+i
                            if cc == c:
                                t_c[(a,b)] = 1
                
                # Build matrix A (126 x 41) and vector t (126)
                # Use Gaussian elimination over rationals
                
                # For efficiency, only include equations where the row is nonzero
                rows = []
                rhs = []
                for a in range(n*m):
                    for b in range(m*p):
                        row = [U[a][r]*V[b][r] for r in range(41)]
                        if any(x != 0 for x in row):
                            rows.append(row)
                            rhs.append(t_c.get((a,b), 0))
                
                if not rows:
                    # No constraints, W[c,:] can be anything (set to 0)
                    continue
                
                # Gaussian elimination
                # This is 126x41, which is manageable
                num_rows = len(rows)
                num_cols = 41
                
                # Augmented matrix
                aug = [rows[i][:] + [rhs[i]] for i in range(num_rows)]
                
                # Eliminate
                rank = 0
                for col in range(num_cols):
                    # Find pivot
                    pivot = -1
                    for row in range(rank, num_rows):
                        if aug[row][col] != 0:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    
                    # Swap
                    aug[rank], aug[pivot] = aug[pivot], aug[rank]
                    
                    # Eliminate below
                    pivot_val = aug[rank][col]
                    for row in range(rank + 1, num_rows):
                        if aug[row][col] != 0:
                            factor = aug[row][col] / pivot_val
                            for j in range(col, num_cols + 1):
                                aug[row][j] -= factor * aug[rank][j]
                    
                    rank += 1
                    if rank == num_cols:
                        break
                
                # Check consistency
                for row in range(rank, num_rows):
                    if aug[row][num_cols] != 0:
                        feasible = False
                        break
                
                if not feasible:
                    break
                
                # Extract solution (free variables set to 0)
                w_c = [0] * num_cols
                for row in range(rank - 1, -1, -1):
                    # Find the leading column
                    lead_col = -1
                    for col in range(num_cols):
                        if aug[row][col] != 0:
                            lead_col = col
                            break
                    if lead_col == -1:
                        continue
                    w_c[lead_col] = aug[row][num_cols] / aug[row][lead_col]
                
                W[c] = w_c
            
            if feasible:
                # Verify
                if verify_decomposition(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a rank-41 decomposition
    result_41 = try_rank_41()
    if result_41 is not None:
        if verify_decomposition(result_41["U"], result_41["V"], result_41["W"]):
            return result_41
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
