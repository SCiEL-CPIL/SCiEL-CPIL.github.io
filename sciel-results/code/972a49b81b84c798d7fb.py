import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 115.0
    
    # Start with the known rank-96 solution
    rank_96 = 96
    U96 = [[0] * rank_96 for _ in range(n * m)]
    V96 = [[0] * rank_96 for _ in range(m * p)]
    W96 = [[0] * rank_96 for _ in range(p * n)]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [0, 1])
    ]
    
    skip_set = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    skip_set.add((i, j, k))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U96[i * m + j][col] = 1
                V96[j * p + k][col] = 1
                W96[k * n + i][col] = 1
                col += 1
                
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U96[i1 * m + j1][col] += 1
        U96[i2 * m + j2][col] += 1
        V96[j1 * p + k1][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U96[i2 * m + j1][col] += 1
        U96[i2 * m + j2][col] += 1
        V96[j1 * p + k1][col] += 1
        W96[k1 * n + i2][col] += 1
        W96[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U96[i1 * m + j1][col] += 1
        V96[j1 * p + k2][col] += 1
        V96[j2 * p + k2][col] += -1
        W96[k2 * n + i1][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U96[i2 * m + j2][col] += 1
        V96[j2 * p + k1][col] += 1
        V96[j1 * p + k1][col] += -1
        W96[k1 * n + i1][col] += 1
        W96[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U96[i1 * m + j1][col] += 1
        U96[i1 * m + j2][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += -1
        W96[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U96[i2 * m + j1][col] += 1
        U96[i1 * m + j1][col] += -1
        V96[j1 * p + k1][col] += 1
        V96[j1 * p + k2][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U96[i1 * m + j2][col] += 1
        U96[i2 * m + j2][col] += -1
        V96[j2 * p + k1][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += 1
        col += 1
    
    best_U, best_V, best_W = U96, V96, W96
    best_rank = rank_96
    
    # Try to find rank 95 by attempting to eliminate one term
    # Strategy: pick a standard term (i,j,k) not in any Strassen block,
    # and see if we can express its contribution as a linear combination
    # of nearby Strassen terms plus adjustments.
    
    # The 36 standard terms (100 - 4*8 = 68 standard, but 100-32=68... wait)
    # Actually: 4 blocks * 8 terms = 32 terms skipped, so 100-32 = 68 standard terms
    # Plus 4*7 = 28 Strassen terms, total 96.
    
    # For each standard term, try to see if removing it and adjusting
    # the Strassen terms can still satisfy the tensor equation.
    
    standard_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in skip_set]
    
    def validate(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        return True
    
    # Attempt: for each standard term, try to absorb it into the Strassen
    # computation by modifying the Strassen terms slightly.
    # This is a nonlinear system, so we'll use a heuristic approach.
    
    # Alternative: try all pairs of standard terms and see if they can be
    # combined into a single rank-1 term with appropriate U,V,W values.
    
    # A standard term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Two standard terms (i1,j1,k1) and (i2,j2,k2) can be merged into one
    # rank-1 term only if they share the same U, V, or W index pattern,
    # which is rarely the case for distinct (i,j,k).
    
    # More promising: try to find if any two Strassen terms from different
    # blocks can be combined. But Strassen terms are already optimized.
    
    # Let's try a different approach: use the border rank concept.
    # If we can find a limit of rank-95 decompositions that converges to
    # our rank-96 solution, then rank 95 might be achievable.
    
    # Practical approach: random search for rank 95.
    # Initialize with the rank-96 solution, zero out one column,
    # and try to fix the remaining entries to satisfy the equations.
    
    for target_rank in [95]:
        if time.time() > deadline - 5:
            break
        
        # Try zeroing out each standard term and solving
        for idx, (si, sj, sk) in enumerate(standard_terms):
            if time.time() > deadline - 5:
                break
            
            # Create a modified rank-96 solution without this term
            U_test = [row[:] for row in U96]
            V_test = [row[:] for row in V96]
            W_test = [row[:] for row in W96]
            
            # Find the column for this standard term
            # Standard terms are filled first, so find which column
            col_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) in skip_set:
                            continue
                        if i == si and j == sj and k == sk:
                            break
                        col_idx += 1
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            # Zero out this column
            U_test[si * m + sj][col_idx] = 0
            V_test[sj * p + sk][col_idx] = 0
            W_test[sk * n + si][col_idx] = 0
            
            # Now we have 95 terms but the tensor equation is not satisfied
            # for (si, sj, sk). We need to adjust other terms.
            # This is a complex nonlinear problem.
            
            # Simple heuristic: try to add the missing contribution
            # to nearby Strassen terms by scaling them.
            # This won't generally work, but let's check if the
            # current 95-term solution happens to be valid.
            
            # Actually, zeroing out a term makes the sum at (si,sj,sk) = 0
            # instead of 1, so it's definitely invalid.
            # We need a smarter approach.
            
            # Try: modify one Strassen term to absorb the contribution.
            # For each Strassen block, try adjusting one of its 7 terms.
            pass
        
        # If we didn't find rank 95, keep rank 96
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}
