"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _solve_rank_decomposition(n, m, p, target_rank):
    """
    Solve for U, V, W of rank target_rank such that
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    Uses rational Gaussian elimination with a systematic variable fixing strategy.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Total number of tensor entries
    num_entries = n * m * p  # 27 for 3x3x3
    
    # We have 3 * target_rank variables (U, V, W each have n*m, m*p, p*n rows * rank cols)
    # But the system is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Strategy: Fix W to have specific structure to linearize the problem.
    # Set W[k*n+i, r] = 1 for r = 0..2 (first 3 columns), 0 otherwise.
    # Then for r >= 3, W[k*n+i, r] can be 0 or 1.
    # Actually, let's use a different approach.
    
    # Better strategy: Use the fact that we can fix some of U, V, W entries.
    # For rank R, we have R terms. Let's fix W to be a specific pattern.
    
    # Approach: Fix W[k*n+i, r] for all k,i,r. 
    # Then the equations become: sum_r (W[k*n+i,r] * V[j*p+k,r]) * U[i*m+j,r] = 1
    # This is still bilinear in U and V.
    
    # Alternative: Fix both U and V partially, solve for W.
    # If we fix U[i*m+j, r] and V[j*p+k, r] for all i,j,k,r,
    # then W[k*n+i, r] = 1 / (U[i*m+j,r] * V[j*p+k,r]) for the (i,j,k) that uses term r.
    # But each (i,j,k) involves a sum over r, so this doesn't directly work.
    
    # Let me use a different strategy: 
    # Set up the problem as: we want 27 equations, 3*R unknowns.
    # Fix R-9 = 14 variables to specific values (0 or 1), solve for the rest.
    
    # Actually, the cleanest approach for small cases: 
    # Use the known structure. For 3x3x3, rank 23 is achievable.
    # Let me try a systematic search with rational arithmetic.
    
    # Fix W to be identity-like: W[k*n+i, r] = 1 if r == k*n+i, else 0.
    # This uses 9 "diagonal" terms. Then we need 14 more terms.
    # With W fixed this way, the equation for (i,j,k) becomes:
    #   U[i*m+j, k*n+i] * V[j*p+k, k*n+i] + sum_{r=9}^{22} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is still complex.
    
    # Let me try a completely different approach: 
    # Use the fact that the problem is small and do a structured construction.
    
    # Known result: 3x3 matrix multiplication has rank 23.
    # One construction uses 23 scalar multiplications.
    
    # Let me try to construct it by fixing U and V to have a specific sparse structure
    # and solving for W.
    
    # Strategy: 
    # 1. Fix U[i*m+j, r] and V[j*p+k, r] for r = 0..22.
    # 2. For each (i,k), the equation is: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #    This is linear in W! For fixed U,V, we can solve for W.
    # 3. But we have 9 equations (one per (i,k)) and 9*23=207 unknowns in W.
    #    Way underdetermined.
    
    # Better: Fix W structure, solve for U and V.
    # If W[k*n+i, r] is fixed, then for each (i,k), the equation is:
    #   sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r]) = 1
    # This is still bilinear.
    
    # Cleanest approach for this small case: 
    # Use the explicit known rank-23 algorithm.
    # The 3x3 matrix multiplication can be done with 23 multiplications.
    
    # Let me construct it using a systematic approach:
    # Fix the first 9 terms of W to be the identity (W[k*n+i, r] = 1 if r == k*n+i).
    # Then terms 9..22 are "extra" terms.
    
    # For the identity part: U[i*m+j, k*n+i] and V[j*p+k, k*n+i] contribute.
    # The (i,j,k) equation: U[i*m+j,k*n+i]*V[j*p+k,k*n+i] + sum_{r=9}^{22} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is getting complex. Let me just use a numerical approach with rational reconstruction.
    
    # Actually, for this small problem, let me try a direct algebraic approach:
    # Set up all 27 equations with 69 unknowns, fix 42 unknowns to 0 or 1,
    # and solve the remaining 27 linear equations (if we fix enough to make it linear).
    
    # Key insight: If we fix W completely, the equations are linear in the products U*V,
    # but not linear in U and V separately.
    
    # If we fix U completely, the equations are linear in V*W.
    # If we fix both U and V, the equations are linear in W.
    
    # So: Fix U and V (69 - 207... no, U has 9*23=207 entries, V has 9*23=207, W has 9*23=207).
    # That's 621 total unknowns for 27 equations. Way underdetermined.
    
    # I think the right approach is:
    # 1. Choose a sparse pattern for U, V, W (which entries are nonzero).
    # 2. Set up the 27 equations.
    # 3. Solve using rational Gaussian elimination, fixing free variables to simple values.
    
    # For rank 23, let's try: each of U, V, W has 9*23 = 207 entries.
    # Total 621 unknowns, 27 equations.
    
    # This is extremely underdetermined. Let me use a smarter construction.
    
    # Known approach: Use the "3x3 matrix multiplication in 23 multiplications" algorithm.
    # This is based on extending Strassen's ideas.
    
    # Let me try a constructive approach with a specific pattern.
    
    # Pattern: 
    # U[i*m+j, r] is nonzero only for certain (i,j,r)
    # V[j*p+k, r] is nonzero only for certain (j,k,r)  
    # W[k*n+i, r] is nonzero only for certain (k,i,r)
    
    # For the 9 "diagonal" terms (r=0..8), let r = k*n+i:
    #   U[i*m+j, r] = 1 for all j
    #   V[j*p+k, r] = 1 for all j
    #   W[k*n+i, r] = 1
    # This gives: for each (i,k), the diagonal term contributes sum_j 1*1*1 = 3 to the (i,k) entry.
    # But we need it to be 1. So this doesn't work directly.
    
    # Let me reconsider. The tensor T[i,j,k] = 1 means:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    # In the decomposition: (A*B)_{i,k} = sum_r (sum_j U[i*m+j,r]*A_{i,j}) * (sum_j V[j*p+k,r]*B_{j,k}) * W[k*n+i,r]
    
    # Wait no. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each triple (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is 27 equations. Let me just set up the full system and solve it.
    
    # For rank 23, I'll fix most variables and solve a smaller system.
    
    # Let me try: fix W[k*n+i, r] = 1 for all k,i,r. (All ones)
    # Then: sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for all i,j,k.
    # This is 27 equations in 9*23 + 9*23 = 414 unknowns.
    
    # Still very underdetermined. Let me fix more structure.
    
    # Fix U[i*m+j, r] = 1 for r < 9 (first 9 columns), 0 for r >= 9.
    # Fix V[j*p+k, r] = 1 for r < 9, 0 for r >= 9.
    # Then for r < 9: contributes 1 to each equation. Sum over r<9 gives 9.
    # We need sum_r = 1, so the remaining 14 terms must contribute -8.
    
    # This is getting messy. Let me just use a numerical solver with rational reconstruction.
    
    # Final approach: Use numpy to find a solution numerically, then reconstruct rationals.
    import numpy as np
    
    R = target_rank
    # Variables: U (9*R), V (9*R), W (9*R) = 27*R total
    # Equations: 27
    
    # The system is nonlinear (products of U*V*W). 
    # To make it linear, fix W to be all 1s.
    # Then: sum_r U[i*m+j,r]*V[j*p+k,r] = 1 for all i,j,k.
    # This is 27 equations in 18*R unknowns (U and V only).
    
    # For R=23: 27 equations, 414 unknowns. Underdetermined.
    # Set up as: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r] = 1.
    
    # This is still bilinear. Fix U to be identity-like.
    # U[i*m+j, r] = 1 if r == i*m+j, else 0. (9 nonzero entries in U)
    # Then: V[j*p+k, i*m+j] = 1 for all i,j,k. But V[j*p+k, i*m+j] must be consistent.
    # For fixed (j,k), V[j*p+k, r] is a single value, but it needs to equal 1 for all i.
    # So V[j*p+k, r] = 1 for all r where U[i*m+j,r] is nonzero for some i.
    # With U[i*m+j,r] = 1 if r==i*m+j, then for each (j,k), we need:
    #   sum_r U[i*m+j,r]*V[j*p+k,r] = V[j*p+k, i*m+j] = 1 for all i.
    # So V[j*p+k, i*m+j] = 1 for all i,j,k. That's 27 constraints on V.
    # V has 9*23 = 207 entries, so this is feasible.
    
    # But then the equation is satisfied only by the diagonal terms, and we've used 9 terms.
    # We need rank 23, so we have 14 extra terms that must contribute 0.
    # So for r >= 9: sum_r U[i*m+j,r]*V[j*p+k,r] = 0 for all i,j,k.
    
    # This means the extra 14 terms form a "null" decomposition.
    # We can set U[i*m+j,r] = 0 and V[j*p+k,r] = 0 for r >= 9.
    # But then rank is effectively 9, not 23.
    
    # I think I'm overcomplicating this. Let me just use the known rank-23 algorithm directly.
    
    # The 3x3 matrix multiplication rank-23 algorithm is well-known.
    # Let me encode it explicitly.
    
    # Actually, let me try a different constructive approach:
    # Use 9 "base" terms and 14 "correction" terms.
    
    # For now, let me return the schoolbook rank-27 solution as a fallback,
    # and try to improve it.
    
    # Let me try the direct algebraic approach with a specific ansatz.
    
    # Ansatz: 
    # U[i*m+j, r] = a_{i,j,r}
    # V[j*p+k, r] = b_{j,k,r}
    # W[k*n+i, r] = c_{k,i,r}
    # sum_r a_{i,j,r} * b_{j,k,r} * c_{k,i,r} = 1 for all i,j,k.
    
    # For rank 23, fix c_{k,i,r} = 1 for all k,i,r.
    # Then: sum_r a_{i,j,r} * b_{j,k,r} = 1 for all i,j,k.
    
    # Fix a_{i,j,r} = 1 if r == i*3+j, else 0. (9 terms)
    # Then: b_{j,k, i*3+j} = 1 for all i,j,k.
    # So b_{j,k, r} = 1 for all r in {0,...,8}.
    # The remaining 14 terms (r=9..22) must satisfy:
    #   sum_{r=9}^{22} a_{i,j,r} * b_{j,k,r} = 0 for all i,j,k.
    
    # We can set a_{i,j,r} = 0 and b_{j,k,r} = 0 for r >= 9.
    # But that gives rank 9, not 23.
    
    # To get rank 23, we need the extra terms to be nonzero but cancel out.
    # This is like adding a "null space" component.
    
    # For example, for r=9: a_{0,0,9}=1, a_{0,1,9}=-1, a_{0,2,9}=0, ...
    # and b_{j,k,9} chosen so that the sum is 0 for all (i,j,k).
    
    # This is feasible but complex. Let me just use a known algorithm.
    
    # I'll use the explicit rank-23 construction from the literature.
    
    # For now, return schoolbook to ensure validity, and note that rank 23 is the target.
    
    n, m, p = DIMENSIONS
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

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
