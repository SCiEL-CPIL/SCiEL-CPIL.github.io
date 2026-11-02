"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # The multiplication tensor T[i,j,k] = delta[i+k, j] where i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i + k == j) else 0
                # Wait, that's not right. Matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor is T[i,j,k] = 1 if this term contributes, which is always 1 for valid (i,j,k)
                # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity on the bilinear map)
                # The decomposition must satisfy: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Wait no. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
                # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
                # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple has coefficient 1.
                # The rank-r decomposition: T_{i,j,k} = sum_r u_{i,j,r} * v_{j,k,r} * w_{k,i,r}
                # where U has entries u_{i,j,r}, V has v_{j,k,r}, W has w_{k,i,r}
                # U shape: (n*m, rank) with index i*m+j
                # V shape: (m*p, rank) with index j*p+k  
                # W shape: (p*n, rank) with index k*n+i
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

def _build_rank14():
    """Build the known rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0: a00, 1: a01, 2: a10, 3: a11
    # Set U for first 7 terms (B1 part)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # Set U for next 7 terms (B2 part) - same forms
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    # Set V for first 7 terms (B1)
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    # Set V for next 7 terms (B2)
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    # Output coefficients W
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank13_search(deadline):
    """
    Try to find a rank-13 decomposition by merging one pair of Strassen products.
    
    Strategy: In the rank-14 solution, we have two independent Strassen decompositions.
    Try to share one bilinear product between the two blocks by finding a common
    linear form in A and a common linear form in B that can serve both blocks.
    
    The key insight: if we can find a single rank-1 term (u, v, w) such that
    u is a linear form in A, v is a linear form in B spanning both blocks,
    and w correctly adjusts the output, we can merge.
    
    We'll try all possible single-term modifications to the rank-14 decomposition.
    """
    n, m, p = 2, 2, 4
    rank = 13
    base = _build_rank14()
    U14 = [row[:] for row in base["U"]]
    V14 = [row[:] for row in base["V"]]
    W14 = [row[:] for row in base["W"]]
    
    # Try to eliminate one term by expressing it as a combination of others
    # We'll try removing each of the 14 terms and checking if the remaining 13
    # can be adjusted to still give the correct decomposition.
    
    # For each term r to remove, we need:
    # sum_{r' != r} u_{ij,r'} * v_{jk,r'} * w_{ki,r'} = 1 for all i,j,k
    # This means the removed term's contribution must be cancelled by adjusting
    # the remaining terms.
    
    # A more structured approach: try to merge two terms that have the same U form.
    # In our rank-14, terms 0-6 use the same U forms as terms 7-13.
    # If we can find a single V form that works for both, we save one rank.
    
    # Let's try: for each pair (r1 in 0..6, r2 in 7..13) where U[:,r1] == U[:,r2],
    # try to find a single V form v and W adjustment such that the merged term works.
    
    # Actually, let's try a more direct approach:
    # Try all 14 terms as the one to eliminate, and for each, try to solve
    # for adjustments to the remaining 13 terms.
    
    # Since exact rational solving is expensive, let's try a targeted approach:
    # The Strassen algorithm for 2x2*2x2 uses 7 terms. For 2x2*2x4, the
    # tensor rank is at most 14. Can we do 13?
    
    # Try: use a modified Strassen where one product is shared.
    # Specifically, try to use the same A-form for a product that covers
    # both b-blocks with a combined B-form.
    
    # Let's try a parametric search: assume rank 13 with a specific structure
    # where one term uses a combined B-form spanning both blocks.
    
    # Structure: 6 terms for block 1, 6 terms for block 2, 1 shared term
    # The shared term has U = some linear form in A, V = combined form in B,
    # W = combined output coefficient.
    
    # This is getting complex. Let's try a simpler randomized approach:
    # Start from rank-14, try to remove one term and solve for the rest.
    
    for remove_idx in range(14):
        if time.time() > deadline:
            return None
        
        # Build system: for each (i,j,k), we need
        # sum_{r != remove_idx} U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        # 
        # This is a system of 16 equations (one per i,j,k triple) in the
        # unknowns which are the entries of U, V, W (excluding column remove_idx).
        # But this is nonlinear (products of variables), so it's hard to solve directly.
        #
        # Instead, let's try a structured modification:
        # Keep the U forms fixed (they're the Strassen forms), and try to
        # adjust V and W to compensate.
        
        # Actually, let's try a different structured approach:
        # Merge two terms that have identical U columns.
        # In rank-14, terms 0&7, 1&8, 2&9, 3&10, 4&11, 5&12, 6&13 have same U.
        
        for pair_idx in range(7):
            if time.time() > deadline:
                return None
            r1 = pair_idx  # first block
            r2 = pair_idx + 7  # second block
            
            # Try to merge: use a single term with U[:,r1] (same as U[:,r2])
            # and a combined V form, and adjusted W.
            # 
            # The contribution of term r1 to output (i,k) is:
            #   sum_j U[i*m+j, r1] * V[j*p+k, r1] * W[k*n+i, r1]
            # The contribution of term r2 is:
            #   sum_j U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
            #
            # Since U[:,r1] == U[:,r2], if we set V_merged[j*p+k] = V[j*p+k,r1] + V[j*p+k,r2]
            # and W_merged[k*n+i] = W[k*n+i,r1] (if W is the same) or adjust...
            #
            # But W might differ. Let's check if W[:,r1] == W[:,r2].
            
            w_same = all(W14[k][r1] == W14[k][r2] for k in range(p*n))
            v_same = all(V14[k][r1] == V14[k][r2] for k in range(m*p))
            
            if w_same and v_same:
                # These two terms are identical - just remove one!
                # But this would mean the rank-14 had redundant terms, which it shouldn't.
                pass
            
            # Try: keep U[:,r1], set V = V[:,r1] + V[:,r2], W = W[:,r1]
            # This works only if W[:,r1] == W[:,r2]
            # Or: keep U[:,r1], V = V[:,r1], W = W[:,r1] + W[:,r2]
            # This works only if V[:,r1] == V[:,r2]
            
            # Let's try both options:
            # Option A: merge V, keep W of r1 (requires W[:,r1]==W[:,r2] or adjust)
            # Option B: merge W, keep V of r1 (requires V[:,r1]==V[:,r2] or adjust)
            
            # For option A: new term has U=U[:,r1], V=V[:,r1]+V[:,r2], W=W[:,r1]
            # Contribution: sum_j U[i*m+j,r1]*(V[j*p+k,r1]+V[j*p+k,r2])*W[k*n+i,r1]
            # = original r1 contribution + sum_j U[i*m+j,r1]*V[j*p+k,r2]*W[k*n+i,r1]
            # Original r2 contribution: sum_j U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]
            # = sum_j U[i*m+j,r1]*V[j*p+k,r2]*W[k*n+i,r2]
            # So we need W[:,r1] == W[:,r2] for this to work.
            
            if w_same:
                # Build rank-13: remove r2, modify V of r1 to be sum
                U13 = [row[:] for row in U14]
                V13 = [row[:] for row in V14]
                W13 = [row[:] for row in W14]
                
                # Remove column r2
                U13 = [row[:r2] + row[r2+1:] for row in U13]
                V13 = [row[:r2] + row[r2+1:] for row in V13]
                W13 = [row[:r2] + row[r2+1:] for row in W13]
                
                # Adjust V of r1 (which is now at index r1 in the new array)
                # r1 is still at index r1 since r2 > r1
                for k in range(m*p):
                    V13[k][r1] = V14[k][r1] + V14[k][r2]
                
                # Verify
                if _verify_decomposition(U13, V13, W13, n, m, p):
                    return {"U": U13, "V": V13, "W": W13}
            
            # Option B: merge W, keep V of r1
            if v_same:
                U13 = [row[:] for row in U14]
                V13 = [row[:] for row in V14]
                W13 = [row[:] for row in W14]
                
                U13 = [row[:r2] + row[r2+1:] for row in U13]
                V13 = [row[:r2] + row[r2+1:] for row in V13]
                W13 = [row[:r2] + row[r2+1:] for row in W13]
                
                for k in range(p*n):
                    W13[k][r1] = W14[k][r1] + W14[k][r2]
                
                if _verify_decomposition(U13, V13, W13, n, m, p):
                    return {"U": U13, "V": V13, "W": W13}
    
    # Try randomized perturbation: start from rank-14, randomly perturb one column
    # and try to find a rank-13 solution via iterative refinement.
    # This is a heuristic approach.
    
    # Try: for each of the 14 terms, try to express it as a linear combination
    # of the other 13 terms in the tensor decomposition sense.
    # This requires solving a nonlinear system, which is hard.
    # 
    # Instead, try a simple randomized search: randomly set one column to zero
    # and adjust nearby columns to compensate.
    
    for _ in range(200):
        if time.time() > deadline:
            return None
        
        # Pick a random term to eliminate
        elim = random.randint(0, 13)
        
        # Build rank-13 by zeroing out column elim and trying to adjust
        U13 = [row[:] for row in U14]
        V13 = [row[:] for row in V14]
        W13 = [row[:] for row in W14]
        
        # Zero out column elim
        for i in range(n*m):
            U13[i][elim] = 0
        for i in range(m*p):
            V13[i][elim] = 0
        for i in range(p*n):
            W13[i][elim] = 0
        
        # Try to compensate by adjusting a random other column
        if elim < 13:
            adjust = random.randint(0, 12)
            # Scale the adjust column to try to compensate
            # This is a very crude heuristic
            for i in range(n*m):
                U13[i][adjust] = U14[i][adjust] + random.randint(-1, 1)
            for i in range(m*p):
                V13[i][adjust] = V14[i][adjust] + random.randint(-1, 1)
            for i in range(p*n):
                W13[i][adjust] = W14[i][adjust] + random.randint(-1, 1)
        else:
            adjust = random.randint(0, 12)
            for i in range(n*m):
                U13[i][adjust] = U14[i][adjust] + random.randint(-1, 1)
            for i in range(m*p):
                V13[i][adjust] = V14[i][adjust] + random.randint(-1, 1)
            for i in range(p*n):
                W13[i][adjust] = W14[i][adjust] + random.randint(-1, 1)
        
        # Remove the zero column
        U13 = [row[:elim] + row[elim+1:] for row in U13]
        V13 = [row[:elim] + row[elim+1:] for row in V13]
        W13 = [row[:elim] + row[elim+1:] for row in W13]
        
        if _verify_decomposition(U13, V13, W13, n, m, p):
            return {"U": U13, "V": V13, "W": W13}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # First try to find rank-13
    result = _try_rank13_search(deadline)
    if result is not None:
        return result
    
    # Fall back to rank-14
    return _build_rank14()
# EVOLVE-BLOCK-END
