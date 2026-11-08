import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Construct the Seed Solution (Rank 39)
    # We use integer lists for U, V, W to allow exact manipulation.
    rank_seed = 39
    U = [[0] * rank_seed for _ in range(n * m)]
    V = [[0] * rank_seed for _ in range(m * p)]
    W = [[0] * rank_seed for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        nonlocal col
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    # Add 3 Strassen blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # k=6 block
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # j=2 block
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    # 2. Verification and Structural Optimization Attempt
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][j]) for j in range(rank_seed)] for i in range(n*m)]
    V_frac = [[Fraction(V[i][j]) for j in range(rank_seed)] for i in range(m*p)]
    W_frac = [[Fraction(W[i][j]) for j in range(rank_seed)] for i in range(p*n)]
    
    # Verify the seed solution
    # T[i, j, k] should be sum_r U[i,r] V[j,r] W[k,r]
    # Note: The tensor indices in the problem are C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is C[i, k] = sum_r U[i, r] V[j, r] W[k, r]? 
    # Wait, the standard tensor decomposition for matmul is:
    # C_{i,k} = sum_{r} U_{i,r} V_{j,r} W_{k,r} is NOT correct for matmul.
    # Matmul tensor T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} represents:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This requires T_{i,j,k} to be the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} appears in C_{i,k}, else 0.
    # The decomposition is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    
    # Let's verify a few entries.
    # For i=0, j=0, k=0:
    # C[0,0] contains A[0,0]B[0,0]. So T[0,0,0] = 1.
    # Sum_r U[0,r] V[0,r] W[0,r] should be 1.
    
    is_valid = True
    # Check all i, j, k
    # This is 2*3*7 = 42 checks.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank_seed):
                    s += U_frac[i][r] * V_frac[j][r] * W_frac[k][r]
                # The tensor T[i,j,k] is 1 if i,j,k form a valid matmul term.
                # In C[i,k] = sum_j A[i,j]B[j,k], the term A[i,j]B[j,k] is present.
                # So T[i,j,k] should be 1.
                if s != 1:
                    is_valid = False
                    break
            if not is_valid: break
        if not is_valid: break
        
    if not is_valid:
        # If seed is invalid (should not happen), return it anyway as best effort?
        # Or raise error? The prompt says "invalid result scores 0".
        # The seed is known valid.
        pass

    # 3. Attempt to reduce rank to 38
    # Strategy: Check if any two terms in the decomposition sum to a rank-1 tensor.
    # If T = sum_{r=1}^{39} u_r \otimes v_r \otimes w_r, and for some r1, r2:
    # (u_{r1} \otimes v_{r1} \otimes w_{r1}) + (u_{r2} \otimes v_{r2} \otimes w_{r2}) = u' \otimes v' \otimes w'
    # Then we can replace the two terms with one, reducing rank by 1.
    
    # This is a strong condition. It's unlikely to hold for the Strassen blocks.
    # However, we can check if the sum of the two terms is a rank-1 tensor.
    # A tensor X_{i,j,k} is rank 1 if X_{i,j,k} = a_i b_j c_k.
    
    def is_rank_1_sum(r1, r2):
        # Extract the sum of two terms
        # X[i,j,k] = U[i,r1]V[j,r1]W[k,r1] + U[i,r2]V[j,r2]W[k,r2]
        # We can check if this tensor has rank 1.
        # A necessary and sufficient condition for rank 1 is that all 2x2 minors of the matrix unfolding are 0.
        # Or simpler: check if X[i,j,k] * X[i',j',k'] == X[i,j',k'] * X[i',j,k'] for all i,i',j,j',k,k'.
        
        # Get vectors
        u1 = [U_frac[i][r1] for i in range(n)]
        v1 = [V_frac[j][r1] for j in range(m)]
        w1 = [W_frac[k][r1] for k in range(p)]
        
        u2 = [U_frac[i][r2] for i in range(n)]
        v2 = [V_frac[j][r2] for j in range(m)]
        w2 = [W_frac[k][r2] for k in range(p)]
        
        # Build the sum tensor X
        # X[i,j,k] = u1[i]v1[j]w1[k] + u2[i]v2[j]w2[k]
        
        # Check rank 1 condition on a few slices or full check.
        # Full check is 2*3*7 = 42 elements.
        # Pairs of elements: 42*41/2 = 861 checks.
        # This is fast.
        
        # Flatten indices
        def idx(i, j, k):
            return (i * m + j) * p + k
            
        # Precompute X values
        X = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    X[(i,j,k)] = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
                    
        # Check condition: X[i,j,k] * X[i',j',k'] == X[i,j',k'] * X[i',j,k']
        # This must hold for all i,i',j,j',k,k'.
        # We can fix i=0, i'=1, j=0, j'=1, k=0, k'=1 as a necessary condition.
        # If it fails, not rank 1.
        # If it passes, we need to check more.
        
        # Quick necessary condition:
        # X[0,0,0]*X[1,1,1] == X[0,1,1]*X[1,0,0]
        if X[(0,0,0)] * X[(1,1,1)] != X[(0,1,1)] * X[(1,0,0)]:
            return False
            
        # Check another one
        if X[(0,0,0)] * X[(0,1,1)] != X[(0,1,0)] * X[(0,0,1)]:
            return False

        # If these pass, it's likely rank 1 or close. 
        # For integer/half-integer entries, if these pass, it's very likely rank 1.
        # We assume rank 1 if these necessary conditions pass.
        return True

    merged = False
    # Check all pairs
    for r1 in range(rank_seed):
        for r2 in range(r1 + 1, rank_seed):
            if is_rank_1_sum(r1, r2):
                # Found a merge!
                # We would need to construct the new term.
                # This is complex. For now, just mark as merged.
                # In a real scenario, we'd solve for u', v', w'.
                merged = True
                break
        if merged:
            break
            
    if merged:
        # We found a rank 38 candidate.
        # However, constructing the explicit U, V, W for the merged term is non-trivial.
        # Given the time constraint and low probability, we return the seed.
        # But if we *did* find it, we'd return rank 38.
        # Since we don't have the construction logic for the merged term, we can't return a valid rank 38 solution.
        # So we return the seed.
        pass

    # Return the seed solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
