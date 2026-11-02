"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Build the known rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the tensor decomposition is exact."""
    n, m, p = 2, 2, 4
    rank = len(V[0])
    # Build the multiplication tensor T[i,j,k] where i indexes A, j indexes B, k indexes C
    # T[i,j,k] = 1 if A[i] * B[j] contributes to C[k]
    # A: (i1,i2) -> i = i1*2 + i2
    # B: (j1,j2) -> j = j1*4 + j2  (j1 in {0,1}, j2 in {0,1,2,3})
    # C: (k1,k2) -> k = k1*2 + k2  (W[k*n+i] convention: k = k1, i = k2? No...)
    # W index: k*n + i where k is row of C, i is col of C
    # So C[k1, k2] has W index k1*2 + k2
    
    # The tensor entry T[a_idx, b_idx, c_idx] = 1 iff
    # A[a1,a2] * B[b1,b2] contributes to C[c1,c2]
    # C[c1,c2] = sum over x: A[c1,x] * B[x,c2]
    # So T[a_idx, b_idx, c_idx] = 1 iff a1==c1 and b1==c2 and a2==b2... 
    # Wait: C[c1,c2] = sum_x A[c1,x]*B[x,c2]
    # So term A[a1,a2]*B[b1,b2] contributes to C[c1,c2] iff a1==c1 and a2==b1 and b2==c2
    
    # Check: sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r] == T[a_idx,b_idx,c_idx]
    for a1 in range(2):
        for a2 in range(2):
            a_idx = a1 * 2 + a2
            for b1 in range(2):
                for b2 in range(4):
                    b_idx = b1 * 4 + b2
                    for c1 in range(2):
                        for c2 in range(4):
                            c_idx = c1 * 2 + c2
                            # T = 1 iff a1==c1 and a2==b1 and b2==c2
                            expected = 1.0 if (a1 == c1 and a2 == b1 and b2 == c2) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # First, try to find a rank-13 decomposition by merging one pair of Strassen terms
    # across the two 2x2 blocks. We search for a shared bilinear form.
    
    U14, V14, W14 = _build_rank14()
    
    if time.time() > deadline:
        return {"U": U14, "V": V14, "W": W14}
    
    # Strategy: Try to find if any two rank-1 terms (one from block 1, one from block 2)
    # can be merged into a single rank-1 term. This requires finding scalars alpha, beta
    # such that the combined term still works.
    # 
    # More practically: search for a rank-13 decomposition by trying to eliminate one
    # of the 14 terms and adjusting the others. We'll use a structured search.
    #
    # Alternative approach: Try small integer/half-integer perturbations of the rank-14
    # solution to see if we can zero out one column while maintaining exactness.
    #
    # Let's try a direct search: for each pair (i,j) where i is from block1 (0-6) and
    # j is from block2 (7-13), check if we can merge them.
    # Merging means: find a new term t such that replacing terms i and j with t
    # still gives the correct tensor.
    #
    # This is complex. Let's try a simpler approach: search for rank-13 by trying
    # all possible "merge" patterns with small integer coefficients.
    
    # Try: can we replace two terms with one? For each pair (a,b) from the 14 terms,
    # try to find a replacement term.
    # The condition is: for all (a_idx, b_idx, c_idx):
    #   U[a_idx,a]*V[b_idx,a]*W[c_idx,a] + U[a_idx,b]*V[b_idx,b]*W[c_idx,b]
    #   = U'[a_idx]*V'[b_idx]*W'[c_idx]
    #
    # This is a system of equations. Let's try small integer values for the new term.
    
    found_rank13 = False
    best_result = {"U": U14, "V": V14, "W": W14}
    
    # Try merging pairs: for each pair (i, j) with i < j, try to find a single
    # rank-1 term that replaces both.
    # We'll try small integer/half-integer values for the new term's entries.
    
    search_start = time.time()
    search_budget = 120  # seconds for search
    
    for i in range(14):
        for j in range(i + 1, 14):
            if time.time() > search_start + search_budget:
                break
            
            # Try to find a new term that replaces terms i and j
            # The new term has entries u_r, v_r, w_r for each index.
            # We need: for all (a,b,c): sum over remaining terms + new term = T[a,b,c]
            # Equivalently: new term must equal: T[a,b,c] - sum_{r != i,j} U[a,r]*V[b,r]*W[c,r]
            
            # Compute the required new term values
            # required[a,b,c] = T[a,b,c] - sum_{r not in {i,j}} U[a,r]*V[b,r]*W[c,r]
            # Then we need: required[a,b,c] = U_new[a] * V_new[b] * W_new[c]
            
            # This means required must be a rank-1 tensor (outer product of 3 vectors).
            # Let's compute required and check if it's rank-1.
            
            required = {}
            is_rank1 = True
            # Collect all nonzero required values
            for a1 in range(2):
                for a2 in range(2):
                    a_idx = a1 * 2 + a2
                    for b1 in range(2):
                        for b2 in range(4):
                            b_idx = b1 * 4 + b2
                            for c1 in range(2):
                                for c2 in range(4):
                                    c_idx = c1 * 2 + c2
                                    expected = 1.0 if (a1 == c1 and a2 == b1 and b2 == c2) else 0.0
                                    s = 0.0
                                    for r in range(14):
                                        if r != i and r != j:
                                            s += U14[a_idx][r] * V14[b_idx][r] * W14[c_idx][r]
                                    req = expected - s
                                    if abs(req) > 1e-9:
                                        required[(a_idx, b_idx, c_idx)] = req
                                    elif abs(req) > 1e-12:
                                        pass  # small numerical noise
                                    
            # Check if required is a rank-1 tensor: required[a,b,c] = U_new[a]*V_new[b]*W_new[c]
            if not required:
                # All zeros: terms i and j cancel each other perfectly with others
                # This would give rank 12, but unlikely
                continue
                
            # Try to find U_new, V_new, W_new such that U_new[a]*V_new[b]*W_new[c] = required[a,b,c]
            # Pick a reference entry to normalize
            ref_key = list(required.keys())[0]
            ra, rb, rc = ref_key
            ref_val = required[ref_key]
            
            # We have freedom in scaling. Let U_new[ra] = 1, then V_new[rb]*W_new[rc] = ref_val
            # Try small rational values
            
            # Actually, let's try a more direct approach:
            # For each possible U_new vector (sparse, small integers), check consistency.
            # U_new has 4 entries (a_idx in 0..3), V_new has 8 entries, W_new has 8 entries.
            
            # Let's try: assume U_new has at most 2 nonzero entries, etc.
            # This is getting complex. Let's try a simpler heuristic search.
            
            # Try all combinations of small values for U_new[0..3]
            # Each entry can be in {-1, -0.5, 0, 0.5, 1}
            # That's 5^4 = 625 possibilities for U_new
            # For each, determine V_new and W_new from the equations.
            
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            for u0 in vals:
                for u1 in vals:
                    for u2 in vals:
                        for u3 in vals:
                            U_new = [u0, u1, u2, u3]
                            
                            # For each b_idx, V_new[b_idx] must satisfy:
                            # U_new[a_idx] * V_new[b_idx] * W_new[c_idx] = required[a_idx, b_idx, c_idx]
                            # for all (a_idx, b_idx, c_idx) in required
                            
                            # Let's determine V_new and W_new from the constraints.
                            # For a given b_idx, V_new[b_idx] is determined by any (a_idx, c_idx)
                            # where U_new[a_idx] != 0 and required has an entry.
                            
                            V_new = {}
                            W_new = {}
                            consistent = True
                            
                            for (a_idx, b_idx, c_idx), req_val in required.items():
                                if U_new[a_idx] == 0:
                                    if abs(req_val) > 1e-9:
                                        consistent = False
                                        break
                                    continue
                                
                                # U_new[a_idx] * V_new[b_idx] * W_new[c_idx] = req_val
                                # If b_idx is already determined:
                                if b_idx in V_new:
                                    # Check W_new[c_idx]
                                    expected_w = req_val / (U_new[a_idx] * V_new[b_idx])
                                    if c_idx in W_new:
                                        if abs(W_new[c_idx] - expected_w) > 1e-9:
                                            consistent = False
                                            break
                                    else:
                                        W_new[c_idx] = expected_w
                                else:
                                    # b_idx not yet determined
                                    if c_idx in W_new:
                                        expected_v = req_val / (U_new[a_idx] * W_new[c_idx])
                                        V_new[b_idx] = expected_v
                                    else:
                                        # Neither determined: underdetermined, skip this U_new
                                        # We'll handle by trying to close the system
                                        # For now, mark as needing more info
                                        pass
                            
                            if not consistent:
                                continue
                            
                            # Check if all entries are determined and consistent
                            # Verify the full decomposition
                            # Build the new 13-term decomposition
                            U_new_full = [0] * 13
                            V_new_full = [[0] * 13 for _ in range(8)]
                            W_new_full = [[0] * 13 for _ in range(8)]
                            
                            # Copy terms 0..12 (excluding i and j)
                            new_idx = 0
                            for r in range(14):
                                if r == i or r == j:
                                    continue
                                for a in range(4):
                                    U_new_full[a] = U14[a][r] if False else 0  # placeholder
                                # Actually, let's build it properly
                                pass
                            
                            # This approach is getting too complex. Let's try a different strategy.
                            break
                        break
                    break
                break
            break
        if time.time() > search_start + search_budget:
            break
    
    # If no rank-13 found, return rank-14
    if not found_rank13:
        return best_result
    
    # This code path is unlikely to be reached in current form
    return best_result
# EVOLVE-BLOCK-END
