"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _decompose_strassen(n, m, p):
    """Build the known rank-17 decomposition (Strassen + direct)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Strassen for A1(2x2) x B1_left(2x2)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Direct for A1(2x2) x B1_right(2x1)
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Direct for A2(2x1) x B2(1x3)
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_rank16_systematic(deadline):
    """
    Systematic search for rank-16 by trying all possible 2-term groupings
    of the 10 "direct" terms with various coefficient patterns.
    """
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    # The 10 direct terms in the rank-17 decomposition
    direct_terms = [
        (0, 2, 4),  # term 7: A[0,0]*B[0,2]->C[0,2]
        (1, 5, 4),  # term 8: A[0,1]*B[1,2]->C[0,2]
        (3, 2, 5),  # term 9: A[1,0]*B[0,2]->C[1,2]
        (4, 5, 5),  # term 10: A[1,1]*B[1,2]->C[1,2]
        (2, 6, 0),  # term 11: A[0,2]*B[2,0]->C[0,0]
        (2, 7, 2),  # term 12: A[0,2]*B[2,1]->C[0,1]
        (2, 8, 4),  # term 13: A[0,2]*B[2,2]->C[0,2]
        (5, 6, 1),  # term 14: A[1,2]*B[2,0]->C[1,0]
        (5, 7, 3),  # term 15: A[1,2]*B[2,1]->C[1,1]
        (5, 8, 5),  # term 16: A[1,2]*B[2,2]->C[1,2]
    ]
    
    target_set = set(direct_terms)
    
    # Try all pairs of direct terms to combine into one rank-1 term
    # with various coefficient patterns
    coeff_patterns = [
        # (u_coeffs, v_coeffs, w_coeffs) for the two targets
        # Pattern: both get coefficient 1 in all three factors
        ({1:1, 2:1}, {1:1, 2:1}, {1:1, 2:1}),
        # Pattern: one gets 1, other gets 1, but different factor gets 2 and 0.5
        ({1:1, 2:1}, {1:1, 2:1}, {1:2, 2:0.5}),
        ({1:1, 2:1}, {1:2, 2:0.5}, {1:1, 2:1}),
        ({1:2, 2:0.5}, {1:1, 2:1}, {1:1, 2:1}),
        # Pattern: using -1 for cancellation
        ({1:1, 2:1}, {1:1, 2:1}, {1:1, 2:-1}),
        ({1:1, 2:1}, {1:1, 2:-1}, {1:1, 2:1}),
        ({1:1, 2:-1}, {1:1, 2:1}, {1:1, 2:1}),
    ]
    
    for i in range(len(direct_terms)):
        for j in range(i+1, len(direct_terms)):
            if time.time() > deadline:
                return None
            
            a1, b1, c1 = direct_terms[i]
            a2, b2, c2 = direct_terms[j]
            
            for (u_pat, v_pat, w_pat) in coeff_patterns:
                # Build candidate
                U = np.zeros((n_a, 16))
                V = np.zeros((n_b, 16))
                W = np.zeros((n_c, 16))
                
                # Strassen part
                U[:, :7] = strassen_U[:, :7]
                V[:, :7] = strassen_V[:, :7]
                W[:, :7] = strassen_W[:, :7]
                
                # Combined term at column 7
                U[a1, 7] = u_pat[1]
                U[a2, 7] = u_pat[2]
                V[b1, 7] = v_pat[1]
                V[b2, 7] = v_pat[2]
                W[c1, 7] = w_pat[1]
                W[c2, 7] = w_pat[2]
                
                # Remaining 8 direct terms
                remaining = [direct_terms[k] for k in range(10) if k != i and k != j]
                for k, (a, b, c) in enumerate(remaining):
                    col = 8 + k
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                
                if _verify_decomposition(U, V, W, T):
                    return U, V, W
    
    return None

def _try_rank16_two_combined(deadline):
    """
    Try rank-16 with two combined terms (each covering 2 direct terms),
    leaving 6 direct terms.
    """
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    direct_terms = [
        (0, 2, 4), (1, 5, 4), (3, 2, 5), (4, 5, 5),
        (2, 6, 0), (2, 7, 2), (2, 8, 4),
        (5, 6, 1), (5, 7, 3), (5, 8, 5),
    ]
    target_set = set(direct_terms)
    
    # Find all valid single-combination pairs
    valid_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            a1, b1, c1 = direct_terms[i]
            a2, b2, c2 = direct_terms[j]
            u_set = {a1, a2}
            v_set = {b1, b2}
            w_set = {c1, c2}
            produced = set()
            for a in u_set:
                for b in v_set:
                    for c in w_set:
                        produced.add((a, b, c))
            if produced & target_set == {(a1,b1,c1), (a2,b2,c2)}:
                valid_pairs.append((i, j))
    
    # Try all pairs of valid pairs that are disjoint
    for pi in range(len(valid_pairs)):
        if time.time() > deadline:
            return None
        for pj in range(pi+1, len(valid_pairs)):
            p1 = valid_pairs[pi]
            p2 = valid_pairs[pj]
            if set(p1) & set(p2):
                continue
            
            i1, j1 = p1
            i2, j2 = p2
            a1, b1, c1 = direct_terms[i1]
            a2, b2, c2 = direct_terms[j1]
            a3, b3, c3 = direct_terms[i2]
            a4, b4, c4 = direct_terms[j2]
            
            U = np.zeros((n_a, 16))
            V = np.zeros((n_b, 16))
            W = np.zeros((n_c, 16))
            
            U[:, :7] = strassen_U[:, :7]
            V[:, :7] = strassen_V[:, :7]
            W[:, :7] = strassen_W[:, :7]
            
            # Combined term 1
            U[a1, 7] = 1; U[a2, 7] = 1
            V[b1, 7] = 1; V[b2, 7] = 1
            W[c1, 7] = 1; W[c2, 7] = 1
            
            # Combined term 2
            U[a3, 8] = 1; U[a4, 8] = 1
            V[b3, 8] = 1; V[b4, 8] = 1
            W[c3, 8] = 1; W[c4, 8] = 1
            
            # Remaining 6 direct terms
            used = {i1, j1, i2, j2}
            remaining = [direct_terms[k] for k in range(10) if k not in used]
            for k, (a, b, c) in enumerate(remaining):
                col = 9 + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
            
            if _verify_decomposition(U, V, W, T):
                return U, V, W
    
    return None

def _try_rank16_generalized_strassen(deadline):
    """
    Try a different Strassen-like decomposition that covers more of the tensor.
    Instead of 2x2 x 2x2, try to find a rank-8 scheme for a larger subproblem.
    
    Alternative: Treat the problem as multiplying [A1 | A2] x [B1; B2]
    where A1 is 2x2, A2 is 2x1, B1 is 2x3, B2 is 1x3.
    
    Use a rank-8 scheme for the 2x3 x 3x2 part (if exists) + rank-8 for 2x2 x 2x2.
    Total: 16.
    
    Actually, let's try: decompose as
    C = A1*B1 + A2*B2
    where A1 is 2x2 (first 2 cols of A), B1 is 2x3 (first 2 rows of B)
    A2 is 2x1 (last col of A), B2 is 1x3 (last row of B)
    
    A1*B1: 2x2 x 2x3 -> need rank for this
    A2*B2: 2x1 x 1x3 -> rank 6 (direct)
    
    So we need A1*B1 to have rank 10. That's the 2x2 x 2x3 problem.
    The 2x2 x 2x2 part has rank 7 (Strassen), and 2x2 x 2x1 has rank 4 (direct).
    Total: 7+4+6 = 17. Same as before.
    
    Let me try a completely different approach: use the fact that
    n=2 is small and try to find a rank-16 scheme by solving the
    tensor decomposition equations directly with integer/half-integer constraints.
    """
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(2, 3, 3)
    
    # Get all 18 tensor entries
    tensor_entries = []
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*2 + i
                tensor_entries.append((a, b, c))
    
    # Try rank-16 with a structured approach:
    # Use 7 Strassen terms for the 2x2x2 core
    # Then find 9 terms for the remaining 11 tensor entries
    # (since Strassen covers 8 entries with 7 terms, we have 18-8=10 remaining)
    # Wait, Strassen covers all 8 entries of the 2x2x2 subproblem.
    # Remaining: 18-8 = 10 entries, need 9 terms.
    
    # The 10 remaining entries are exactly the direct_terms from before.
    # We need to cover them with 9 rank-1 terms.
    # This means we need one term that covers 2 entries, or use cancellation.
    
    # Let's try a more general search: allow any 9 terms to cover the 10 entries
    # by trying all ways to split 10 entries into 9 groups (one group of 2, eight groups of 1)
    # and checking if the group of 2 can be covered by a single rank-1 term.
    
    # This is equivalent to _try_rank16_systematic but let's also try
    # terms that have support on entries OUTSIDE the 10 direct entries
    # (i.e., they also hit Strassen-covered entries, which must cancel to 0).
    
    # For a rank-1 term to be valid, its product must be 0 on all Strassen-covered
    # entries and 1 on its target entries.
    
    # Strassen-covered entries (the 2x2x2 core):
    strassen_entries = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i*3 + j
                b = j*3 + k
                c = k*2 + i
                strassen_entries.add((a, b, c))
    
    # For a rank-1 term with support sets U_sup, V_sup, W_sup:
    # For each (a,b,c) in strassen_entries:
    #   U[a]*V[b]*W[c] must be 0
    # This means: for each Strassen entry (a,b,c), at least one of U[a], V[b], W[c] is 0.
    
    # Let's enumerate possible support patterns for a rank-1 term that covers
    # two of the 10 direct entries and is zero on all Strassen entries.
    
    direct_terms = [
        (0, 2, 4), (1, 5, 4), (3, 2, 5), (4, 5, 5),
        (2, 6, 0), (2, 7, 2), (2, 8, 4),
        (5, 6, 1), (5, 7, 3), (5, 8, 5),
    ]
    
    for i in range(10):
        for j in range(i+1, 10):
            if time.time() > deadline:
                return None
            
            a1, b1, c1 = direct_terms[i]
            a2, b2, c2 = direct_terms[j]
            
            # Support must include a1,a2 in U; b1,b2 in V; c1,c2 in W
            # And must be zero on all Strassen entries.
            
            # Try: U support = {a1, a2}, V support = {b1, b2}, W support = {c1, c2}
            # Check if this is zero on Strassen entries
            u_sup = {a1, a2}
            v_sup = {b1, b2}
            w_sup = {c1, c2}
            
            valid = True
            for (sa, sb, sc) in strassen_entries:
                if sa in u_sup and sb in v_sup and sc in w_sup:
                    # This Strassen entry would be hit by the rank-1 term
                    # We need the coefficient product to be 0
                    # With all coefficients non-zero, this fails
                    valid = False
                    break
            
            if valid:
                # Check that the product hits exactly the two target entries
                produced = set()
                for a in u_sup:
                    for b in v_sup:
                        for c in w_sup:
                            produced.add((a, b, c))
                
                if produced == {(a1,b1,c1), (a2,b2,c2)}:
                    # Build the full rank-16 decomposition
                    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
                    U = np.zeros((n_a, 16))
                    V = np.zeros((n_b, 16))
                    W = np.zeros((n_c, 16))
                    
                    U[:, :7] = strassen_U[:, :7]
                    V[:, :7] = strassen_V[:, :7]
                    W[:, :7] = strassen_W[:, :7]
                    
                    U[a1, 7] = 1; U[a2, 7] = 1
                    V[b1, 7] = 1; V[b2, 7] = 1
                    W[c1, 7] = 1; W[c2, 7] = 1
                    
                    used = {i, j}
                    remaining = [direct_terms[k] for k in range(10) if k not in used]
                    for k, (a, b, c) in enumerate(remaining):
                        col = 8 + k
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                    
                    if _verify_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def _try_rank16_with_cancellation(deadline):
    """
    Try rank-16 where one term has non-trivial coefficients that cancel
    unwanted contributions on Strassen-covered entries.
    
    Idea: A rank-1 term might hit a Strassen entry, but if its coefficient
    product is 0 there (due to a zero in one of the factors), it's fine.
    More generally, we can have terms that partially overlap with Strassen
    terms if the net contribution is correct.
    """
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(2, 3, 3)
    
    # This is a more general search. Let's try to find ANY rank-16 decomposition
    # by using a greedy/iterative approach.
    
    # Actually, let's try a specific known construction:
    # The 2x3 x 3x3 matrix multiplication can be viewed as:
    # C = A * B where A is 2x3, B is 3x3
    # 
    # Split A = [A_left(2x2) | A_right(2x1)]
    # Split B = [B_top(2x3); B_bottom(1x3)]
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: 2x2 x 2x3, rank <= 7+4 = 11 (Strassen + direct)
    # A_right * B_bottom: 2x1 x 1x3, rank = 6
    # Total: 17
    #
    # Alternative split:
    # A = [A_col0(2x1) | A_col1(2x1) | A_col2(2x1)]
    # B = [B_row0(1x3); B_row1(1x3); B_row2(1x3)]
    # C = sum_j A_colj * B_rowj
    # Each term is 2x1 x 1x3, rank 6. Total: 18. Worse.
    #
    # Let me try: use a rank-7 scheme for 2x2x2 and find a rank-9 scheme
    # for the remaining 10 entries.
    #
    # The remaining 10 entries form a "cross" pattern. Let's see if there's
    # a rank-9 scheme for them.
    
    # Actually, I realize I should try a completely different approach:
    # Use the fact that we can have terms with support on multiple A entries,
    # multiple B entries, and multiple C entries, as long as the product
    # matches the tensor.
    
    # Let's try a rank-16 scheme based on a different algebraic identity.
    # Consider: for the 2x3 x 3x3 case, we can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Let's define:
    # X0 = A[0,0] + A[1,1]
    # X1 = A[0,1] + A[1,2]  
    # X2 = A[0,2] + A[1,0]
    # X3 = A[0,0]
    # X4 = A[0,1]
    # X5 = A[0,2]
    # X6 = A[1,0]
    # X7 = A[1,1]
    # X8 = A[1,2]
    #
    # This is getting complicated. Let me just try the systematic search
    # with a wider set of coefficient patterns.
    
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    direct_terms = [
        (0, 2, 4), (1, 5, 4), (3, 2, 5), (4, 5, 5),
        (2, 6, 0), (2, 7, 2), (2, 8, 4),
        (5, 6, 1), (5, 7, 3), (5, 8, 5),
    ]
    
    # Try all pairs with a wider set of coefficient patterns
    # including patterns where one coefficient is 0 (reducing support)
    coeff_vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for i in range(10):
        for j in range(i+1, 10):
            if time.time() > deadline:
                return None
            
            a1, b1, c1 = direct_terms[i]
            a2, b2, c2 = direct_terms[j]
            
            # Try patterns where the support might be smaller
            # Pattern 1: U supports only a1, V supports b1,b2, W supports c1,c2
            # This gives: (a1,b1,c1) and (a1,b2,c2) - need (a1,b2,c2) to not be a tensor entry
            # or to be canceled.
            
            # For simplicity, let's just try the basic pattern and a few variants
            patterns = [
                # (u1,u2), (v1,v2), (w1,w2)
                ((1,1), (1,1), (1,1)),
                ((1,1), (1,1), (2,0.5)),
                ((1,1), (2,0.5), (1,1)),
                ((2,0.5), (1,1), (1,1)),
                ((1,1), (1,1), (1,-1)),
                ((1,1), (1,-1), (1,1)),
                ((1,-1), (1,1), (1,1)),
            ]
            
            for (u1,u2), (v1,v2), (w1,w2) in patterns:
                U = np.zeros((n_a, 16))
                V = np.zeros((n_b, 16))
                W = np.zeros((n_c, 16))
                
                U[:, :7] = strassen_U[:, :7]
                V[:, :7] = strassen_V[:, :7]
                W[:, :7] = strassen_W[:, :7]
                
                U[a1, 7] = u1; U[a2, 7] = u2
                V[b1, 7] = v1; V[b2, 7] = v2
                W[c1, 7] = w1; W[c2, 7] = w2
                
                used = {i, j}
                remaining = [direct_terms[k] for k in range(10) if k not in used]
                for k, (a, b, c) in enumerate(remaining):
                    col = 8 + k
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                
                if _verify_decomposition(U, V, W, T):
                    return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank-16 with systematic pair search
    result = _try_rank16_systematic(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with two combined terms
    result = _try_rank16_two_combined(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with generalized Strassen
    result = _try_rank16_generalized_strassen(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with cancellation
    result = _try_rank16_with_cancellation(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-17
    U, V, W = _decompose_strassen(n, m, p)
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Schoolbook fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
