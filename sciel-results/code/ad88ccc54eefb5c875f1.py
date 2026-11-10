import random
from itertools import combinations, product

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k, (i, ci) in enumerate(nz):
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    # Step 1: Find good 6-term building blocks
    # We search for 6-term polynomials with coefficients in {1, -1}
    # that have a small number of non-zero terms in their square.
    
    best_blocks = []
    
    # Search exponents in a small range to find patterns
    # We can fix the lowest exponent to 0 by shifting.
    # So we choose 5 other exponents from 1..15.
    exp_range = range(1, 16)
    exps_combinations = list(combinations(exp_range, 5))
    
    # Sign patterns: 2^6 = 64. Fix first sign to 1 to reduce symmetry?
    # Actually, P and -P have same square. So fix first coeff to 1.
    # But the "first" term in the list is the lowest exponent.
    # So we fix coeff at exponent 0 to be 1? No, we shift to make min exp 0.
    # So yes, we can assume the term at the lowest exponent has coeff 1.
    
    signs = list(product([1, -1], repeat=5)) # 5 remaining signs
    
    for exps in exps_combinations:
        full_exps = (0,) + exps
        for s in signs:
            coeffs_dict = {0: 1}
            for i, e in enumerate(exps):
                coeffs_dict[e] = s[i]
                
            # Construct list
            max_exp = max(full_exps)
            p_list = [0] * (max_exp + 1)
            for e, c in coeffs_dict.items():
                p_list[e] = c
                
            sq = poly_square(p_list)
            nz_sq = count_nonzero(sq)
            
            # Store block info
            best_blocks.append((nz_sq, p_list, full_exps))
            
    # Sort by nz_sq
    best_blocks.sort(key=lambda x: x[0])
    
    # Keep top 20 unique blocks (by exponents pattern)
    seen_patterns = set()
    unique_blocks = []
    for nz_sq, p_list, exps in best_blocks:
        pattern = tuple(exps)
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        unique_blocks.append((nz_sq, p_list, exps))
        if len(unique_blocks) >= 20:
            break
    
    # Step 2: Combine two blocks
    # P = A + x^shift B
    # We need to ensure the squares don't overlap.
    # deg(A^2) = 2 * max(expA)
    # deg(B^2) = 2 * max(expB)
    # If we shift B by S, deg((x^S B)^2) = 2S + 2 max(expB)
    # We need 2 * max(expA) < 2S + min(expB) ? No.
    # Support of A^2 is [0, 2*maxA].
    # Support of (x^S B)^2 is [2S+2*minB, 2S+2*maxB].
    # We need 2*maxA < 2S + 2*minB.
    # Since minB is usually 0 (after shifting B's local min to 0?), let's normalize B.
    # In our generation, min exp is 0.
    # So we need 2*maxA < 2S. => S > maxA.
    # Also, we need the cross term A * (x^S B) to not interfere?
    # The task asks for the square of the polynomial P.
    # P^2 = A^2 + 2 x^S A B + x^{2S} B^2.
    # We assumed disjoint supports of A^2 and B^2.
    # But the cross term 2 x^S A B might overlap with A^2 or B^2 or be new.
    # To minimize non-zeros, we want the cross term to cancel or be zero?
    # It's hard to make cross term zero unless A or B is zero.
    # So we must account for the cross term.
    # This decomposition is only valid if we ignore cross terms, which is wrong.
    
    # Alternative: Just search for 12-term polynomials directly using the insight from blocks?
    # Or, use the local search but with a better initialization and move set.
    
    # Let's try a hybrid: 
    # 1. Use the found good 6-term blocks as seeds for a 12-term search?
    # No, let's stick to the directive: "improve it within that approach" OR "design a new program".
    # The directive says: "RESEARCH PROGRAMME: design a new program for this task from an approach family that none of the current programs use, from scratch; do not reuse the current designs."
    
    # So I should NOT use the local search. I should use the decomposition or a direct exhaustive search on a reduced space.
    
    # Let's try a direct search for 12-term polynomials with coeffs in {1, -1} and exponents in [0, 20].
    # This space is too large.
    
    # Let's refine the "Combine two blocks" idea.
    # If we choose A and B such that the cross term 2AB has very few non-zeros?
    # Or if we choose A and B to be "orthogonal" in some sense?
    
    # Actually, there is a known result:
    # If P(x) = x^{11} + x^{10} + ... + 1, nz(P^2)=21.
    # If P(x) = x^{11} - x^{10} + ... - x + 1, nz(P^2)=21.
    
    # What if we use coefficients other than 1, -1?
    # The problem allows [-64, 64].
    
    # Let's try a random search with a better fitness function and more iterations, but structured.
    # Wait, the directive says "do not reuse the current designs".
    # The current design is "Random Local Search".
    # I will implement "Exhaustive Search over Structured Candidates".
    
    # Candidate 1: Symmetric Polynomials
    # Candidate 2: Polynomials derived from cyclotomic factors
    
    # Let's try generating candidates from $P(x) = \prod_{i=1}^k (1 + x^{a_i})$ and then truncating/adjusting?
    # No, must be exactly 12 terms.
    
    # Let's try a simple exhaustive search over "gap" structures.
    # P(x) = A(x) + x^M B(x)
    # We want to minimize nz(P^2).
    # We will search for A (6 terms) and B (6 terms) from a precomputed list of "good" 6-term polys.
    # We will calculate the FULL square of A + x^M B for various M.
    
    # Re-evaluating the "good" 6-term polys.
    # We already computed `unique_blocks`.
    
    best_score = float('inf')
    best_cand = None
    
    # We will test combinations of the top blocks
    # To keep time low, we only use the top 5 blocks.
    top_blocks = unique_blocks[:5]
    
    for i, (nz_a, pa, ea) in enumerate(top_blocks):
        for j, (nz_b, pb, eb) in enumerate(top_blocks):
            # Try several shifts M
            max_a = max(ea)
            max_b = max(eb)
            # We need M large enough to separate A^2 and B^2 roughly
            # But we must compute the actual square.
            for M in range(max_a + 1, max_a + max_b + 5):
                # Construct P
                # A is pa (len = max_a+1)
                # B is pb (len = max_b+1), shifted by M
                len_p = max(len(pa), M + len(pb))
                if len_p > 41:
                    continue
                
                p = [0] * len_p
                for k, c in enumerate(pa):
                    p[k] += c
                for k, c in enumerate(pb):
                    p[M+k] += c
                    
                # Check validity
                if not valid_candidate(p):
                    continue
                    
                sq = poly_square(p)
                score = count_nonzero(sq)
                
                if score < best_score:
                    best_score = score
                    best_cand = p[:]
                    
    # If no candidate found (unlikely), fallback to a known good one
    if best_cand is None:
        # Fallback: A simple 12-term poly
        best_cand = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 6 terms? No, need 12.
        best_cand = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
    return best_cand
