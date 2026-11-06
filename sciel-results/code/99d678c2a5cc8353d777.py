import time
import random
from itertools import combinations

DEADLINE = 90.0
SAFETY_MARGIN = 5.0

def _count_nonzero(coeffs):
    return sum(1 for v in coeffs.values() if v != 0)

def _eval_pair(a_exps, a_signs, b_exps, b_signs):
    """
    Evaluate the number of nonzero terms in (A+B)^2 given 6-term A and 6-term B.
    A^2 + B^2 + 2AB
    """
    # A^2
    a2 = {}
    for i in range(6):
        for j in range(i, 6):
            e = a_exps[i] + a_exps[j]
            if i == j:
                a2[e] = a2.get(e, 0) + 1
            else:
                a2[e] = a2.get(e, 0) + 2 * a_signs[i] * a_signs[j]
    
    # B^2
    b2 = {}
    for i in range(6):
        for j in range(i, 6):
            e = b_exps[i] + b_exps[j]
            if i == j:
                b2[e] = b2.get(e, 0) + 1
            else:
                b2[e] = b2.get(e, 0) + 2 * b_signs[i] * b_signs[j]
    
    # 2AB
    ab = {}
    for i in range(6):
        for j in range(6):
            e = a_exps[i] + b_exps[j]
            ab[e] = ab.get(e, 0) + 2 * a_signs[i] * b_signs[j]
            
    # Sum
    total = {}
    for e, v in a2.items():
        total[e] = total.get(e, 0) + v
    for e, v in b2.items():
        total[e] = total.get(e, 0) + v
    for e, v in ab.items():
        total[e] = total.get(e, 0) + v
        
    return _count_nonzero(total)

def _optimize_6term(exps, deadline):
    """
    Find best signs for a 6-term polynomial to minimize its square's nonzero terms.
    Returns (count, signs).
    """
    best_count = 10**9
    best_signs = None
    # 2^6 = 64 possibilities
    for mask in range(64):
        signs = [1 if (mask >> i) & 1 == 0 else -1 for i in range(6)]
        # Quick check: does this square have few terms?
        sq = {}
        for i in range(6):
            for j in range(i, 6):
                e = exps[i] + exps[j]
                if i == j:
                    sq[e] = sq.get(e, 0) + 1
                else:
                    sq[e] = sq.get(e, 0) + 2 * signs[i] * signs[j]
        cnt = _count_nonzero(sq)
        if cnt < best_count:
            best_count = cnt
            best_signs = signs
            if cnt == 6: # Theoretical min for 6 terms? (1+x+...+x^5)^2 has 11. 
                # Can we get 6? 
                # (1-x+x^2-x^3+x^4-x^5)^2 ... 
                # Actually min is likely higher. 
                # But if we find it, we can't do better.
                pass
    return best_count, best_signs

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_coefs = None
    best_sq_nz = 10**9
    
    def update(a_exps, a_signs, b_exps, b_signs):
        nonlocal best_coefs, best_sq_nz
        # Combine
        all_exps = a_exps + b_exps
        all_signs = a_signs + b_signs
        
        # Check constraints
        if len(set(all_exps)) != 12:
            return
        if max(all_exps) >= 41:
            return
            
        coefs = [0] * (max(all_exps) + 1)
        for e, s in zip(all_exps, all_signs):
            coefs[e] += s # Should be 1 or -1 if disjoint
            
        # Verify exactly 12 nonzero terms
        if sum(1 for c in coefs if c != 0) != 12:
            return
        if coefs[-1] == 0:
            return
            
        # Evaluate square
        count = _eval_pair(a_exps, a_signs, b_exps, b_signs)
        
        if count < best_sq_nz:
            best_sq_nz = count
            best_coefs = coefs

    # Strategy:
    # 1. Generate candidate 6-term polynomials A and B.
    # 2. Ensure their exponent sets are disjoint.
    # 3. Optimize signs for A and B independently (or jointly if we have time).
    # 4. Evaluate (A+B)^2.
    
    # Pre-generate a pool of good 6-term polynomials
    # A "good" 6-term poly has a sparse square.
    # Let's search for 6-term polys with sparse squares first.
    
    candidate_6s = []
    
    # Generate random 6-term exponent sets
    for _ in range(5000):
        if time.time() > deadline - 30: # Leave time for pair evaluation
            break
        # Small range to encourage collisions in square
        base = rng.randint(0, 20)
        exps = set()
        while len(exps) < 6:
            e = base + rng.randint(0, 20)
            if e < 41:
                exps.add(e)
        exps = sorted(exps)
        
        # Optimize signs
        cnt, signs = _optimize_6term(exps, deadline)
        
        # Keep only "good" ones (sparse square)
        # Square of 6 terms has at least 6 terms. 
        # (1+x+...+x^5)^2 has 11 terms.
        # (1-x+x^2-x^3+x^4-x^5)^2:
        # 1 - 2x + 3x^2 - 4x^3 + 5x^4 - 6x^5 + 7x^6 - 6x^7 + 5x^8 - 4x^9 + 3x^10 - 2x^11 + x^12 (13 terms)
        # So 11 is better.
        if cnt <= 12:
            candidate_6s.append((exps, signs, cnt))
            
    # Sort by square sparsity
    candidate_6s.sort(key=lambda x: x[2])
    
    # Evaluate pairs
    # We want to pick A and B such that:
    # 1. Exponent sets are disjoint.
    # 2. Cross terms 2AB don't collide with A^2 or B^2.
    # 3. Total nonzero count is minimized.
    
    # To speed up, we can limit the number of pairs we check.
    # If we have K candidates, checking all pairs is O(K^2).
    # K might be large. Let's restrict to top 500 candidates.
    top_candidates = candidate_6s[:500]
    
    # Further filter: only keep candidates that are "compatible" (disjoint exponents)
    # This is a bipartite matching problem essentially, but we just want min cost pair.
    
    # Let's iterate over pairs
    for i in range(len(top_candidates)):
        if time.time() > deadline:
            break
        a_exps, a_signs, a_cnt = top_candidates[i]
        a_set = set(a_exps)
        
        for j in range(i + 1, len(top_candidates)):
            b_exps, b_signs, b_cnt = top_candidates[j]
            
            # Check disjointness
            if a_set.intersection(set(b_exps)):
                continue
                
            # Check max degree
            if max(max(a_exps), max(b_exps)) >= 41:
                continue
                
            # Evaluate
            update(a_exps, a_signs, b_exps, b_signs)
            
            # If we found a very good one, we might stop early?
            # Theoretical min for 12 terms?
            # If A^2 has 6 terms and B^2 has 6 terms and 2AB has 6 terms and no collisions, total is 18.
            # Can we get lower?
            # If A and B are such that A^2 and B^2 cancel each other? No, they are positive sums of squares? 
            # Wait, coefficients are integers.
            # A^2 terms are sums of products.
            
            # Let's just keep searching.
            
    # Fallback if no pair found or time ran out
    if best_coefs is None:
        # Construct a default valid polynomial
        # 12 terms, e.g. 1, x, ..., x^11
        exps = list(range(12))
        signs = [1] * 12
        coefs = [0] * 12
        for e, s in zip(exps, signs):
            coefs[e] = s
        best_coefs = coefs
        # Calculate its score for reporting (though we return coefs)
        # (1+...+x^11)^2 has 23 terms.
        
    return best_coefs

if __name__ == "__main__":
    res = construct()
    print(res)
