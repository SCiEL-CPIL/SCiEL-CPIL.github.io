import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21
    
    best_score = fallback_score
    best_candidate = fallback

    def compute_score(exps, coeffs):
        """
        Computes the number of non-zero terms in P(x)^2.
        exps: list of 12 exponents
        coeffs: list of 12 coefficients
        """
        if not exps or not coeffs:
            return float('inf')
        
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
            
        q = np.convolve(p, p)
        count = np.count_nonzero(q)
        return count

    def _update_best(exps, coeffs):
        """Update the global best candidate dense list."""
        nonlocal best_score, best_candidate
        score = compute_score(exps, coeffs)
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    # Strategy: Model P(x) = A(x) * B(x) * C(x) where A, B, C are 4-term polynomials.
    # The exponents of P are sums a_i + b_j + c_k.
    # We fix the exponent supports (a_i, b_j, c_k) such that all 12 sums are distinct and <= 40.
    # Then we optimize the coefficients a_i, b_j, c_k.
    # The coefficient of x^{a_i+b_j+c_k} in P is a_i b_j c_k.
    # We want to minimize non-zero terms in P^2.
    
    # To allow fast search, we will precompute a list of valid exponent triples (A, B, C).
    # We can fix A = (0, 1, 2, 3) to reduce symmetry? No, let's search a bit more broadly
    # but constrained.
    
    # Actually, let's just search for valid (A, B, C) on the fly or precompute a small set.
    # Given the tight time limit, precomputing a small set of diverse bases is safer.
    
    valid_bases = []
    
    # We want 12 distinct sums from sets of size 4.
    # Let's try to find bases where the sums are "spread out" to reduce collisions in P^2.
    # However, we also want the sums to be "clustered" in a way that allows coefficient cancellation?
    # Actually, if the exponents of P are far apart, P^2 exponents are sums of pairs.
    # If exponents are e_1 < e_2 < ... < e_12.
    # P^2 has terms x^{e_i + e_j}.
    # If e_i + e_j = e_k + e_l for (i,j) != (k,l), then cancellation is possible.
    # To maximize cancellation opportunities, we want many collisions in the sumset.
    # This suggests that arithmetic progressions or structured sets are good.
    
    # Let's try to find bases where the 12 exponents form a structure with many additive relations.
    
    # Search for bases A, B, C (each size 4) such that sums are distinct.
    # We'll iterate over small ranges.
    
    # To speed up, we fix A = (0, 1, 2, 3) and search B, C.
    # This might miss optimal structures where A is not an AP, but it's a good heuristic.
    # Actually, let's allow A to vary slightly.
    
    # Let's just hardcode a few good structures found by manual inspection or quick search.
    # Structure 1: A=(0,1,2,3), B=(0,4,8,12), C=(0,16,32,48) -> Max 3+12+48=63 (Too big)
    # Structure 2: A=(0,1,2,3), B=(0,4,8,12), C=(0,13,26,39) -> Max 3+12+39=54 (Too big)
    # Structure 3: A=(0,1,2,3), B=(0,4,8,12), C=(0,10,20,30) -> Max 3+12+30=45 (Too big)
    # Structure 4: A=(0,1,2,3), B=(0,4,8,12), C=(0,9,18,27) -> Max 3+12+27=42 (Too big)
    # Structure 5: A=(0,1,2,3), B=(0,4,8,12), C=(0,8,16,24) -> Max 3+12+24=39 (OK)
    #    Sums:
    #    0+0+0=0, 0+0+8=8, 0+0+16=16, 0+0+24=24
    #    0+4+0=4, 0+4+8=12, 0+4+16=20, 0+4+24=28
    #    0+8+0=8 (Collision with 0+0+8) -> Invalid
    
    # Let's write a quick search for valid bases within the time limit.
    
    def find_valid_bases(limit=50):
        bases = []
        # Fix A to reduce search space?
        # Let's try a few fixed A's.
        A_candidates = [
            (0, 1, 2, 3),
            (0, 1, 2, 4),
            (0, 1, 3, 4),
            (0, 1, 2, 5),
            (0, 2, 4, 6),
            (0, 1, 3, 5)
        ]
        
        for A in A_candidates:
            max_a = max(A)
            # B and C must be such that max_b + max_c <= 40 - max_a
            # We'll iterate B and C from small sets.
            
            # To avoid duplicates and speed up, we can fix B_0 = 0, C_0 = 0.
            # And sort B and C.
            
            # Iterate B
            # b1, b2, b3
            for b1 in range(1, 12):
                for b2 in range(b1+1, 20):
                    for b3 in range(b2+1, 40):
                        B = (0, b1, b2, b3)
                        max_b = b3
                        if max_a + max_b > 40: continue
                        
                        # Iterate C
                        max_c_allowed = 40 - max_a - max_b
                        if max_c_allowed < 3: continue
                        
                        for c1 in range(1, 12):
                            for c2 in range(c1+1, 20):
                                for c3 in range(c2+1, max_c_allowed+1):
                                    C = (0, c1, c2, c3)
                                    
                                    # Check distinctness of sums
                                    sums = set()
                                    valid = True
                                    for ai in A:
                                        for bj in B:
                                            for ck in C:
                                                s = ai + bj + ck
                                                if s in sums:
                                                    valid = False
                                                    break
                                                sums.add(s)
                                            if not valid: break
                                        if not valid: break
                                    if valid:
                                        bases.append((A, B, C))
                                        if len(bases) >= limit:
                                            return bases
        return bases

    # This search might be slow if not careful.
    # Let's try to be smarter.
    # We want the 12 sums to have many additive relations.
    # A good candidate is when the sums form an arithmetic progression or close to it.
    
    # Let's try to construct a specific good candidate manually first to ensure we have a strong baseline.
    # Consider P(x) = (1 - x + x^2 - x^3)(1 + x^4 + x^8 + x^12)(1 + x^16 + x^20 + x^24) ... wait, degrees.
    
    # Let's just run the search but with a strict time check inside.
    
    bases_found = []
    A_candidates = [
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 3, 4)
    ]
    
    search_start = time.time()
    
    for A in A_candidates:
        if time.time() > search_start + 5: break
        
        max_a = max(A)
        
        # We'll sample B and C to keep it fast.
        # Instead of full nested loops, let's try some structured ones.
        
        # Try B = (0, b, 2b, 3b) and C = (0, c, 2c, 3c)
        for b in range(1, 10):
            for c in range(1, 10):
                B = (0, b, 2*b, 3*b)
                C = (0, c, 2*c, 3*c)
                if max_a + 3*b + 3*c > 40: continue
                
                sums = set()
                valid = True
                for ai in A:
                    for bj in B:
                        for ck in C:
                            s = ai + bj + ck
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid: break
                    if not valid: break
                if valid:
                    bases_found.append((A, B, C))
        
        if len(bases_found) > 10: break

    # If we didn't find enough, do a random search
    if len(bases_found) < 5:
        import random
        random.seed(42)
        for _ in range(2000):
            if time.time() > search_start + 5: break
            A = tuple(sorted(random.sample(range(0, 10), 4)))
            B = tuple(sorted(random.sample(range(0, 20), 4)))
            C = tuple(sorted(random.sample(range(0, 30), 4)))
            if max(A) + max(B) + max(C) > 40: continue
            
            sums = set()
            valid = True
            for ai in A:
                for bj in B:
                    for ck in C:
                        s = ai + bj + ck
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid: break
                if not valid: break
            if valid:
                bases_found.append((A, B, C))
                if len(bases_found) >= 20: break

    # If still none, use the fallback exponents (0..11)
    if not bases_found:
        # Just use the fallback polynomial directly
        # We can't easily decompose 0..11 into 3 sets of 4 with distinct sums easily?
        # Actually, we can just optimize the coefficients of the fallback polynomial.
        exps = list(range(12))
        coeffs = [1] * 12
        _update_best(exps, coeffs)
        
        # Local search on coeffs
        current_score = compute_score(exps, coeffs)
        COEFF_VALUES = [-3, -2, -1, 1, 2, 3]
        
        no_improve = 0
        while no_improve < 20 and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for new_c in COEFF_VALUES:
                    if new_c == coeffs[i]:
                        continue
                    old_c = coeffs[i]
                    coeffs[i] = new_c
                    new_score = compute_score(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        _update_best(exps, coeffs)
                    else:
                        coeffs[i] = old_c
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return best_candidate

    # Now we have a list of valid bases.
    # For each base, we optimize the 12 coefficients (which are products a_i b_j c_k).
    # Actually, we optimize the 12 base coefficients a_i, b_j, c_k.
    # There are 12 variables.
    # The coefficient of P at exponent e_{ijk} is a_i b_j c_k.
    
    # We will perform a local search on the 12 variables (a_0..a_3, b_0..b_3, c_0..c_3).
    # Initial values: all 1s.
    
    for A, B, C in bases_found:
        if time.time() > deadline - 10:
            break
            
        exps = []
        # Map (i,j,k) to exponent
        exp_map = {}
        for i, ai in enumerate(A):
            for j, bj in enumerate(B):
                for k, ck in enumerate(C):
                    e = ai + bj + ck
                    exps.append(e)
                    exp_map[(i,j,k)] = e
        
        # Sort exponents to have a consistent order for the dense array?
        # compute_score handles unsorted exps as long as they are distinct.
        
        # Variables: a (4), b (4), c (4)
        vars_ = [1] * 12
        a = vars_[0:4]
        b = vars_[4:8]
        c = vars_[8:12]
        
        def get_coeffs_from_vars(v):
            a_v = v[0:4]
            b_v = v[4:8]
            c_v = v[8:12]
            coeffs = []
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        coeffs.append(a_v[i] * b_v[j] * c_v[k])
            return coeffs

        current_coeffs = get_coeffs_from_vars(vars_)
        current_score = compute_score(exps, current_coeffs)
        _update_best(exps, current_coeffs)
        
        # Local search
        # We can change one variable at a time.
        # Values to try: -3, -2, -1, 1, 2, 3
        
        no_improve = 0
        while no_improve < 30 and time.time() < deadline - 5:
            improved = False
            for idx in range(12):
                for new_val in [-3, -2, -1, 1, 2, 3]:
                    if new_val == vars_[idx]:
                        continue
                    old_val = vars_[idx]
                    vars_[idx] = new_val
                    new_coeffs = get_coeffs_from_vars(vars_)
                    new_score = compute_score(exps, new_coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        _update_best(exps, new_coeffs)
                    else:
                        vars_[idx] = old_val
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    return best_candidate

# EVOLVE-BLOCK-END
