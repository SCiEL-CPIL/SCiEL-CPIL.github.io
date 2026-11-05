import time
import itertools
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit

    def count_square_terms(coeffs_dict):
        """Count nonzero terms in square given sparse dict {exp: coeff}."""
        if not coeffs_dict:
            return 0
        square = {}
        items = list(coeffs_dict.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                key = ei + items[j][0]
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * items[j][1]
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    def evaluate_from_dict(coeffs_dict):
        """Validate and score a sparse polynomial representation."""
        if not coeffs_dict:
            return None
        max_exp = max(coeffs_dict.keys())
        if max_exp > 40:
            return None
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in coeffs_dict.items():
            coeffs_list[e] = c
        while len(coeffs_list) > 1 and coeffs_list[-1] == 0:
            coeffs_list.pop()
        if len(coeffs_list) > 41:
            return None
        if coeffs_list[-1] == 0:
            return None
        nonzero_count = sum(1 for c in coeffs_list if c != 0)
        if nonzero_count != 12:
            return None
        for c in coeffs_list:
            if not (-64 <= c <= 64):
                return None
        score = count_square_terms(coeffs_dict)
        return score

    def dict_to_list(d):
        """Convert sparse dict to dense list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst

    def poly_mul(dict_a, dict_b):
        """Multiply two sparse polynomials (dicts)."""
        result = {}
        for ea, ca in dict_a.items():
            for eb, cb in dict_b.items():
                key = ea + eb
                result[key] = result.get(key, 0) + ca * cb
        result = {k: v for k, v in result.items() if v != 0}
        return result

    best_poly = None
    best_score = float('inf')

    def update_best(d):
        nonlocal best_poly, best_score
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)
            return True
        return False

    # ====================================================================
    # NEW STRATEGY: Arithmetic progression exponents with step k,
    # coefficients c_i = (-1)^i * a (alternating with amplitude a).
    # 
    # P(x) = sum_{i=0}^{11} (-1)^i * a * x^{i*k}
    # 
    # P^2 = a^2 * sum_{i,j} (-1)^{i+j} x^{(i+j)*k}
    # 
    # The exponent (i+j)*k is always a multiple of k.
    # The coefficient at exponent t*k (where t = i+j) is:
    #   a^2 * sum_{i+j=t} (-1)^{i+j} = a^2 * sum_{i+j=t} 1 = a^2 * (number of pairs)
    # 
    # Wait, (-1)^{i+j} is NOT always 1. Let me recompute.
    # (-1)^{i+j} depends on parity of i+j.
    # 
    # For fixed t = i+j, the number of pairs (i,j) with i+j=t is:
    #   min(t+1, 23-t, 12) for t in 0..22
    # 
    # But the coefficient is sum_{i=0}^{min(t,11)} (-1)^{i+(t-i)} = sum_{i} (-1)^t = (-1)^t * (count)
    # 
    # So coefficient at t*k is a^2 * (-1)^t * N(t) where N(t) = number of (i,j) with i+j=t.
    # 
    # N(t) for t=0..22: 1,2,3,...,11,12,11,10,...,2,1
    # 
    # So P^2 has support at {0, k, 2k, ..., 22k} with coefficients a^2*(-1)^t*N(t).
    # All these are nonzero (N(t) >= 1, a != 0), so P^2 has exactly 23 terms.
    # 
    # That's not great. Let me think differently.
    # 
    # What if we use c_i = (-1)^{floor(i/2)} or some other pattern?
    # 
    # Actually, the key insight from the directive: if exponents form AP with step k
    # and coefficients satisfy c_i = (-1)^i * k, the square's support shrinks to every 2k-th degree.
    # 
    # Let me verify: P(x) = sum_{i=0}^{11} (-1)^i * k * x^{i*k}
    # P^2 = k^2 * sum_{i,j} (-1)^{i+j} x^{(i+j)*k}
    # 
    # The term at exponent (i+j)*k has coefficient k^2 * (-1)^{i+j} * (number of such pairs).
    # This is k^2 * (-1)^t * N(t) for t = i+j.
    # 
    # These are all nonzero, so we get 23 terms. Not 2k-th degrees only.
    # 
    # Hmm, maybe the idea is different. Let me try:
    # P(x) = sum_{i=0}^{5} x^{i*2k} - sum_{i=0}^{5} x^{(i+1)*2k} ... no.
    # 
    # Let me try a completely different approach: use the structure where
    # P(x) = Q(x^k) for some 12-term Q, and Q^2 has few terms.
    # 
    # If Q has 12 terms and Q^2 has m terms, then P^2 has m terms (at exponents that are multiples of k).
    # 
    # So the problem reduces to: find a 12-term polynomial Q with minimal square terms,
    # then embed it with step k to control degree.
    # 
    # For Q with consecutive exponents 0..11, we already searched.
    # 
    # NEW IDEA: Use Q(x) = (1+x+x^2+...+x^5) - x^6*(1+x+x^2+...+x^5) = 1+x+...+x^5 - x^6-...-x^11
    # This is the alternating block pattern. Let's compute its square.
    # 
    # Even better: try Q(x) where the 12 coefficients are chosen to make Q^2 sparse.
    # Use a SAT-like approach: for each possible square term position, decide if it's zero or not.
    # 
    # MOST PROMISING: Exhaustive search over coefficient patterns for consecutive exponents,
    # but with a much larger coefficient range and smarter initialization.
    # ====================================================================

    # First: try the arithmetic progression with alternating coefficients
    for k in range(1, 4):
        for a in range(1, 7):
            d = {}
            for i in range(12):
                e = i * k
                if e <= 40:
                    d[e] = d.get(e, 0) + a * (1 if i % 2 == 0 else -1)
            if len(d) == 12:
                update_best(d)

    # Try: P(x) = sum_{i=0}^{5} x^{i*k} - sum_{i=0}^{5} x^{(i+6)*k} (block difference)
    for k in range(1, 4):
        d = {}
        for i in range(6):
            d[i * k] = d.get(i * k, 0) + 1
        for i in range(6):
            d[(i + 6) * k] = d.get((i + 6) * k, 0) - 1
        if len(d) == 12:
            update_best(d)

    # Try: P(x) = sum_{i=0}^{11} c_i * x^{i*k} with c_i from a structured pattern
    # Pattern: c_i = 1 for i in {0,1,2,3,4,5}, c_i = -1 for i in {6,7,8,9,10,11}
    # Already done above.
    
    # Pattern: c_i = (-1)^{i//3} (blocks of 3)
    for k in range(1, 4):
        for amp in [1, 2, 3]:
            d = {}
            for i in range(12):
                e = i * k
                if e <= 40:
                    sign = 1 if (i // 3) % 2 == 0 else -1
                    d[e] = d.get(e, 0) + amp * sign
            if len(d) == 12:
                update_best(d)

    # ====================================================================
    # CORE: Systematic search over exponent sets with structured coefficients
    # 
    # Key insight: The square P^2 = sum_{i,j} c_i c_j x^{e_i + e_j}
    # has a term at exponent s only if there exists (i,j) with e_i + e_j = s.
    # The coefficient at s is sum_{i+j: e_i+e_j=s} c_i c_j.
    # 
    # To minimize nonzero terms in P^2, we want many cancellations:
    # sum_{i+j: e_i+e_j=s} c_i c_j = 0 for as many s as possible.
    # 
    # This is a system of quadratic equations in the c_i's.
    # 
    # APPROACH: Fix the exponent set, then solve for coefficients that
    # maximize cancellations. Use linear algebra: for each target exponent s,
    # the cancellation condition is a linear equation in the products c_i*c_j.
    # 
    # For small coefficient values (±1, ±2), we can enumerate.
    # ====================================================================

    # Generate candidate exponent sets
    def generate_exponent_sets():
        """Generate diverse 12-element exponent subsets of {0,...,40}."""
        sets = []
        
        # Consecutive
        for start in range(0, 30):
            exps = list(range(start, start + 12))
            if max(exps) <= 40:
                sets.append(frozenset(exps))
        
        # Arithmetic progressions
        for start in range(0, 10):
            for step in range(1, 5):
                exps = [start + i * step for i in range(12)]
                if max(exps) <= 40:
                    sets.append(frozenset(exps))
        
        # Two interleaved APs
        for s1 in range(1, 5):
            for s2 in range(1, 5):
                if s1 == s2:
                    continue
                exps = []
                for i in range(6):
                    exps.append(i * s1)
                    exps.append(1 + i * s2)
                exps = sorted(set(exps))
                if len(exps) == 12 and max(exps) <= 40:
                    sets.append(frozenset(exps))
        
        # Random sets (seeded for reproducibility)
        rng = random.Random(42)
        for _ in range(200):
            exps = rng.sample(range(0, 41), 12)
            sets.append(frozenset(exps))
        
        # Structured: pairs (i, i+k) for i=0..5
        for k in range(1, 10):
            exps = []
            for i in range(6):
                exps.append(i)
                exps.append(i + k)
            exps = sorted(set(exps))
            if len(exps) == 12 and max(exps) <= 40:
                sets.append(frozenset(exps))
        
        return list(set(sets))

    def try_coefficients(exponents, coeff_range):
        """Try all coefficient assignments from coeff_range for given exponents."""
        exps = sorted(exponents)
        best_local = None
        best_local_score = float('inf')
        
        n = len(exps)
        if n != 12:
            return None, float('inf')
        
        # If coefficient range is small enough, enumerate
        total_combos = len(coeff_range) ** n
        if total_combos > 100000:
            # Too many, use random sampling
            rng = random.Random(hash(frozenset(exps)))
            samples = min(total_combos, 5000)
            for _ in range(samples):
                coeffs = [rng.choice(coeff_range) for _ in range(n)]
                d = {exps[i]: coeffs[i] for i in range(n)}
                score = count_square_terms(d)
                if score < best_local_score:
                    best_local_score = score
                    best_local = d
        else:
            for combo in itertools.product(coeff_range, repeat=n):
                d = {exps[i]: combo[i] for i in range(n)}
                score = count_square_terms(d)
                if score < best_local_score:
                    best_local_score = score
                    best_local = d
        
        return best_local, best_local_score

    # Try small coefficient ranges on diverse exponent sets
    exponent_sets = generate_exponent_sets()
    
    # Sort by a heuristic: prefer sets with more "structure" (APs, etc.)
    def structure_score(exps):
        exps = sorted(exps)
        diffs = [exps[i+1] - exps[i] for i in range(11)]
        # Prefer uniform diffs
        if len(set(diffs)) == 1:
            return 0
        if len(set(diffs)) == 2:
            return 1
        return 2

    exponent_sets.sort(key=structure_score)

    coeff_ranges = [
        [-1, 1],
        [-2, -1, 1, 2],
        [-1, 0, 1],  # This won't give 12 nonzero terms, skip
    ]
    
    # Only use ranges that can give 12 nonzero terms
    coeff_ranges = [
        [-1, 1],
        [-2, -1, 1, 2],
        [-3, -2, -1, 1, 2, 3],
    ]

    for cr in coeff_ranges:
        if time.time() > deadline:
            break
        
        for exps in exponent_sets:
            if time.time() > deadline:
                break
            
            d, local_score = try_coefficients(exps, cr)
            if d is not None:
                # Validate
                if len(d) == 12:
                    all_nonzero = all(c != 0 for c in d.values())
                    in_range = all(-64 <= c <= 64 for c in d.values())
                    if all_nonzero and in_range:
                        update_best(d)

    # ====================================================================
    # TARGETED: Use the identity (1+x^k+x^{2k}+...+x^{11k})^2 structure
    # and modify coefficients to create cancellations.
    # 
    # Base: P0(x) = sum_{i=0}^{11} x^{i*k} has square with 23 terms.
    # Modify: P(x) = sum_{i=0}^{11} c_i x^{i*k}
    # 
    # The square has terms at exponents 0, k, 2k, ..., 22k.
    # Coefficient at t*k: sum_{i+j=t} c_i c_j for t=0..22.
    # 
    # We want many of these to be zero.
    # 
    # For k=1: 23 positions. For k=2: 23 positions (0,2,4,...,44) but max is 44>40.
    # So k must be such that 22k <= 40, i.e., k <= 1.
    # 
    # Wait, 22*1 = 22 <= 40. k=1 works. k=2: 22*2=44>40. So only k=1.
    # 
    # For k=1, we need c_0..c_11 such that sum_{i+j=t} c_i c_j = 0 for many t.
    # 
    # This is equivalent to finding a 12-term polynomial whose square has few terms.
    # Let's do a focused search.
    # ====================================================================

    # Focused search on consecutive exponents 0..11 with wider coefficient range
    random.seed(999)
    
    def fast_square_count(coeffs_list):
        """Fast square count for consecutive exponents 0..11."""
        n = len(coeffs_list)
        square = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs_list[i]
            if ci == 0:
                continue
            for j in range(i, n):
                cj = coeffs_list[j]
                if cj == 0:
                    continue
                key = i + j
                if i == j:
                    square[key] += ci * cj
                else:
                    square[key] += 2 * ci * cj
        return sum(1 for v in square if v != 0)

    # Simulated annealing on consecutive exponents
    for restart in range(500):
        if time.time() > deadline:
            break
        
        curr = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        curr_score = fast_square_count(curr)
        
        T = 10.0
        for step in range(200):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_val = curr[idx]
            new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val == old_val:
                continue
            curr[idx] = new_val
            new_score = fast_square_count(curr)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    d = {i: curr[i] for i in range(12) if curr[i] != 0}
                    if len(d) == 12:
                        best_poly = dict_to_list(d)
            else:
                # Accept with probability exp(-delta/T)
                import math
                if random.random() < math.exp(-(new_score - curr_score) / T):
                    curr_score = new_score
                else:
                    curr[idx] = old_val
            
            T *= 0.999  # Cool down

    # ====================================================================
    # ALTERNATIVE: Use the structure P(x) = A(x) + B(x) where
    # A and B have 6 terms each, and A^2, B^2, and AB have overlapping
    # supports to create cancellations.
    # 
    # If A = sum_{i=0}^{5} a_i x^{p_i} and B = sum_{i=0}^{5} b_i x^{q_i},
    # then P^2 = A^2 + 2AB + B^2.
    # 
    # Choose p_i and q_i so that the supports of A^2, AB, and B^2 overlap
    # significantly, allowing cancellations.
    # 
    # Simplest: p_i = i, q_i = i + offset (shifted copy).
    # Then A and B have overlapping exponents if offset is small.
    # But we need exactly 12 nonzero terms in P, so A and B must have
    # disjoint supports (otherwise some terms cancel).
    # 
    # Let A = sum_{i=0}^{5} a_i x^i, B = sum_{i=0}^{5} b_i x^{i+6}.
    # P = A + x^6 * B'.
    # P^2 = A^2 + 2 x^6 A B' + x^12 B'^2.
    # 
    # A^2 has terms at degrees 0..10.
    # x^6 A B' has terms at degrees 6..16.
    # x^12 B'^2 has terms at degrees 12..22.
    # 
    # Overlap regions: 6..10 (A^2 andxAB), 12..16 (xAB and B'^2).
    # 
    # To create cancellations, we need coefficients in the overlap regions
    # to sum to zero.
    # ====================================================================

    for a_coeffs in itertools.product([-2, -1, 1, 2], repeat=6):
        if time.time() > deadline:
            break
        for b_coeffs in itertools.product([-2, -1, 1, 2], repeat=6):
            if time.time() > deadline:
                break
            
            d = {}
            for i in range(6):
                d[i] = a_coeffs[i]
            for i in range(6):
                d[6 + i] = b_coeffs[i]
            
            if len(d) != 12:
                continue
            
            score = count_square_terms(d)
            if score < best_score:
                best_score = score
                best_poly = dict_to_list(d)

    # ====================================================================
    # FINAL: If we haven't found anything, use the best from any strategy
    # ====================================================================
    
    if best_poly is None:
        # Fallback: 12 consecutive terms with alternating signs
        best_poly = [1 if i % 2 == 0 else -1 for i in range(12)]
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    final_score = evaluate_from_dict(final_dict)
    if final_score is None:
        best_poly = [1] * 12
    
    return best_poly
