import time
import random
import sys
from itertools import product

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    best_score = float('inf')
    best_poly = None

    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            # Convert dict to list
            if not poly_dict:
                return
            max_exp = max(poly_dict.keys())
            min_exp = min(poly_dict.keys())
            # Ensure length <= 41 and valid range
            if max_exp >= 41:
                return
            
            # We want ascending degree list starting from 0? 
            # The task says "coefficient list in ascending degree".
            # Usually this means index i corresponds to degree i.
            # If min_exp > 0, we need to pad with zeros at the start.
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                if 0 <= e <= 40:
                    lst[e] = c
            
            # Validate
            if len(lst) > 41:
                return
            if lst[-1] == 0:
                # Trim trailing zeros if any? No, "nonzero final coefficient"
                # If the highest degree term is 0, we should trim.
                while lst and lst[-1] == 0:
                    lst.pop()
                if not lst:
                    return
                if len(lst) > 41:
                    return
            
            # Check exactly 12 nonzero terms
            nonzero_count = sum(1 for c in lst if c != 0)
            if nonzero_count != 12:
                return
                
            # Check bounds
            for c in lst:
                if c < -64 or c > 64:
                    return
            
            best_poly = lst

    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    def dict_to_list_check(d):
        """Helper to validate a dict before updating best."""
        max_exp = max(d.keys())
        if max_exp >= 41:
            return False
        for c in d.values():
            if c < -64 or c > 64:
                return False
        return True

    # ====================================================================
    # STRATEGY: Factor-Based Construction
    # P(x) = A(x) B(x) C(x)
    # A: 2 terms, B: 3 terms, C: 2 terms
    # This structure allows P to have 2*3*2 = 12 terms if exponents don't collide.
    # We search for small integer coefficients for A, B, C such that
    # the exponents of A, B, C are chosen to maximize collisions in P^2.
    #
    # Key Insight:
    # If A(x) = a0 + a1 x^u, B(x) = b0 + b1 x^v + b2 x^w, C(x) = c0 + c1 x^z
    # The degrees in P are sums of one degree from each factor.
    # The square P^2 involves sums of pairs of these degrees.
    # By choosing u, v, w, z to be small and related (e.g., arithmetic progression),
    # we can force many P^2 coefficients to cancel or merge.
    # ====================================================================

    # Coefficients for factors. Small integers are key for cancellation.
    # Since P coeffs are products, they can grow.
    # If A, B, C coeffs are in {-1, 1}, P coeffs are in {-1, 1}.
    # If we allow {-2, -1, 1, 2}, P coeffs can be up to 8.
    # Let's stick to {-1, 1} for factors first to keep P coeffs bounded and simple.
    # Actually, if A, B, C have coeffs in {-1, 1}, P has coeffs in {-1, 1}.
    # This restricts us to sign patterns.
    # To get more flexibility, let's allow {-2, -1, 1, 2} for factors,
    # but check P coeff bounds.
    
    factor_coeff_choices = [-1, 1] # Start with simple signs
    
    # Exponent ranges for factors
    # A: 2 terms, gap u
    # B: 3 terms, gaps v, w (relative to first term)
    # C: 2 terms, gap z
    
    # Let's iterate over possible "shapes" of exponents.
    # We can shift the whole polynomial, so we fix A's first term at 0.
    # A: {0, u}
    # B: {0, v, w} with 0 < v < w
    # C: {0, z}
    
    # The resulting degrees in P are:
    # 0 + 0 + 0 = 0
    # 0 + 0 + z = z
    # 0 + v + 0 = v
    # 0 + v + z = v + z
    # 0 + w + 0 = w
    # 0 + w + z = w + z
    # u + 0 + 0 = u
    # u + 0 + z = u + z
    # u + v + 0 = u + v
    # u + v + z = u + v + z
    # u + w + 0 = u + w
    # u + w + z = u + w + z
    
    # We need these 12 degrees to be distinct to have 12 terms?
    # If they collide, the number of terms in P is < 12.
    # The task requires EXACTLY 12 nonzero terms in P.
    # So we must ensure the 12 sums are distinct.
    
    # However, even if P has 12 terms, P^2 might have fewer nonzero coefficients
    # due to cancellation if coefficients are signed appropriately.
    
    # Let's search for u, v, w, z such that the 12 degrees are distinct.
    # And then assign signs (coeffs +/- 1) to A, B, C to minimize nonzero terms in P^2.
    
    max_gap = 10
    
    # Precompute valid (u, v, w, z) tuples where degrees are distinct
    valid_shapes = []
    for u in range(1, max_gap + 1):
        for v in range(1, max_gap + 1):
            for w in range(v + 1, max_gap + 1):
                for z in range(1, max_gap + 1):
                    degrees = [
                        0, z, v, v+z, w, w+z,
                        u, u+z, u+v, u+v+z, u+w, u+w+z
                    ]
                    if len(set(degrees)) == 12:
                        valid_shapes.append((u, v, w, z))
    
    # If no valid shapes found, expand range
    if not valid_shapes:
        for u in range(1, 20):
            for v in range(1, 20):
                for w in range(v + 1, 20):
                    for z in range(1, 20):
                        degrees = [
                            0, z, v, v+z, w, w+z,
                            u, u+z, u+v, u+v+z, u+w, u+w+z
                        ]
                        if len(set(degrees)) == 12:
                            valid_shapes.append((u, v, w, z))
    
    # Shuffle to randomize search order
    random.shuffle(valid_shapes)
    
    # For each shape, we have 2^2 * 2^3 * 2^2 = 16 * 8 * 4 = 512 sign combinations.
    # We can iterate through all of them quickly.
    
    for shape in valid_shapes:
        if time.time() > deadline:
            break
            
        u, v, w, z = shape
        
        # Degrees list
        degs = [
            0, z, v, v+z, w, w+z,
            u, u+z, u+v, u+v+z, u+w, u+w+z
        ]
        
        # Map degree to index in degs
        # Note: degs might not be sorted.
        
        # Iterate over sign patterns for A, B, C
        # A: a0, a1
        # B: b0, b1, b2
        # C: c0, c1
        
        # To save time, we can fix a0=1, b0=1, c0=1 (global sign and factor signs don't change P^2 nonzero count)
        # Actually, flipping sign of a whole factor flips sign of P, which doesn't change P^2.
        # So we can fix a0=1.
        # We can fix b0=1.
        # We can fix c0=1.
        
        # Remaining signs: a1, b1, b2, c1
        for a1 in [-1, 1]:
            for b1 in [-1, 1]:
                for b2 in [-1, 1]:
                    for c1 in [-1, 1]:
                        # Construct P coefficients
                        # P(e) = sum of a_i b_j c_k where deg_i + deg_j + deg_k = e
                        
                        # Let's build the dict
                        p_dict = {}
                        
                        # A terms: (0, 1), (u, a1)
                        # B terms: (0, 1), (v, b1), (w, b2)
                        # C terms: (0, 1), (z, c1)
                        
                        a_terms = [(0, 1), (u, a1)]
                        b_terms = [(0, 1), (v, b1), (w, b2)]
                        c_terms = [(0, 1), (z, c1)]
                        
                        for ea, ca in a_terms:
                            for eb, cb in b_terms:
                                for ec, cc in c_terms:
                                    e = ea + eb + ec
                                    c = ca * cb * cc
                                    p_dict[e] = p_dict.get(e, 0) + c
                        
                        # Check if exactly 12 nonzero terms
                        if len(p_dict) != 12:
                            continue
                            
                        # Check bounds
                        valid = True
                        for c in p_dict.values():
                            if c < -64 or c > 64:
                                valid = False
                                break
                        if not valid:
                            continue
                            
                        # Calculate score
                        score = get_score(p_dict)
                        
                        if score < best_score:
                            update_best(p_dict, score)
                            if best_score <= 12: # Can't do better than 12? 
                                # Actually, min nonzero terms in square of 12-term poly?
                                # If P is monomial, 1 term. But P has 12 terms.
                                # Theoretically, could be low.
                                # Let's not break early unless we hit a theoretical lower bound.
                                # For now, keep searching.
                                pass

    # Fallback: If no factor-based solution found, try a simple random search
    if best_poly is None:
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            d = {i: random.choice([-1, 1]) for i in range(12)}
            score = get_score(d)
            update_best(d, score)
            
    # Fallback 2: Hardcode a known good solution if search fails
    if best_poly is None:
        # Example: 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^10 - x^11
        # This is (1+x)(1-x^2)(1+x^4)(1-x^6)? No.
        # Let's just use the alternating pattern.
        best_poly = [1 if i % 4 < 2 else -1 for i in range(12)]
        # Validate
        d = {i: best_poly[i] for i in range(12)}
        update_best(d, get_score(d))

    # Final validation
    if best_poly is None:
        return [1] * 12
        
    return best_poly
