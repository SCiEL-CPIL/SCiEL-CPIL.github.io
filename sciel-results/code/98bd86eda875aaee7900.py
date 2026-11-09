import numpy as np
import time
import random
from itertools import combinations, product

def _sq_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the square of a polynomial given by coefficient list."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _valid(coeffs):
    """Check if coeffs is a valid candidate: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(abs(c) > 64 for c in coeffs):
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 90s limit
    
    best_score = float('inf')
    best_coeffs = None
    
    np.random.seed(42)
    random.seed(42)
    
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return int(np.count_nonzero(sq))
        
    def poly_mul(p1, p2):
        """Multiply two polynomials represented as lists."""
        if not p1 or not p2:
            return []
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1 == 0: continue
            for j, c2 in enumerate(p2):
                if c2 == 0: continue
                res[i+j] += c1 * c2
        # Trim trailing zeros
        while len(res) > 0 and res[-1] == 0:
            res.pop()
        return res if res else [0]

    def binom_to_list(a, b, sign):
        """Create a list for x^a - sign*x^b (or similar). 
        Let's define a binomial as c1*x^a + c2*x^b.
        For simplicity, we use sparse dicts or lists.
        """
        l = [0] * (max(a, b) + 1)
        l[a] = 1
        l[b] = sign
        return l

    # Phase 1: Factorization-based Search
    # We look for P(x) = prod (x^{a_i} - x^{b_i}) or similar.
    # To get 12 terms, we might need products of 2-3 binomials or 1 trinomial * binomial.
    # (x^a - x^b)(x^c - x^d) = x^{a+c} - x^{a+d} - x^{b+c} + x^{b+d}. (4 terms)
    # (x^a - x^b)(x^c - x^d)(x^e - x^f) = 8 terms.
    # (x^a - x^b)(x^c - x^d)(x^e - x^f)(x^g - x^h) = 16 terms.
    # We need exactly 12. This implies collisions in the exponents of the product.
    
    candidates = []
    
    # Strategy: Generate random products of binomials and check term count.
    # If term count is > 12, we can't use it directly.
    # If term count < 12, we can't use it directly.
    # We need exactly 12.
    
    # Let's try products of 3 binomials (8 terms base) and see if we can get 12? No, 3 binomials give max 8 terms.
    # 4 binomials give max 16 terms. We need exactly 12. This requires 4 collisions in the 16 terms.
    # This is a very specific constraint.
    
    # Alternative: Product of a Trinomial and a Binomial.
    # (x^a + x^b + x^c)(x^d + x^e) -> 6 terms.
    # (x^a + x^b + x^c)(x^d + x^e + x^f) -> 9 terms.
    # (x^a + x^b + x^c + x^d)(x^e + x^f + x^g) -> 12 terms.
    # This looks promising! A 4-term polynomial multiplied by a 3-term polynomial.
    # Or a 2-term by a 6-term, etc.
    
    # Let's search for P = P1 * P2 where P1 has k1 terms and P2 has k2 terms, k1*k2 >= 12.
    # If k1*k2 == 12 and no collisions, we have 12 terms.
    # If k1*k2 > 12, we need collisions.
    
    # Let's try k1=4, k2=3.
    # P1: 4 terms, P2: 3 terms.
    # We will randomly select exponents and signs.
    
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        # Generate P1 (4 terms)
        exps1 = sorted(random.sample(range(41), 4))
        signs1 = [random.choice([-1, 1]) for _ in range(4)]
        p1 = [0] * 41
        for e, s in zip(exps1, signs1):
            p1[e] = s
            
        # Generate P2 (3 terms)
        exps2 = sorted(random.sample(range(41), 3))
        signs2 = [random.choice([-1, 1]) for _ in range(3)]
        p2 = [0] * 41
        for e, s in zip(exps2, signs2):
            p2[e] = s
            
        p = poly_mul(p1, p2)
        
        if len(p) > 41:
            continue
            
        # Pad to length 41
        if len(p) < 41:
            p = p + [0] * (41 - len(p))
        else:
            p = p[:41]
            
        if p[40] == 0:
            # Shift to make leading term non-zero? Or just skip?
            # The constraint is "nonzero final coefficient" of the list.
            # If p[40] is 0, we can shift the whole polynomial by k such that max_exp + k <= 40.
            # But we want to keep it simple. Let's just check if it's valid as is.
            # If p[40] is 0, it's invalid unless we shift.
            # Shifting P(x) to x^k P(x) doesn't change the number of non-zero terms in the square.
            # So we can shift it to fit in [0, 40].
            max_exp = len(p) - 1
            while p[max_exp] == 0 and max_exp >= 0:
                max_exp -= 1
            if max_exp < 0:
                continue
            shift = 40 - max_exp
            if shift > 0:
                p = [0] * shift + p
            # Now p[40] should be non-zero
            
        nz = sum(1 for x in p if x != 0)
        if nz == 12:
            if any(abs(c) > 64 for c in p):
                continue
            candidates.append(p[:])
            
        # Also try 2 terms * 6 terms?
        # Or 2 * 2 * 3?
        
        # Let's also try 2 * 6
        if random.random() < 0.5:
            exps1 = sorted(random.sample(range(41), 2))
            signs1 = [random.choice([-1, 1]) for _ in range(2)]
            p1 = [0] * 41
            for e, s in zip(exps1, signs1):
                p1[e] = s
                
            exps2 = sorted(random.sample(range(41), 6))
            signs2 = [random.choice([-1, 1]) for _ in range(6)]
            p2 = [0] * 41
            for e, s in zip(exps2, signs2):
                p2[e] = s
                
            p = poly_mul(p1, p2)
            if len(p) > 41:
                continue
            if len(p) < 41:
                p = p + [0] * (41 - len(p))
            else:
                p = p[:41]
                
            max_exp = len(p) - 1
            while p[max_exp] == 0 and max_exp >= 0:
                max_exp -= 1
            if max_exp < 0:
                continue
            shift = 40 - max_exp
            if shift > 0:
                p = [0] * shift + p
                
            nz = sum(1 for x in p if x != 0)
            if nz == 12:
                if any(abs(c) > 64 for c in p):
                    continue
                candidates.append(p[:])

    # Evaluate candidates
    for p in candidates:
        if not _valid(p):
            continue
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_coeffs = p[:]
            
    # If no good candidate found, fallback to random search
    if best_coeffs is None:
        # Simple random search
        for _ in range(1000):
            if time.time() > deadline:
                break
            p = [0] * 41
            indices = random.sample(range(41), 12)
            for idx in indices:
                p[idx] = random.choice([-1, 1])
            if p[40] == 0:
                nz_indices = [i for i in range(41) if p[i] != 0]
                k = random.choice(nz_indices)
                p[40] = p[k]
                p[k] = 0
            if _valid(p):
                score = count_nz_square(p)
                if score < best_score:
                    best_score = score
                    best_coeffs = p[:]
                    
    if best_coeffs is None:
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
        best_score = count_nz_square(best_coeffs)
        
    # Phase 2: Local Refinement (Hill Climbing)
    # We will try to improve the best_coeffs found.
    # Move: change one coefficient, or move one term.
    
    current_poly = best_coeffs[:]
    current_score = best_score
    
    # Simple Hill Climbing with random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # If restart > 0, start from a random valid poly
        if restart > 0:
            # Find a random valid poly
            while time.time() < deadline:
                p = [0] * 41
                indices = random.sample(range(41), 12)
                for idx in indices:
                    p[idx] = random.choice([-1, 1])
                if p[40] == 0:
                    nz_indices = [i for i in range(41) if p[i] != 0]
                    k = random.choice(nz_indices)
                    p[40] = p[k]
                    p[k] = 0
                if _valid(p):
                    current_poly = p[:]
                    current_score = count_nz_square(p)
                    break
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all single-step moves
            # 1. Change value of a non-zero term
            for i in range(41):
                if current_poly[i] == 0:
                    continue
                old_val = current_poly[i]
                for new_val in range(-64, 65):
                    if new_val == old_val or new_val == 0:
                        continue
                    neighbor = current_poly[:]
                    neighbor[i] = new_val
                    if not _valid(neighbor):
                        continue
                    new_score = count_nz_square(neighbor)
                    if new_score < current_score:
                        current_poly = neighbor
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current_poly[:]
                        break # Restart loop with new current_poly
                        
            # 2. Move a term
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            z_indices = [i for i in range(41) if current_poly[i] == 0]
            if len(nz_indices) < 2 or not z_indices:
                continue
                
            for src in nz_indices:
                if src == 40:
                    continue
                val = current_poly[src]
                for dst in z_indices:
                    neighbor = current_poly[:]
                    neighbor[src] = 0
                    neighbor[dst] = val
                    if not _valid(neighbor):
                        continue
                    new_score = count_nz_square(neighbor)
                    if new_score < current_score:
                        current_poly = neighbor
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current_poly[:]
                        break # Restart loop
                if improved:
                    break
            
            # 3. Swap two terms
            if len(nz_indices) >= 2:
                for i1 in nz_indices:
                    if i1 == 40: continue
                    for i2 in nz_indices:
                        if i2 == 40 or i2 <= i1: continue
                        neighbor = current_poly[:]
                        neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
                        if not _valid(neighbor):
                            continue
                        new_score = count_nz_square(neighbor)
                        if new_score < current_score:
                            current_poly = neighbor
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_coeffs = current_poly[:]
                            break # Restart loop
                    if improved:
                        break
                        
    return best_coeffs
