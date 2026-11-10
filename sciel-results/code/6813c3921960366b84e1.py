import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True
        
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
        
    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        sq = np.convolve(coeffs, coeffs)
        return count_nonzero(sq)

    best_score = float('inf')
    best_poly = [1] * 12

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # 1. Deterministic Support-Compression Construction
    # Strategy: Build P from "blocks" of terms that produce very few nonzeros in P^2.
    # Block A: 2 terms (a, b) at (i, j). Square has 3 nonzeros: 2i, i+j, 2j.
    # Block B: 2 terms (c, d) at (k, l). Square has 3 nonzeros: 2k, k+l, 2l.
    # If supports of A^2 and B^2 are disjoint, total nonzeros = 6.
    # We need 12 terms. We can use 6 such blocks, or mix blocks of different sizes.
    
    # Let's try to construct a polynomial with 12 terms where the square has minimal nonzeros.
    # A known sparse square construction is based on roots of unity or specific symmetric patterns.
    # However, a simple constructive approach:
    # P(x) = sum_{i=0}^{11} c_i x^{p_i}
    
    # Let's search for a specific structured pattern that is known to be sparse.
    # Example: P(x) = x^{10} + x^9 + ... + x + 1 has 12 terms. Square has 21 terms (triangular). Score 21.
    # We want much lower.
    
    # Let's use the "orthogonal blocks" idea.
    # We need to pick 12 exponents and coefficients.
    # If we pick exponents such that all sums p_i + p_j (for i<=j) are distinct except for forced symmetries, 
    # the square will have ~78 terms. We want collisions.
    
    # Let's try a specific family:
    # P(x) = (x^a - 1) * Q(x) ? No, that changes term count.
    
    # Let's try to find a set of 12 exponents with many sum collisions.
    # This is equivalent to finding a Sidon set with many collisions? No, we WANT collisions.
    # We want the set of sums {p_i + p_j} to be as small as possible.
    
    # A very sparse square is achieved if P is a sum of disjoint "clusters" where intra-cluster sums don't hit inter-cluster sums.
    # Actually, the absolute minimum for a square of a polynomial with N terms is related to the structure.
    # For N=12, if P = A + B where A, B are monomials, P^2 has 3 terms.
    # If P = A + B + C (3 monomials), P^2 has up to 6 terms.
    # If we group 12 terms into 6 pairs of monomials that are far apart?
    # Let P = sum_{k=0}^5 (x^{e_{2k}} - x^{e_{2k+1}}).
    # If the supports of the squares of each pair are disjoint, and cross terms between pairs are disjoint too...
    # Wait, cross terms between pair k and pair m will produce terms x^{e_{2k}+e_{2m}} etc.
    # There are 6*5 = 30 cross pairs, each producing 4 terms (if distinct).
    # Total terms in square: 6 (self) + 30*4 (cross) = 126 terms. That's worse.
    
    # To minimize cross terms, we need the exponents to be "additively independent" in a way that creates collisions.
    # Or, we need the coefficients to cancel cross terms.
    # If P = sum c_i x^{p_i}, then (P^2)_{p_i+p_j} = 2 c_i c_j (for i!=j) and c_i^2 (for i=j).
    # We want many of these to be 0.
    # This requires c_i c_j = 0 for many pairs, which is impossible if all c_i != 0.
    # OR, we want 2 c_i c_j + 2 c_k c_l = 0 for distinct pairs (i,j) and (k,l) with p_i+p_j = p_k+p_l.
    
    # Let's try a specific construction:
    # P(x) = x^{10} - x^8 + x^6 - x^4 + x^2 - 1 ... no, that's 6 terms.
    # We need 12 terms.
    
    # Let's use a random search but guided by the "collision" objective.
    # Objective: Minimize number of distinct sums p_i + p_j where the sum of coefficients is non-zero.
    
    # Heuristic: Use exponents that are powers of 2?
    # p_i = 2^i. Sums are distinct (Sidon). Bad.
    # Use exponents that are arithmetic progression?
    # p_i = i. Sums range 0..22. 23 distinct sums.
    # If coefficients are chosen such that many cancel?
    # For AP, (P^2)_k is a convolution of the coefficient sequence with itself.
    # If P = sum_{i=0}^{11} c_i x^i, then P^2 = sum_{k=0}^{22} (sum_{i+j=k} c_i c_j) x^k.
    # We want many of these inner sums to be 0.
    # This is a linear algebra problem on the coefficients.
    
    # Let's try to solve for coefficients c_i in {-1, 1} for an AP support {0, 1, ..., 11}.
    # We want to minimize nonzeros in the autocorrelation of c.
    
    # Let's run a fast local search on the coefficients of a fixed support.
    # Support 1: {0, 1, ..., 11}
    # Support 2: {0, 2, 4, ..., 22}
    # Support 3: Random sparse sets
    
    supports_to_try = []
    
    # 1. AP supports
    for start in range(0, 20):
        for step in range(1, 4):
            support = [start + i*step for i in range(12)]
            if max(support) < 41:
                supports_to_try.append(support)
                
    # 2. Symmetric supports around center 20
    for _ in range(50):
        s = sorted(random.sample(range(41), 12))
        supports_to_try.append(s)
        
    # 3. "Clumped" supports (to encourage collisions)
    for _ in range(50):
        base = random.randint(0, 20)
        offsets = random.sample(range(-5, 6), 12)
        s = sorted(set(base + o for o in offsets if 0 <= base+o < 41))
        if len(s) == 12:
            supports_to_try.append(s)
        else:
            # Fill gaps
            while len(s) < 12:
                s.append(random.randint(0, 40))
            s = sorted(s)
            supports_to_try.append(s)

    # For each support, optimize coefficients
    for support in supports_to_try[:200]: # Limit count
        if time.time() > deadline - 10:
            break
            
        # Initialize with random signs
        coeffs = [0] * 41
        for i in support:
            coeffs[i] = random.choice([-1, 1])
        
        # Local search on coefficients
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in support:
                # Try flipping sign
                old_val = coeffs[i]
                coeffs[i] = -old_val
                score = evaluate(coeffs)
                if score < best_score:
                    best_score = score
                    best_poly = list(coeffs)
                    improved = True
                else:
                    coeffs[i] = old_val
                    
            # Try changing magnitude (if not +/- 1)
            # But we are restricted to integers. Let's stick to +/- 1 first, 
            # then maybe try small integers if time permits.
            
        # If we found a good score, do a more thorough coefficient search
        if best_score <= 20:
             # Try all combinations of a subset of coefficients? No, 2^12 is 4096.
             # Let's do a random walk in coefficient space.
             for _ in range(100):
                 if time.time() > deadline - 5:
                     break
                 perturbed = list(best_poly)
                 # Randomly change a few coefficients
                 for _ in range(3):
                     idx = random.choice(support)
                     perturbed[idx] = random.randint(-4, 4)
                     if perturbed[idx] == 0:
                         perturbed[idx] = 1
                 if is_valid(perturbed):
                     try_poly(perturbed)

    # Final polish: Local search on the best polynomial found
    if time.time() < deadline - 2:
        # Support perturbation
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            perturbed = list(best_poly)
            # Move a term
            nz = [i for i, c in enumerate(perturbed) if c != 0]
            if nz:
                idx = random.choice(nz)
                val = perturbed[idx]
                perturbed[idx] = 0
                new_pos = random.randint(0, 40)
                if new_pos < len(perturbed):
                    perturbed[new_pos] += val
                else:
                    perturbed.extend([0] * (new_pos - len(perturbed) + 1))
                    perturbed[new_pos] = val
                # Fix to 12 nonzeros
                nz2 = [i for i, c in enumerate(perturbed) if c != 0]
                if len(nz2) != 12:
                    continue
                try_poly(perturbed)
                
            # Coefficient perturbation
            perturbed = list(best_poly)
            nz = [i for i, c in enumerate(perturbed) if c != 0]
            if nz:
                idx = random.choice(nz)
                perturbed[idx] = random.randint(-64, 64)
                if perturbed[idx] == 0:
                    perturbed[idx] = 1
                try_poly(perturbed)

    return best_poly
