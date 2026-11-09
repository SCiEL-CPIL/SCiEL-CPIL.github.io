import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Best known solution found so far
    best_terms = None
    best_score = float('inf')
    
    # Helper to compute square sparsity
    def square_sparsity(terms):
        # terms is a list of (deg, coeff)
        # Result is a dict {deg: coeff}
        sq = {}
        n = len(terms)
        for i in range(n):
            d1, c1 = terms[i]
            for j in range(i, n):
                d2, c2 = terms[j]
                d = d1 + d2
                c = c1 * c2
                if i != j:
                    c *= 2
                sq[d] = sq.get(d, 0) + c
        # Count non-zeros
        count = 0
        for v in sq.values():
            if v != 0:
                count += 1
        return count, sq

    # Helper to convert terms to coefficient list
    def terms_to_list(terms):
        if not terms:
            return []
        max_deg = max(d for d, c in terms)
        lst = [0] * (max_deg + 1)
        for d, c in terms:
            lst[d] = c
        # Ensure last coefficient is non-zero (it is, by construction of terms)
        # But we must ensure length <= 41
        if max_deg > 40:
            return None
        return lst

    # Initial candidates
    # 1. Random
    # 2. Symmetric / Palindromic
    # 3. Specific patterns like 1 + x^k + ...

    def random_terms():
        # 12 distinct degrees in [0, 40]
        degs = random.sample(range(0, 41), 12)
        degs.sort()
        coeffs = [random.randint(1, 64) * (1 if random.random() > 0.5 else -1) for _ in range(12)]
        return list(zip(degs, coeffs))

    # Evaluate a set of initial candidates
    candidates = []
    
    # Random starts
    for _ in range(20):
        candidates.append(random_terms())
        
    # Structured starts
    # Symmetric degrees around 20
    base_degs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Try palindromic coefficients
    c1 = [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1]
    candidates.append(list(zip(base_degs, c1)))
    
    # Try to find a good start
    for terms in candidates:
        score, _ = square_sparsity(terms)
        if score < best_score:
            best_score = score
            best_terms = terms[:]

    # Local Search
    # We will iterate, picking a random term and perturbing it.
    # Perturbation: change degree (shift by small amount) or change coefficient.
    
    # To speed up, we can use a list representation for the square dict? 
    # No, dict is fine for 12 terms (max 78 entries in square).
    
    while time.time() < deadline:
        # Create a new candidate by perturbing the best one
        # Or restart from a random one if stuck? 
        # Let's stick to hill climbing on the current best for now, 
        # but occasionally restart to avoid local minima.
        
        if random.random() < 0.1:
            # Restart
            current_terms = random_terms()
            score, _ = square_sparsity(current_terms)
            if score < best_score:
                best_score = score
                best_terms = current_terms[:]
        else:
            current_terms = best_terms[:]

        # Perturb
        idx = random.randint(0, 11)
        d, c = current_terms[idx]
        
        if random.random() < 0.5:
            # Change degree
            new_d = d + random.randint(-3, 3)
            if 0 <= new_d <= 40 and new_d not in [dd for dd, cc in current_terms if dd != d]:
                current_terms[idx] = (new_d, c)
            else:
                continue
        else:
            # Change coefficient
            new_c = random.randint(1, 64) * (1 if random.random() > 0.5 else -1)
            if new_c != c:
                current_terms[idx] = (d, new_c)
            else:
                continue
        
        # Evaluate
        score, _ = square_sparsity(current_terms)
        
        if score < best_score:
            best_score = score
            best_terms = current_terms[:]
            
            # If we hit a very low score, we might want to polish it further
            # The minimum possible is 1 (perfect square of a monomial, but we need 12 terms).
            # With 12 terms, can we get < 20?
            # (1+x^2+x^4...)^2 has many terms.
            # Let's see what we get.

    # Convert best_terms to list
    # Ensure valid output
    if best_terms is None:
        # Fallback
        return [1] * 12
    
    # Sort by degree to ensure correct list construction
    best_terms.sort(key=lambda x: x[0])
    
    lst = terms_to_list(best_terms)
    if lst is None:
        return [1] * 12
        
    # Final check: length <= 41, nonzero final coeff, exactly 12 nonzero terms
    if len(lst) > 41:
        return [1] * 12
    if lst[-1] == 0:
        # Should not happen if we track terms correctly, but just in case
        return [1] * 12
        
    nonzero_count = sum(1 for x in lst if x != 0)
    if nonzero_count != 12:
        return [1] * 12
        
    return lst

# EVOLVE-BLOCK-END
