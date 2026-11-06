"""Improved program for sparse squares of twelve-term integer polynomials."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _square_nonzero_sparse(terms):
    """Given list of (exponent, coefficient) pairs, return number of nonzero terms in the square.
    Each term has distinct exponents and nonzero coefficients.
    Square = sum over all ordered pairs (i,j) of c_i*c_j * x^(e_i+e_j)
    We can use unordered pairs: for i<j, contribution is 2*c_i*c_j at e_i+e_j,
    and for i==j, contribution is c_i^2 at 2*e_i.
    """
    sums = defaultdict(int)
    n = len(terms)
    for i in range(n):
        e_i, c_i = terms[i]
        # i==i
        sums[2 * e_i] += c_i * c_i
        for j in range(i + 1, n):
            e_j, c_j = terms[j]
            sums[e_i + e_j] += 2 * c_i * c_j
    return sum(1 for v in sums.values() if v != 0)

def _to_coeff_list(terms):
    """Convert list of (exp, coeff) to coefficient list in ascending degree."""
    if not terms:
        return [1]
    max_exp = max(e for e, c in terms)
    coeffs = [0] * (max_exp + 1)
    for e, c in terms:
        coeffs[e] = c
    # Ensure length <= 41 and final coefficient nonzero
    while len(coeffs) > 41:
        coeffs = coeffs[:-1]
    # Ensure final coefficient is nonzero
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    if not coeffs:
        coeffs = [1]
    return coeffs

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    
    best_count = 23  # baseline: 12 consecutive ones -> 23 nonzero square terms
    best_terms = [(i, 1) for i in range(12)]
    
    # Precompute: try structured patterns first
    # Pattern 1: all ones with gaps
    # Pattern 2: alternating signs
    # Pattern 3: powers of 2 exponents
    
    # Try a few structured candidates
    structured = [
        [(i, 1) for i in range(12)],  # consecutive
        [(i, 1 if i % 2 == 0 else -1) for i in range(12)],  # alternating
        [(0,1),(1,1),(2,1),(4,1),(5,1),(6,1),(8,1),(9,1),(10,1),(12,1),(13,1),(14,1)],
        [(0,1),(1,1),(2,1),(4,1),(8,1),(16,1),(3,1),(5,1),(7,1),(9,1),(11,1),(13,1)],
        [(0,1),(1,-1),(2,1),(3,-1),(4,1),(5,-1),(6,1),(7,-1),(8,1),(9,-1),(10,1),(11,-1)],
    ]
    
    for terms in structured:
        cnt = _square_nonzero_sparse(terms)
        if cnt < best_count:
            best_count = cnt
            best_terms = terms[:]
    
    random.seed(12345)
    coeff_pool = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10]
    
    def try_local_search(terms, max_iter=200):
        """Local refinement: mutate one term at a time."""
        nonlocal best_count, best_terms
        current = terms[:]
        current_count = _square_nonzero_sparse(current)
        if current_count < best_count:
            best_count = current_count
            best_terms = current[:]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            old_exp, old_coeff = current[idx]
            
            # Try changing coefficient
            new_coeff = random.choice(coeff_pool)
            if new_coeff != old_coeff:
                current[idx] = (old_exp, new_coeff)
                cnt = _square_nonzero_sparse(current)
                if cnt < current_count:
                    current_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_terms = current[:]
                else:
                    current[idx] = (old_exp, old_coeff)
            
            # Try changing exponent
            new_exp = old_exp + random.choice([-3,-2,-1,1,2,3])
            if 0 <= new_exp <= 40:
                # Check no duplicate exponent
                other_exps = [current[j][0] for j in range(12) if j != idx]
                if new_exp not in other_exps:
                    current[idx] = (new_exp, old_coeff)
                    cnt = _square_nonzero_sparse(current)
                    if cnt < current_count:
                        current_count = cnt
                        if cnt < best_count:
                            best_count = cnt
                            best_terms = current[:]
                    else:
                        current[idx] = (old_exp, old_coeff)
    
    # Random search
    trial = 0
    while time.time() < deadline:
        trial += 1
        
        # Choose 12 distinct exponents
        # Bias toward clustered exponents for more cancellation opportunities
        if random.random() < 0.3:
            # Clustered: pick a base and spread
            base = random.randint(0, 30)
            exps = set()
            while len(exps) < 12:
                e = base + random.randint(-5, 15)
                if 0 <= e <= 40:
                    exps.add(e)
        else:
            exps = set(random.sample(range(41), 12))
        
        exps = sorted(exps)
        coeffs = [random.choice(coeff_pool) for _ in range(12)]
        terms = list(zip(exps, coeffs))
        
        cnt = _square_nonzero_sparse(terms)
        if cnt < best_count:
            best_count = cnt
            best_terms = terms[:]
            # Local refinement
            try_local_search(terms, max_iter=100)
        elif cnt <= best_count + 2 and trial % 10 == 0:
            # Near-miss: try local search
            try_local_search(terms, max_iter=50)
        
        if best_count <= 14:
            # Getting good, intensify search
            if trial % 5 == 0:
                try_local_search(best_terms, max_iter=200)
    
    # Final local search on best solution
    try_local_search(best_terms, max_iter=500)
    
    # Convert to coefficient list
    result = _to_coeff_list(best_terms)
    
    # Validate
    assert len(result) <= 41, f"Length {len(result)} > 41"
    assert result[-1] != 0, "Final coefficient is zero"
    assert all(-64 <= c <= 64 for c in result), "Coefficient out of range"
    assert sum(1 for c in result if c != 0) == 12, f"Nonzero count {sum(1 for c in result if c != 0)} != 12"
    
    return result
# EVOLVE-BLOCK-END
