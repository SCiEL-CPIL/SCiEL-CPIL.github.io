import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _square_poly(coeffs):
    """Compute the square of a polynomial given by coefficient list (ascending degree)."""
    n = len(coeffs)
    # The square will have degree up to 2*(n-1), so length up to 2n-1
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            sq[i + j] += ci * cj
    # Trim trailing zeros but keep at least one element
    while len(sq) > 1 and sq[-1] == 0:
        sq.pop()
    return sq

def _valid_candidate(coeffs):
    """Check if coeffs is a valid candidate per the contract."""
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if _count_nonzero(coeffs) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = float('inf')
    best_candidate = [1] * 12  # Fallback: 12 ones. Square has 23 terms.
    
    # Strategy: Try to find a polynomial with 12 terms whose square has few nonzeros.
    # A known structure for sparse squares:
    # 1. Palindromic/anti-palindromic structures.
    # 2. Sums of disjoint monomials where cross terms cancel or are sparse.
    # 3. "Two-cluster" approach: P = A + B where A and B have disjoint support ranges
    #    such that cross terms don't interfere with the main sparse structure, or
    #    use specific identities.
    
    # Let's try a structured search around known sparse patterns.
    # Pattern: x^k * (a0 + a1*x + ... + am*x^m) * (b0 + b1*x + ... + bn*x^n)
    # If we pick coefficients carefully, we might reduce the square's density.
    
    # Simple heuristic: Random search with local optimization on the square's density.
    # But pure random is slow. Let's use a "seeded" random search with constraints.
    
    # We will generate random polynomials with exactly 12 nonzero terms,
    # coefficients in [-64, 64], degree <= 40.
    # We'll bias towards low degrees first as they are easier to cancel.
    
    # To speed up, we can pre-compute the square count for small changes?
    # No, let's just do a fast random search with a time guard.
    
    # Optimization: Instead of full random, try specific families.
    # Family 1: (1 + x + ... + x^{k-1}) * (sparse part)
    # Family 2: Symmetric polynomials.
    
    # Let's try a hybrid:
    # 1. Try a few hardcoded "good" candidates if we know them.
    # 2. Random search with local refinement.
    
    # Hardcoded attempt:
    # If we take P = 1 + x + x^2 + ... + x^11, square has 23 terms.
    # Can we do better?
    # Consider P = (1 + x^k)(1 + x^{2k})... No, that increases terms.
    # Consider cancellation: (1 - x)(1 + x + x^2 + x^3) = 1 - x^4. Square is 1 - 2x^4 + x^8 (3 terms).
    # But we need 12 terms in P.
    # Let P = (1 - x)(1 + x + ... + x^{11}) = 1 - x^{12}. This has 2 terms. Not 12.
    # We need exactly 12 nonzero terms in P.
    
    # Let's try P = 1 + x + ... + x^5 + x^7 + ... ?
    
    # Efficient Search Strategy:
    # Generate random sets of 12 indices in [0, 40].
    # Assign random coefficients in [-64, 64] (non-zero).
    # Calculate square count.
    # If count < best, update.
    # Use local search: flip a coefficient or index slightly.
    
    # To make it faster, we can restrict the degree range initially.
    
    max_degree = 40
    num_terms = 12
    
    # Pre-allocate for speed
    def random_candidate():
        # Choose 12 distinct indices
        indices = random.sample(range(max_degree + 1), num_terms)
        indices.sort()
        coeffs = [0] * (max_degree + 1)
        for i in indices:
            # Coefficients must be non-zero and in [-64, 64]
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            coeffs[i] = c
        # Ensure last coeff is non-zero (it is, since we picked indices randomly)
        # But we must return length <= 41.
        # The contract says "length<=41".
        # If the highest index is 40, length is 41.
        return coeffs

    # Local search step: change one coefficient or move one index.
    def local_search(coeffs):
        current_sq = _square_poly(coeffs)
        current_count = _count_nonzero(current_sq)
        
        # Try modifying a random nonzero term
        nonzero_indices = [i for i, c in enumerate(coeffs) if c != 0]
        if not nonzero_indices:
            return coeffs, current_count
            
        idx = random.choice(nonzero_indices)
        old_val = coeffs[idx]
        
        # Try changing value
        new_val = random.randint(1, 64)
        if random.random() < 0.5:
            new_val = -new_val
        if new_val == old_val:
            new_val = old_val + 1 if old_val < 64 else old_val - 1
            
        coeffs[idx] = new_val
        new_sq = _square_poly(coeffs)
        new_count = _count_nonzero(new_sq)
        
        if new_count < current_count:
            return coeffs, new_count
        else:
            coeffs[idx] = old_val
            return coeffs, current_count

    # Main loop
    while time.time() < deadline:
        # Generate a new candidate
        cand = random_candidate()
        sq = _square_poly(cand)
        count = _count_nonzero(sq)
        
        if count < best_score:
            best_score = count
            best_candidate = cand[:]
            
        # If we find a very good one, do local search
        if count <= best_score + 5:
            # Do a few local search steps
            for _ in range(10):
                if time.time() >= deadline:
                    break
                cand, count = local_search(cand)
                if count < best_score:
                    best_score = count
                    best_candidate = cand[:]
                    
        # Early exit if we find a perfect square (count 1? Impossible for 12 terms unless zero, but nonzero final coeff required)
        # Minimum square terms for 12-term poly?
        # If P = x^k * Q, square is x^{2k} * Q^2. Same count.
        # Theoretical minimum?
        # (1+x)^2 = 1+2x+x^2 (3 terms)
        # (1+x+x^2)^2 = 1+2x+3x^2+2x^3+x^4 (5 terms)
        # Generally, square of n-term poly has up to 2n-1 terms.
        # Can we get fewer? Yes, via cancellation.
        # Example: (1-x)^2 = 1-2x+x^2 (3 terms).
        # (1-x+x^2)^2 = 1 - 2x + 3x^2 - 2x^3 + x^4 (5 terms).
        # It seems hard to get significantly below 2n-1 without specific structures.
        # For n=12, 23 is the max. Can we get 12? 10?
        
        # If we find a score of 12 or less, that's excellent.
        # Let's continue searching.

    # Ensure valid format for return
    # The contract requires:
    # - length <= 41
    # - nonzero final coefficient
    # - entries in [-64, 64]
    # - exactly 12 nonzero terms
    
    # Our generator ensures this.
    # However, if best_candidate is the fallback [1]*12, it is valid.
    
    # One final check: trim trailing zeros to minimize length?
    # The contract says "length<=41". It doesn't say "minimal length".
    # But "nonzero final coefficient" implies we shouldn't have trailing zeros.
    # My _square_poly trims, but construct returns the original coeffs.
    # I should trim the returned coeffs to remove trailing zeros to be safe, 
    # although the input to _square_poly handles it, the return value must be valid.
    
    # Trim trailing zeros from best_candidate
    while len(best_candidate) > 1 and best_candidate[-1] == 0:
        best_candidate.pop()
        
    # Ensure length is not 0
    if not best_candidate:
        best_candidate = [1]
        
    return best_candidate
# EVOLVE-BLOCK-END
