"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _square_count(coeffs):
    """Count nonzero terms in the square of a polynomial given as coefficient list."""
    if len(coeffs) == 0:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _construct_from_ap(start, step, signs, magnitudes=None):
    """Construct polynomial from arithmetic progression with given signs and magnitudes."""
    n = len(signs)
    if magnitudes is None:
        magnitudes = [1] * n
    max_deg = start + (n - 1) * step
    result = [0] * (max_deg + 1)
    for i in range(n):
        deg = start + i * step
        if 0 <= deg <= max_deg:
            result[deg] = signs[i] * magnitudes[i]
    return result

def _generate_identity_candidates():
    """Generate candidates based on known algebraic structures that produce sparse squares."""
    candidates = []
    
    # Strategy: Use polynomials of form (1 + x^a + x^b + ...) where the square has cancellations
    # Known identity: (1 - x + x^2 - x^3 + ...)^2 can have structure
    
    # Candidate 1: Alternating signs in AP
    for start in range(0, 5):
        for step in range(1, 8):
            n = 12
            max_deg = start + (n - 1) * step
            if max_deg > 40:
                continue
            # Alternating signs
            signs = [(-1)**i for i in range(n)]
            poly = _construct_from_ap(start, step, signs)
            if len([x for x in poly if x != 0]) == 12:
                candidates.append(poly)
    
    # Candidate 2: Symmetric sign patterns
    for start in range(0, 5):
        for step in range(1, 8):
            n = 12
            max_deg = start + (n - 1) * step
            if max_deg > 40:
                continue
            # Pattern: + + - - + + - - + + - -
            for pattern in [[1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                           [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
                           [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1]]:
                poly = _construct_from_ap(start, step, pattern)
                if len([x for x in poly if x != 0]) == 12:
                    candidates.append(poly)
    
    return candidates

def _local_search(poly, max_iters=5000, deadline=None):
    """Local search: mutate one coefficient at a time to reduce square sparsity."""
    best = poly[:]
    best_score = _square_count(best)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random position to mutate
        pos = random.randint(0, len(best) - 1)
        old_val = best[pos]
        
        # Try changing to a different value
        new_val = old_val + random.choice([-2, -1, 1, 2])
        if new_val == old_val:
            continue
        if abs(new_val) > 64:
            continue
        
        best[pos] = new_val
        
        # Check constraints: exactly 12 nonzero terms
        nz_count = sum(1 for x in best if x != 0)
        if nz_count != 12:
            best[pos] = old_val
            continue
        
        # Check final coefficient is nonzero
        if best[-1] == 0:
            best[pos] = old_val
            continue
        
        score = _square_count(best)
        if score < best_score:
            best_score = score
            # Keep the change
        else:
            # Revert with some probability (simulated annealing-like)
            if random.random() > 0.1:
                best[pos] = old_val
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_poly = None
    best_score = 1000
    
    # Phase 1: Try structured identity candidates
    candidates = _generate_identity_candidates()
    
    for poly in candidates:
        if time.time() > deadline - 30.0:
            break
        
        score = _square_count(poly)
        if score < best_score:
            best_score = score
            best_poly = poly[:]
    
    # Phase 2: Local search from best candidate and from random starts
    if best_poly is not None:
        improved, improved_score = _local_search(best_poly, max_iters=10000, deadline=deadline - 10.0)
        if improved_score < best_score:
            best_score = improved_score
            best_poly = improved
    
    # Phase 3: Random restarts with local search
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random polynomial with 12 nonzero terms
        poly = [0] * random.randint(15, 40)
        positions = random.sample(range(len(poly)), 12)
        for pos in positions:
            poly[pos] = random.choice([-2, -1, 1, 2])
        
        # Ensure final coefficient is nonzero
        if poly[-1] == 0:
            poly[-1] = 1
        
        # Ensure exactly 12 nonzero
        nz = sum(1 for x in poly if x != 0)
        if nz != 12:
            continue
        
        score = _square_count(poly)
        if score < best_score:
            best_score = score
            best_poly = poly[:]
        
        # Local search
        improved, improved_score = _local_search(poly, max_iters=2000, deadline=deadline - 5.0)
        if improved_score < best_score:
            best_score = improved_score
            best_poly = improved
    
    # Fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # That's only 3 nonzero... let's make a proper fallback
        best_poly = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # Final validation
    assert len(best_poly) <= 41
    assert best_poly[-1] != 0
    assert all(abs(x) <= 64 for x in best_poly)
    assert sum(1 for x in best_poly if x != 0) == 12
    
    return best_poly

# EVOLVE-BLOCK-END
