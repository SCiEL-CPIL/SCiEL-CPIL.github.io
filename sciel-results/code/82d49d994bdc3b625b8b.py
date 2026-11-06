"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 60s/90s limit
    deadline = start_time + 50

    def get_nz(poly):
        """Get list of (index, coeff) for non-zero terms."""
        return [(i, poly[i]) for i in range(41) if poly[i] != 0]

    def square_nonzero_count(poly):
        """Count nonzero terms in exact integer square of poly."""
        nz = get_nz(poly)
        # Use a dictionary to track sums, or a list since max degree is 80
        # Max degree is 40+40=80. List size 81.
        sq = [0] * 81
        for i, ai in nz:
            for j, aj in nz:
                sq[i + j] += ai * aj
        count = 0
        for c in sq:
            if c != 0:
                count += 1
        return count

    def local_search(poly, max_iter=20):
        """Hill-climbing: mutate one coefficient or move one term; accept if improves."""
        current = poly[:]
        current_score = square_nonzero_count(current)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            nz = get_nz(current)
            if not nz:
                break
            
            # Try changing a coefficient
            idx, old_val = random.choice(nz)
            new_val = random.randint(1, 64)
            if random.random() < 0.5:
                new_val = -new_val
            
            if new_val != old_val:
                current[idx] = new_val
                ns = square_nonzero_count(current)
                if ns < current_score:
                    current_score = ns
                else:
                    current[idx] = old_val
                    # If not improved, try moving a term
                    if random.random() < 0.5:
                         idx2, coeff = random.choice(nz)
                         new_pos = random.randint(0, 40)
                         if current[new_pos] == 0:
                             current[idx2] = 0
                             current[new_pos] = coeff
                             ns2 = square_nonzero_count(current)
                             if ns2 < current_score:
                                 current_score = ns2
                             else:
                                 current[new_pos] = 0
                                 current[idx2] = coeff
        return current, current_score

    # 1. Try a few structured candidates
    best_poly = [0] * 41
    best_score = float('inf')
    
    # Candidate 1: All ones at 0..11
    p1 = [0] * 41
    for i in range(12): p1[i] = 1
    s1 = square_nonzero_count(p1)
    if s1 < best_score:
        best_score = s1
        best_poly = p1[:]

    # Candidate 2: Geometric progression 1, 2, 4, ... might have cancellations? No, all positive.
    # Try alternating signs to encourage cancellation?
    # Example: 1 - x + x^2 - ...
    p2 = [0] * 41
    for i in range(12):
        p2[i] = 1 if i % 2 == 0 else -1
    s2 = square_nonzero_count(p2)
    if s2 < best_score:
        best_score = s2
        best_poly = p2[:]

    # Candidate 3: Random restarts with limited local search
    restarts = 0
    while time.time() < deadline:
        # Generate random poly
        positions = random.sample(range(41), 12)
        poly = [0] * 41
        for pos in positions:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            poly[pos] = c
        
        # Quick local search
        poly, score = local_search(poly, max_iter=10)
        
        if score < best_score:
            best_score = score
            best_poly = poly[:]
            # If we found a very good one, we can stop early or continue to refine
            if best_score <= 13: # Theoretical minimum is likely higher, but 13 is a good target
                 # Continue searching a bit more to see if we can go lower
                 pass
        restarts += 1
        if restarts > 1000: # Cap restarts to avoid infinite loop if time check fails
             break

    # Trim trailing zeros to ensure the final coefficient is non-zero
    # and satisfy length <= 41 with a non-empty, valid polynomial representation
    last_nz = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last_nz = i
            break
    
    if last_nz == -1:
        # Fallback: should not happen if logic holds, but ensure validity
        # Return a valid simple polynomial if something went wrong
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    return best_poly[:last_nz + 1]
# EVOLVE-BLOCK-END
