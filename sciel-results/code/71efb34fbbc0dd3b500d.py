"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    def get_nz(poly):
        return [(i, poly[i]) for i in range(len(poly)) if poly[i] != 0]

    def square_nonzero_count_fast(poly):
        """Count nonzero terms in exact integer square using symmetry."""
        nz = get_nz(poly)
        n = len(nz)
        if n == 0:
            return 0
        max_deg = 2 * (len(poly) - 1)
        sq = [0] * (max_deg + 1)
        # Use symmetry: for i <= j, add 2*a_i*a_j to sq[i+j], and a_i*a_i to sq[2*i]
        for ii in range(n):
            i, ai = nz[ii]
            # Diagonal: i + i
            sq[2 * i] += ai * ai
            # Off-diagonal: i + j for j > i, count twice
            for jj in range(ii + 1, n):
                j, aj = nz[jj]
                sq[i + j] += 2 * ai * aj
        count = 0
        for c in sq:
            if c != 0:
                count += 1
        return count

    def local_search(poly, max_iter=30, deadline=None):
        """Hill-climbing with diverse moves."""
        current = poly[:]
        current_score = square_nonzero_count_fast(current)
        if current_score == 12:  # Can't do better than 12 (one per term squared, if all merge)
            return current, current_score
        
        for _ in range(max_iter):
            if deadline and time.time() > deadline:
                break
            nz = get_nz(current)
            if not nz:
                break
            
            improved = False
            move_type = random.random()
            
            if move_type < 0.4:
                # Change a coefficient value
                idx, old_val = random.choice(nz)
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val != old_val:
                    current[idx] = new_val
                    ns = square_nonzero_count_fast(current)
                    if ns < current_score:
                        current_score = ns
                        improved = True
                    else:
                        current[idx] = old_val
            
            elif move_type < 0.7:
                # Move a term to a new position
                idx, coeff = random.choice(nz)
                new_pos = random.randint(0, len(current) - 1)
                if current[new_pos] == 0 and new_pos != idx:
                    current[idx] = 0
                    current[new_pos] = coeff
                    ns = square_nonzero_count_fast(current)
                    if ns < current_score:
                        current_score = ns
                        improved = True
                    else:
                        current[new_pos] = 0
                        current[idx] = coeff
            
            else:
                # Change coefficient and possibly move
                idx, old_val = random.choice(nz)
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val != old_val:
                    current[idx] = new_val
                    ns = square_nonzero_count_fast(current)
                    if ns < current_score:
                        current_score = ns
                        improved = True
                    else:
                        current[idx] = old_val
        
        return current, current_score

    best_poly = [0] * 41
    best_score = float('inf')
    
    # Try structured candidates
    # Candidate: terms at 0,1,2,...,11 with all 1s
    p1 = [0] * 41
    for i in range(12):
        p1[i] = 1
    s1 = square_nonzero_count_fast(p1)
    if s1 < best_score:
        best_score = s1
        best_poly = p1[:]
    
    # Candidate: alternating signs
    p2 = [0] * 41
    for i in range(12):
        p2[i] = 1 if i % 2 == 0 else -1
    s2 = square_nonzero_count_fast(p2)
    if s2 < best_score:
        best_score = s2
        best_poly = p2[:]
    
    # Candidate: sparse placement with powers of 2
    p3 = [0] * 41
    for i in range(12):
        p3[i * 4] = 1  # positions 0,4,8,...,44 -> but 44 > 40, so use 0,4,8,...,40
    # Actually let's use positions 0,3,6,9,...,33
    p3 = [0] * 41
    for i in range(12):
        pos = i * 3
        if pos < 41:
            p3[pos] = 1
    s3 = square_nonzero_count_fast(p3)
    if s3 < best_score:
        best_score = s3
        best_poly = p3[:]
    
    # Candidate: geometric with ratio 2, alternating signs for cancellation
    p4 = [0] * 41
    pos = 0
    for i in range(12):
        if pos >= 41:
            break
        p4[pos] = 1 if i % 2 == 0 else -1
        pos += 1
    s4 = square_nonzero_count_fast(p4)
    if s4 < best_score:
        best_score = s4
        best_poly = p4[:]
    
    # Random restarts with local search
    restarts = 0
    while time.time() < deadline:
        # Generate random poly with 12 nonzero terms
        positions = random.sample(range(41), 12)
        poly = [0] * 41
        for pos in positions:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            poly[pos] = c
        
        # Local search
        poly, score = local_search(poly, max_iter=15, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_poly = poly[:]
            if best_score <= 13:
                # Very good, try to refine further
                poly2, score2 = local_search(poly, max_iter=20, deadline=deadline)
                if score2 < best_score:
                    best_score = score2
                    best_poly = poly2[:]
        
        restarts += 1
        if restarts > 5000:
            break
    
    # Trim trailing zeros
    last_nz = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last_nz = i
            break
    
    if last_nz == -1:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    return best_poly[:last_nz + 1]
# EVOLVE-BLOCK-END
