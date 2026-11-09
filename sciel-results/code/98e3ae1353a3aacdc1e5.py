import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Max degree 40, so vector length 41
    N = 41
    TERMS = 12
    
    # Helper to get random valid value
    def rand_val():
        return random.randint(1, 64) * random.choice([1, -1])

    def init_state():
        # Pick 12 distinct exponents
        exps = random.sample(range(N), TERMS)
        vals = [rand_val() for _ in range(TERMS)]
        poly = np.zeros(N, dtype=int)
        for e, v in zip(exps, vals):
            poly[e] = v
        return poly, exps

    def get_score(poly):
        sq = np.convolve(poly, poly)
        return np.count_nonzero(sq)

    # Initial state
    curr_poly, curr_exps = init_state()
    curr_score = get_score(curr_poly)
    
    best_poly = curr_poly.copy()
    best_score = curr_score
    best_exps = curr_exps[:]

    T = 10.0
    T_min = 0.01
    cooling = 0.9999 # Slower cooling for more exploration with fast eval
    
    while time.time() < deadline:
        # Generate neighbor
        new_poly = curr_poly.copy()
        new_exps = curr_exps[:]
        
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Change one term (exponent and/or coeff)
            idx = random.randint(0, TERMS-1)
            old_e = new_exps[idx]
            old_v = new_poly[old_e]
            
            # Pick new exponent (must be distinct from others)
            # Others are new_exps excluding idx
            others = [e for i, e in enumerate(new_exps) if i != idx]
            # Available exponents
            avail = [e for e in range(N) if e not in others]
            # Include old_e? If we keep same exponent, it's just a coeff change.
            # If we pick new_e, we must clear old_e.
            
            new_e = random.choice(avail)
            new_v = rand_val()
            
            if new_e != old_e:
                new_poly[old_e] = 0
                new_exps[idx] = new_e
                # Note: if new_e was in others, we'd have a collision, but avail excludes others.
            else:
                # Just coeff change
                pass
                
            new_poly[new_e] = new_v

        elif move_type == 1:
            # Change two terms
            i, j = random.sample(range(TERMS), 2)
            e1, e2 = new_exps[i], new_exps[j]
            v1, v2 = new_poly[e1], new_poly[e2]
            
            # Clear them
            new_poly[e1] = 0
            new_poly[e2] = 0
            
            # Pick new exponents
            others = [e for k, e in enumerate(new_exps) if k not in [i, j]]
            avail = [e for e in range(N) if e not in others]
            
            new_e1, new_e2 = random.sample(avail, 2)
            new_v1, new_v2 = rand_val(), rand_val()
            
            new_poly[new_e1] = new_v1
            new_poly[new_e2] = new_v2
            new_exps[i] = new_e1
            new_exps[j] = new_e2

        else:
            # Change coefficient only
            idx = random.randint(0, TERMS-1)
            e = new_exps[idx]
            new_poly[e] = rand_val()

        new_score = get_score(new_poly)
        
        delta = new_score - curr_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            curr_poly = new_poly
            curr_exps = new_exps
            curr_score = new_score
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr_poly.copy()
                best_exps = curr_exps[:]
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Convert best_poly to list
    # Find max exponent
    max_exp = np.max(best_exps)
    # List length must be max_exp + 1
    # But wait, the problem says "length <= 41".
    # If max_exp is 40, length is 41.
    # If max_exp is 0, length is 1.
    
    # However, "nonzero final coefficient" implies the last element of the list is non-zero.
    # If we return a list of length L, the element at L-1 corresponds to x^(L-1).
    # So we must ensure that the highest degree term in our polynomial is at the end of the list.
    # Our `best_poly` is a fixed size array of 41.
    # We need to trim it to `max_exp + 1`.
    
    result = best_poly[:max_exp+1].tolist()
    
    # Validation
    # 1. Length <= 41 (Yes, max 41)
    # 2. Exactly 12 non-zeros
    # 3. Non-zero final coefficient (result[-1] != 0)
    # 4. Integers in [-64, 64]
    
    if len(result) > 41: return [1]*12
    if sum(1 for x in result if x != 0) != 12: return [1]*12
    if result[-1] == 0: return [1]*12
    
    return result
