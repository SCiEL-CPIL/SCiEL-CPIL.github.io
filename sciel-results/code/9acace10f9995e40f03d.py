import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    # Helper to count nonzero terms in square
    def count_square_terms(poly):
        # poly is dict {exp: coeff}
        square = {}
        for e1, c1 in poly.items():
            for e2, c2 in poly.items():
                e = e1 + e2
                square[e] = square.get(e, 0) + c1 * c2
        return sum(1 for c in square.values() if c != 0)
        
    # Helper to generate random poly
    def gen_random_poly():
        exponents = random.sample(range(0, 41), 12)
        coeffs = [random.randint(1, 64) for _ in range(12)]
        # Random signs
        for i in range(12):
            if random.random() < 0.5:
                coeffs[i] = -coeffs[i]
        return {e: c for e, c in zip(exponents, coeffs)}

    best_poly = None
    best_score = 100 # Max possible is 24 (12*2) but usually less
    
    # Initialize with a few random candidates and a known baseline
    candidates = [gen_random_poly() for _ in range(5)]
    # Add a structured one: 1..12
    candidates.append({i: 1 for i in range(12)})
    
    for poly in candidates:
        current_score = count_square_terms(poly)
        if current_score < best_score:
            best_score = current_score
            best_poly = dict(poly)
            
    # Local Search Loop
    while time.time() < deadline:
        # Pick a candidate to improve (best so far or random from pool)
        # Let's focus on the best one found so far to try to beat it
        poly = dict(best_poly)
        current_score = count_square_terms(poly)
        
        improved = False
        # Try local moves
        # 1. Flip signs
        for e in list(poly.keys()):
            new_poly = dict(poly)
            new_poly[e] = -new_poly[e]
            s = count_square_terms(new_poly)
            if s < current_score:
                poly = new_poly
                current_score = s
                improved = True
                break
        if improved:
            if current_score < best_score:
                best_score = current_score
                best_poly = dict(poly)
            continue
            
        # 2. Change coefficient magnitude
        for e in list(poly.keys()):
            for delta in [-1, 1, -2, 2, -5, 5]:
                new_c = poly[e] + delta
                if -64 <= new_c <= 64 and new_c != 0:
                    new_poly = dict(poly)
                    new_poly[e] = new_c
                    s = count_square_terms(new_poly)
                    if s < current_score:
                        poly = new_poly
                        current_score = s
                        improved = True
                        break
            if improved: break
        if improved:
            if current_score < best_score:
                best_score = current_score
                best_poly = dict(poly)
            continue
            
        # 3. Move exponent
        # Try moving one term to a nearby empty exponent
        exponents = sorted(poly.keys())
        for i, e in enumerate(exponents):
            # Try nearby slots
            for new_e in [e-1, e+1, e-2, e+2, e-3, e+3]:
                if 0 <= new_e <= 40 and new_e not in poly:
                    new_poly = dict(poly)
                    del new_poly[e]
                    new_poly[new_e] = poly[e]
                    s = count_square_terms(new_poly)
                    if s < current_score:
                        poly = new_poly
                        current_score = s
                        improved = True
                        break
            if improved: break
        if improved:
            if current_score < best_score:
                best_score = current_score
                best_poly = dict(poly)
            continue
            
        # If no improvement found in a full pass, restart with a random perturbation or new candidate
        # To avoid getting stuck in local minima, occasionally switch to a new random candidate
        if random.random() < 0.1:
             poly = gen_random_poly()
             current_score = count_square_terms(poly)
             if current_score < best_score:
                 best_score = current_score
                 best_poly = dict(poly)
             # Continue optimizing this new poly in next iterations? 
             # For simplicity, let's just restart the local search on this new poly
             # by setting best_poly to it if it's good, or just letting it be.
             # Actually, it's better to keep the global best and just restart the *current* optimization target.
             # But my loop structure optimizes `best_poly`. 
             # Let's modify: maintain a pool of candidates? 
             # Simpler: Just restart the local search on the current best. 
             # If stuck, generate a new random poly, check its score. If better, update best. 
             # Then try to improve the new best.
             
             # If the new random poly is worse, we are stuck. 
             # Let's try a "kick": change a random coefficient significantly.
             e = random.choice(list(poly.keys()))
             poly[e] = random.randint(-64, 64)
             if poly[e] == 0: poly[e] = 1
             current_score = count_square_terms(poly)
             if current_score < best_score:
                 best_score = current_score
                 best_poly = dict(poly)
             else:
                 # Revert to best
                 poly = dict(best_poly)
                 current_score = best_score
                 
    # Convert best_poly to list
    max_exp = max(best_poly.keys())
    result = [0] * (max_exp + 1)
    for e, c in best_poly.items():
        result[e] = c
        
    # Ensure constraints: length <= 41, nonzero final coeff, exactly 12 nonzero terms
    # The search ensures 12 terms.
    # Final coeff must be nonzero.
    if result[-1] == 0:
        # Find last nonzero
        while result and result[-1] == 0:
            result.pop()
        if not result:
            result = [1] # Fallback, though invalid
        # Pad if needed? No, length is determined by max exp.
        
    # Check length <= 41
    if len(result) > 41:
        # This shouldn't happen as exponents are < 41
        pass
        
    return result
