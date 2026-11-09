"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    def square_nonzero_count_from_terms(terms):
        """
        Compute number of nonzero terms in the square of a polynomial
        given as a list of (exponent, coefficient) pairs.
        O(n^2) where n = number of nonzero terms (12 here).
        """
        sq = defaultdict(int)
        n = len(terms)
        for i in range(n):
            ei, ci = terms[i]
            for j in range(i, n):
                ej, cj = terms[j]
                if i == j:
                    sq[2 * ei] += ci * ci
                else:
                    sq[ei + ej] += 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    def terms_to_poly(terms):
        """Convert list of (exponent, coefficient) to coefficient list."""
        if not terms:
            return [1]
        max_deg = max(e for e, c in terms)
        poly = [0] * (max_deg + 1)
        for e, c in terms:
            poly[e] = c
        return poly
    
    def is_valid_poly(poly):
        """Check if polynomial meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 for c in poly):
            return False
        return True
    
    # Simulated annealing over the 12-term support
    # State: list of 12 (exponent, coefficient) pairs
    # Moves: change one exponent, change one coefficient, swap two exponents
    
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    best_terms = [(i, 1) for i in range(12)]
    
    def valid_state(terms):
        """Check if a state is valid: 12 distinct exponents in [0,40], coeffs in [-64,64], nonzero."""
        if len(terms) != 12:
            return False
        exps = [e for e, c in terms]
        if len(set(exps)) != 12:
            return False
        if any(e < 0 or e > 40 for e in exps):
            return False
        if any(c == 0 or c < -64 or c > 64 for e, c in terms):
            return False
        return True
    
    def random_state(rng):
        """Generate a random valid state."""
        exps = sorted(rng.sample(range(41), 12))
        coeffs = [rng.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
        return list(zip(exps, coeffs))
    
    def make_move(terms, rng):
        """Make a random move on the state. Returns new terms or None if invalid."""
        new_terms = [(e, c) for e, c in terms]
        move_type = rng.random()
        
        if move_type < 0.35:
            # Change one exponent
            idx = rng.randint(0, 11)
            old_e = new_terms[idx][0]
            new_e = rng.randint(0, 40)
            if new_e == old_e:
                return None
            # Check no collision
            exps = set(e for e, c in new_terms)
            exps.discard(old_e)
            if new_e in exps:
                return None
            new_terms[idx] = (new_e, new_terms[idx][1])
            
        elif move_type < 0.60:
            # Change one coefficient
            idx = rng.randint(0, 11)
            old_c = new_terms[idx][1]
            new_c = rng.choice([-4,-3,-2,-1,1,2,3,4])
            if new_c == old_c or new_c == 0:
                return None
            new_terms[idx] = (new_terms[idx][0], new_c)
            
        elif move_type < 0.80:
            # Swap two exponents
            i, j = rng.sample(range(12), 2)
            new_terms[i], new_terms[j] = new_terms[j], new_terms[i]
            # Actually swap the exponents only
            ei, ci = new_terms[i]
            ej, cj = new_terms[j]
            new_terms[i] = (ej, ci)
            new_terms[j] = (ei, cj)
            
        else:
            # Change two exponents simultaneously (helps escape local optima)
            i, j = rng.sample(range(12), 2)
            old_ei = new_terms[i][0]
            old_ej = new_terms[j][0]
            new_ei = rng.randint(0, 40)
            new_ej = rng.randint(0, 40)
            if new_ei == old_ei and new_ej == old_ej:
                return None
            exps = set(e for e, c in new_terms)
            exps.discard(old_ei)
            exps.discard(old_ej)
            if new_ei in exps or new_ej in exps:
                return None
            if new_ei == new_ej:
                return None
            new_terms[i] = (new_ei, new_terms[i][1])
            new_terms[j] = (new_ej, new_terms[j][1])
        
        if not valid_state(new_terms):
            return None
        return new_terms
    
    # Phase 1: Quick systematic search over structured exponent sets with ±1
    exponent_families = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15],
        [0, 1, 2, 3, 5, 8, 13, 21, 28, 35, 38, 40],
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        [0, 1, 2, 4, 8, 16, 24, 28, 32, 36, 38, 40],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27],
    ]
    
    valid_families = [f for f in exponent_families if max(f) <= 40]
    
    # Try ±1 sign patterns for each family
    for exp_set in valid_families:
        if time.time() > deadline:
            break
        # Try all 2^11 sign patterns
        for mask in range(1024):
            if time.time() > deadline:
                break
            terms = [(exp_set[0], 1)]
            for i in range(1, 12):
                sign = 1 if (mask >> i) & 1 else -1
                terms.append((exp_set[i], sign))
            cnt = square_nonzero_count_from_terms(terms)
            if cnt < best_count:
                best_count = cnt
                best_terms = terms[:]
    
    # Phase 2: Simulated annealing from multiple starting points
    rng = random.Random(42)
    
    # Starting points: best found + several random
    start_states = [best_terms[:]]
    for _ in range(8):
        start_states.append(random_state(rng))
    
    for start in start_states:
        if time.time() > deadline:
            break
        
        current = start[:]
        current_count = square_nonzero_count_from_terms(current)
        
        # Simulated annealing parameters
        T = 2.0  # initial temperature (in terms of count difference)
        T_min = 0.1
        cooling = 0.995
        steps = 3000
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            new_terms = make_move(current, rng)
            if new_terms is None:
                continue
            
            new_count = square_nonzero_count_from_terms(new_terms)
            delta = new_count - current_count
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                current = new_terms
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_terms = current[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Phase 3: Focused local search on best solution found
    if time.time() < deadline:
        current = best_terms[:]
        current_count = best_count
        rng2 = random.Random(99)
        
        # More aggressive local search with steepest descent
        no_improve = 0
        max_no_improve = 500
        
        while no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            best_neighbor = None
            best_neighbor_count = current_count
            
            # Try all single-exponent changes
            for idx in range(12):
                old_e = current[idx][0]
                for new_e in range(40):
                    if new_e == old_e:
                        continue
                    exps = set(e for e, c in current)
                    exps.discard(old_e)
                    if new_e in exps:
                        continue
                    trial = current[:]
                    trial[idx] = (new_e, current[idx][1])
                    cnt = square_nonzero_count_from_terms(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
            
            # Try all single-coefficient changes
            for idx in range(12):
                old_c = current[idx][1]
                for new_c in [-4,-3,-2,-1,1,2,3,4]:
                    if new_c == old_c or new_c == 0:
                        continue
                    trial = current[:]
                    trial[idx] = (current[idx][0], new_c)
                    cnt = square_nonzero_count_from_terms(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
            
            if best_neighbor is not None:
                current = best_neighbor
                current_count = best_neighbor_count
                no_improve = 0
                if current_count < best_count:
                    best_count = current_count
                    best_terms = current[:]
            else:
                # Try random perturbation to escape local optimum
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    perturbed = make_move(current, rng2)
                    if perturbed is None:
                        continue
                    pc = square_nonzero_count_from_terms(perturbed)
                    if pc < current_count:
                        current = perturbed
                        current_count = pc
                        no_improve = 0
                        if current_count < best_count:
                            best_count = current_count
                            best_terms = current[:]
                        break
                else:
                    no_improve += 1
    
    # Phase 4: More random restarts if time remains
    if time.time() < deadline:
        rng3 = random.Random(777)
        for restart in range(20):
            if time.time() > deadline:
                break
            
            start = random_state(rng3)
            current = start[:]
            current_count = square_nonzero_count_from_terms(current)
            
            # Quick hill climbing
            T = 1.0
            cooling = 0.99
            for step in range(1000):
                if time.time() > deadline:
                    break
                new_terms = make_move(current, rng3)
                if new_terms is None:
                    continue
                new_count = square_nonzero_count_from_terms(new_terms)
                delta = new_count - current_count
                if delta < 0 or rng3.random() < np.exp(-delta / max(T, 0.01)):
                    current = new_terms
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_terms = current[:]
                T *= cooling
                if T < 0.1:
                    T = 0.1
    
    # Convert best_terms to polynomial
    best_poly = terms_to_poly(best_terms)
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in best_poly if c != 0)
    if nonzero_count != 12:
        return [1] * 12
    
    # Ensure entries in [-64, 64] and are integers
    for i, c in enumerate(best_poly):
        best_poly[i] = int(c)
        if best_poly[i] < -64:
            best_poly[i] = -64
        elif best_poly[i] > 64:
            best_poly[i] = 64
    
    # Final validation
    if not is_valid_poly(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
