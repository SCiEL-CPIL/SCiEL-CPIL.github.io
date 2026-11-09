"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    random.seed(42)
    np.random.seed(42)
    
    best_poly = [1] * 12
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count(coeff_list):
        """Exact square computation using integer convolution."""
        nz = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
        if not nz:
            return 0
        max_deg = max(i for i, _ in nz)
        sq_len = 2 * max_deg + 1
        sq = [0] * sq_len
        for i, ci in nz:
            for j, cj in nz:
                sq[i + j] += ci * cj
        return sum(1 for v in sq if v != 0)
    
    def trim(poly):
        """Trim trailing zeros, ensure length >= 1."""
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def is_valid(poly):
        """Check if polynomial meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 or c != int(c) for c in poly):
            return False
        return True
    
    def collision_score(exp_set):
        """Count number of distinct sum collisions. Higher = more potential cancellation."""
        sums = {}
        n = len(exp_set)
        for i in range(n):
            for j in range(i, n):
                s = exp_set[i] + exp_set[j]
                sums[s] = sums.get(s, 0) + 1
        # Count how many sums appear more than once (collisions)
        collisions = sum(1 for v in sums.values() if v > 1)
        return collisions
    
    def generate_exp_set_with_high_collisions(max_deg=40):
        """Generate an exponent set with high collision density."""
        # Start with a base pattern and perturb
        base = list(range(12))
        # Try different structures
        strategies = [
            # Dense cluster with gaps
            lambda: sorted(random.sample(range(0, max_deg+1), 12)),
            # Arithmetic progression
            lambda: [0] + sorted(random.sample(range(1, max_deg+1), 11)),
            # Two clusters
            lambda: sorted(list(range(0, 6)) + list(range(random.randint(10, 20), random.randint(10, 20)+6))),
            # Three clusters
            lambda: sorted(list(range(0, 4)) + list(range(random.randint(8, 15), random.randint(8, 15)+4)) + list(range(random.randint(25, 35), random.randint(25, 35)+4))),
            # Symmetric
            lambda: sorted([random.randint(0, 10), random.randint(1, 11), random.randint(2, 12), 
                           random.randint(20, 30), random.randint(21, 31), random.randint(22, 32),
                           random.randint(30, 40), random.randint(31, 40), random.randint(32, 40),
                           random.randint(33, 40), random.randint(34, 40), random.randint(35, 40)]),
        ]
        # Pick a random strategy
        strat = random.choice(strategies)
        exp_set = strat()
        # Ensure sorted and unique, within bounds
        exp_set = sorted(set(exp_set))
        if len(exp_set) < 12:
            # Fill in missing
            missing = [x for x in range(0, max_deg+1) if x not in exp_set]
            random.shuffle(missing)
            exp_set = sorted(exp_set + missing[:12-len(exp_set)])
        exp_set = exp_set[:12]
        if max(exp_set) > max_deg:
            exp_set = list(range(12))
        return exp_set
    
    # Phase 1: Generate and evaluate high-collision exponent sets
    evaluated_sets = set()
    num_evaluated = 0
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        exp_set = generate_exp_set_with_high_collisions()
        key = tuple(exp_set)
        if key in evaluated_sets:
            continue
        evaluated_sets.add(key)
        num_evaluated += 1
        
        # Try several sign patterns for this exponent set
        for sign_trial in range(20):
            if time.time() > deadline:
                break
            
            coeffs = [0] * 41
            coeffs[exp_set[0]] = 1
            for i in range(1, 12):
                coeffs[exp_set[i]] = random.choice([1, -1, 2, -2, 3, -3])
            
            cnt = square_nonzero_count(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
    
    # Phase 2: Simulated annealing on the full space
    if time.time() < deadline:
        work_poly = best_poly[:]
        while len(work_poly) < 41:
            work_poly.append(0)
        work_count = best_count
        
        temp = 5.0
        cooling = 0.9995
        
        for iteration in range(20000):
            if time.time() > deadline:
                break
            
            # Pick a random position to modify
            pos = random.randint(0, 40)
            old_val = work_poly[pos]
            
            # Generate new value
            if old_val == 0:
                new_val = random.choice([1, -1, 2, -2, 3, -3, 4, -4])
            else:
                new_val = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
            
            if abs(new_val) > 64:
                continue
            
            work_poly[pos] = new_val
            
            # Check constraints
            nonzero = sum(1 for c in work_poly if c != 0)
            if nonzero != 12:
                work_poly[pos] = old_val
                continue
            
            if work_poly[-1] == 0:
                work_poly[pos] = old_val
                continue
            
            cnt = square_nonzero_count(work_poly)
            delta = cnt - work_count
            
            if delta < 0:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(work_poly[:])
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < np.exp(-delta / max(temp, 0.1)):
                    work_count = cnt
                else:
                    work_poly[pos] = old_val
            
            temp *= cooling
    
    # Phase 3: Targeted search around best solution
    if time.time() < deadline:
        work_poly = best_poly[:]
        while len(work_poly) < 41:
            work_poly.append(0)
        work_count = best_count
        
        # Identify nonzero positions
        nz_positions = [i for i, c in enumerate(work_poly) if c != 0]
        
        # Try modifying each nonzero position's coefficient
        for pos in nz_positions:
            if time.time() > deadline:
                break
            old_val = work_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                work_poly[pos] = new_val
                cnt = square_nonzero_count(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work_poly[:])
                work_poly[pos] = old_val
        
        # Try moving one term to a different position
        for pos in nz_positions[:]:
            if time.time() > deadline:
                break
            old_val = work_poly[pos]
            for new_pos in range(41):
                if new_pos == pos or work_poly[new_pos] != 0:
                    continue
                work_poly[pos] = 0
                work_poly[new_pos] = old_val
                if work_poly[-1] == 0 and new_pos != 40:
                    # Last position must be nonzero
                    work_poly[pos] = old_val
                    work_poly[new_pos] = 0
                    continue
                cnt = square_nonzero_count(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work_poly[:])
                work_poly[pos] = old_val
                work_poly[new_pos] = 0
    
    # Phase 4: Exponent set optimization
    if time.time() < deadline:
        # Extract current exponent set
        nz = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        current_exp = sorted([i for i, _ in nz])
        current_coeffs = [c for _, c in nz]
        
        # Try nearby exponent sets
        for _ in range(500):
            if time.time() > deadline:
                break
            
            new_exp = current_exp[:]
            # Move one exponent
            idx = random.randint(0, 11)
            delta = random.randint(-3, 3)
            new_exp[idx] = max(0, min(40, new_exp[idx] + delta))
            new_exp = sorted(new_exp)
            
            # Check uniqueness
            if len(set(new_exp)) != 12:
                continue
            
            # Keep same coefficients
            coeffs = [0] * 41
            for i, (e, c) in enumerate(zip(new_exp, current_coeffs)):
                coeffs[e] = c
            
            cnt = square_nonzero_count(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
    
    # Final cleanup: ensure valid output
    best_poly = trim(best_poly)
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in best_poly if c != 0)
    if nonzero_count != 12:
        # Fallback to seed
        return [1] * 12
    
    # Ensure entries in [-64, 64]
    for i, c in enumerate(best_poly):
        if c < -64:
            best_poly[i] = -64
        elif c > 64:
            best_poly[i] = 64
        best_poly[i] = int(best_poly[i])
    
    # Final validation
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
