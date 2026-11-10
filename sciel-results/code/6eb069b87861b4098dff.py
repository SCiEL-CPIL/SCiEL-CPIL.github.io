import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_result = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    best_nonzero = 144
    
    def coeffs_from_state(coeffs, exponents):
        max_deg = max(exponents)
        result = [0] * (2 * max_deg + 1)
        for c, e in zip(coeffs, exponents):
            result[e] = c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    def is_valid(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        if 2 * max(exponents) + 1 > 41:
            return False
        return True
    
    def square_nonzero_count(coeffs, exponents):
        n = len(coeffs)
        max_deg = max(exponents)
        poly = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for c, e in zip(coeffs, exponents):
            poly[e] = c
        square = np.convolve(poly, poly)
        return int(np.count_nonzero(square))
    
    def build_degree_map(exponents):
        n = len(exponents)
        dm = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                w = 1 if i == j else 2
                dm[d].append((i, j, w))
        return dict(dm)
    
    def fast_sq_vals(coeffs, exponents, dm):
        """Compute all square degree values"""
        vals = {}
        for d, pairs in dm.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * coeffs[i] * coeffs[j]
            vals[d] = val
        return vals
    
    def count_nz_from_vals(vals):
        return sum(1 for v in vals.values() if v != 0)
    
    def update_best(coeffs, exponents, nz):
        nonlocal best_nonzero, best_result
        if nz < best_nonzero:
            best_nonzero = nz
            result = coeffs_from_state(coeffs, exponents)
            if len(result) <= 41 and result[-1] != 0:
                best_result = result
    
    # Exponent patterns - diverse collision structures
    exponent_patterns = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
        [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17],
        [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20],
        [0, 2, 4, 6, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25],
        [0, 1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18],
        [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
    ]
    
    # SAT-style backtracking solver
    def solve_with_backtracking(exponents, target_nz, max_nodes=50000):
        """
        Try to find coefficients where the square has at most target_nz nonzero terms.
        Uses backtracking with constraint propagation.
        """
        dm = build_degree_map(exponents)
        n = 12
        all_degrees = sorted(dm.keys())
        
        # Degrees with only one pair are forced nonzero
        forced = set()
        cancellable = {}
        for d in all_degrees:
            pairs = dm[d]
            if len(pairs) == 1:
                forced.add(d)
            else:
                cancellable[d] = pairs
        
        if len(forced) > target_nz:
            return None  # Impossible
        
        # We need at most (target_nz - len(forced)) cancellable degrees to be nonzero
        # So at least len(cancellable) - (target_nz - len(forced)) must cancel to zero
        max_cancellable_nz = target_nz - len(forced)
        if max_cancellable_nz < 0:
            return None
        
        # Variable ordering: process coefficients in order
        # Use magnitudes from a small set
        mag_options = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
        
        best_solution = None
        nodes_visited = 0
        
        def backtrack(idx, coeffs, current_vals):
            nonlocal best_solution, nodes_visited
            
            if nodes_visited > max_nodes:
                return False
            
            nodes_visited += 1
            if time.time() > deadline:
                return False
            
            if idx == n:
                # Check final count
                nz = len(forced) + sum(1 for d in cancellable if current_vals.get(d, 0) != 0)
                if nz <= target_nz:
                    best_solution = coeffs[:]
                    return True
                return False
            
            # Try different magnitudes and signs
            for mag in mag_options:
                for sign in [1, -1]:
                    c = sign * mag
                    coeffs[idx] = c
                    
                    # Compute new square values for degrees involving idx
                    new_vals = dict(current_vals)
                    for d, pairs in dm.items():
                        val = 0
                        for (i, j, w) in pairs:
                            ci = coeffs[i] if i == idx else (coeffs[i] if i < idx else 0)
                            cj = coeffs[j] if j == idx else (coeffs[j] if j < idx else 0)
                            if i < idx or i == idx:
                                if j < idx or j == idx:
                                    val += w * ci * cj
                            else:
                                # One index not yet assigned
                                break
                        else:
                            new_vals[d] = val
                    
                    # Pruning: check if already too many nonzeros
                    nz_so_far = len(forced) + sum(1 for d in cancellable if new_vals.get(d, 0) != 0)
                    if nz_so_far > target_nz:
                        continue
                    
                    if backtrack(idx + 1, coeffs, new_vals):
                        return True
            
            coeffs[idx] = 0
            return False
        
        coeffs = [0] * n
        initial_vals = {}
        if backtrack(0, coeffs, initial_vals):
            return best_solution
        return None
    
    # Strategy 1: Quick evaluation of structured patterns
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        dm = build_degree_map(exponents)
        
        # Try all sign patterns with magnitude 1
        for signs in range(256):  # Sample of 2^12
            coeffs = []
            valid = True
            for i in range(12):
                s = 1 if (signs >> i) & 1 else -1
                coeffs.append(s)
            
            if not is_valid(coeffs, exponents):
                continue
            vals = fast_sq_vals(coeffs, exponents, dm)
            nz = count_nz_from_vals(vals)
            update_best(coeffs, exponents, nz)
    
    # Strategy 2: SAT-style search for low target_nz
    for target_nz in range(12, 25):
        if time.time() > deadline:
            break
        
        for exponents in exponent_patterns[:10]:
            if time.time() > deadline:
                break
            
            dm = build_degree_map(exponents)
            forced = set()
            for d in dm:
                if len(dm[d]) == 1:
                    forced.add(d)
            
            if len(forced) > target_nz:
                continue
            
            solution = solve_with_backtracking(exponents, target_nz, max_nodes=20000)
            if solution:
                if is_valid(solution, exponents):
                    vals = fast_sq_vals(solution, exponents, dm)
                    nz = count_nz_from_vals(vals)
                    update_best(solution, exponents, nz)
    
    # Strategy 3: Local search with multiple coordinate changes
    for exponents in exponent_patterns[:15]:
        if time.time() > deadline:
            break
        dm = build_degree_map(exponents)
        
        rng = random.Random(hash(tuple(exponents)) % 100000)
        
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization
            coeffs = []
            for i in range(12):
                mag = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 8])
                sign = 1 if rng.random() < 0.5 else -1
                coeffs.append(sign * mag)
            
            if not is_valid(coeffs, exponents):
                continue
            
            current_vals = fast_sq_vals(coeffs, exponents, dm)
            current_nz = count_nz_from_vals(current_vals)
            
            # Local search: try flipping 1-2 coefficients at a time
            improved = True
            for it in range(60):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                
                best_change = None
                best_nz = current_nz
                
                # Try single flips
                for flip_idx in range(12):
                    old_c = coeffs[flip_idx]
                    coeffs[flip_idx] = -old_c
                    new_vals = fast_sq_vals(coeffs, exponents, dm)
                    nz = count_nz_from_vals(new_vals)
                    coeffs[flip_idx] = old_c
                    
                    if nz < best_nz:
                        best_nz = nz
                        best_change = ('flip', flip_idx, None)
                
                # Try magnitude changes
                for idx in range(12):
                    for new_mag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
                        for sign in [1, -1]:
                            new_c = sign * new_mag
                            if new_c == coeffs[idx]:
                                continue
                            old_c = coeffs[idx]
                            coeffs[idx] = new_c
                            new_vals = fast_sq_vals(coeffs, exponents, dm)
                            nz = count_nz_from_vals(new_vals)
                            coeffs[idx] = old_c
                            
                            if nz < best_nz:
                                best_nz = nz
                                best_change = ('mag', idx, new_c)
                
                if best_change:
                    if best_change[0] == 'flip':
                        coeffs[best_change[1]] = -coeffs[best_change[1]]
                    else:
                        coeffs[best_change[1]] = best_change[2]
                    current_nz = best_nz
                    improved = True
            
            if is_valid(coeffs, exponents):
                vals = fast_sq_vals(coeffs, exponents, dm)
                nz = count_nz_from_vals(vals)
                update_best(coeffs, exponents, nz)
    
    # Strategy 4: Targeted cancellation - force specific degrees to zero
    for exponents in exponent_patterns[:8]:
        if time.time() > deadline:
            break
        dm = build_degree_map(exponents)
        
        # Identify which cancellable degrees to force to zero
        cancellable_degrees = [d for d in dm if len(dm[d]) > 1]
        
        # Try forcing subsets of cancellable degrees to zero
        for num_force in range(0, min(6, len(cancellable_degrees))):
            if time.time() > deadline:
                break
            
            for force_subset in range(50):
                if time.time() > deadline:
                    break
                
                rng = random.Random(hash((tuple(exponents), num_force, force_subset)) % 100000)
                if len(cancellable_degrees) < num_force:
                    continue
                
                forced_to_zero = set(rng.sample(cancellable_degrees, num_force))
                
                # Try to find coefficients that make these degrees zero
                # Use iterative constraint satisfaction
                for trial in range(50):
                    if time.time() > deadline:
                        break
                    
                    coeffs = []
                    for i in range(12):
                        mag = rng.choice([1, 2, 3, 4, 5, 6, 8, 10])
                        sign = 1 if rng.random() < 0.5 else -1
                        coeffs.append(sign * mag)
                    
                    if not is_valid(coeffs, exponents):
                        continue
                    
                    # Iteratively adjust to satisfy constraints
                    for it in range(30):
                        if time.time() > deadline:
                            break
                        
                        # Check which constraints are violated
                        violated = []
                        for d in forced_to_zero:
                            val = 0
                            for (i, j, w) in dm[d]:
                                val += w * coeffs[i] * coeffs[j]
                            if val != 0:
                                violated.append((d, val))
                        
                        if not violated:
                            break
                        
                        # Try to fix one violation
                        d, val = violated[0]
                        fixed = False
                        for (i, j, w) in dm[d]:
                            if i != j:
                                # Try changing coeffs[i]
                                for new_ci in range(-10, 11):
                                    if new_ci == 0 or new_ci == coeffs[i]:
                                        continue
                                    old_ci = coeffs[i]
                                    coeffs[i] = new_ci
                                    
                                    # Check if this fixes the violation without breaking others
                                    new_val = 0
                                    for (a, b, ww) in dm[d]:
                                        new_val += ww * coeffs[a] * coeffs[b]
                                    
                                    if new_val == 0:
                                        # Check other forced constraints
                                        all_ok = True
                                        for d2 in forced_to_zero:
                                            if d2 == d:
                                                continue
                                            v2 = 0
                                            for (a, b, ww) in dm[d2]:
                                                v2 += ww * coeffs[a] * coeffs[b]
                                            if v2 != 0:
                                                all_ok = False
                                                break
                                        
                                        if all_ok:
                                            fixed = True
                                            break
                                        else:
                                            coeffs[i] = old_ci
                                    else:
                                        coeffs[i] = old_ci
                            else:
                                # Diagonal term: c_i^2 * w
                                # Can't easily fix
                                pass
                        
                        if not fixed:
                            break
                    
                    if is_valid(coeffs, exponents):
                        vals = fast_sq_vals(coeffs, exponents, dm)
                        nz = count_nz_from_vals(vals)
                        update_best(coeffs, exponents, nz)
    
    # Strategy 5: Simulated annealing style
    for exponents in exponent_patterns[:10]:
        if time.time() > deadline:
            break
        dm = build_degree_map(exponents)
        
        rng = random.Random(hash(tuple(exponents)) + 777)
        
        # Start from best known or random
        coeffs = []
        for i in range(12):
            mag = rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 8])
            sign = 1 if rng.random() < 0.5 else -1
            coeffs.append(sign * mag)
        
        if not is_valid(coeffs, exponents):
            continue
        
        current_vals = fast_sq_vals(coeffs, exponents, dm)
        current_nz = count_nz_from_vals(current_vals)
        
        temp = 10.0
        for it in range(500):
            if time.time() > deadline:
                break
            
            temp *= 0.97
            
            # Random move
            idx = rng.randint(0, 11)
            old_c = coeffs[idx]
            new_mag = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32])
            new_sign = 1 if rng.random() < 0.5 else -1
            new_c = new_sign * new_mag
            
            if new_c == 0:
                continue
            
            coeffs[idx] = new_c
            if not is_valid(coeffs, exponents):
                coeffs[idx] = old_c
                continue
            
            new_vals = fast_sq_vals(coeffs, exponents, dm)
            new_nz = count_nz_from_vals(new_vals)
            
            delta = new_nz - current_nz
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.1)):
                current_nz = new_nz
                current_vals = new_vals
            else:
                coeffs[idx] = old_c
        
        if is_valid(coeffs, exponents):
            vals = fast_sq_vals(coeffs, exponents, dm)
            nz = count_nz_from_vals(vals)
            update_best(coeffs, exponents, nz)
    
    # Final refinement on best found
    if time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            dm = build_degree_map(best_exponents)
            
            # Intensive local search
            current_coeffs = best_coeffs[:]
            current_vals = fast_sq_vals(current_coeffs, best_exponents, dm)
            current_nz = count_nz_from_vals(current_vals)
            
            for it in range(2000):
                if time.time() > deadline:
                    break
                
                improved = False
                for idx in range(12):
                    for new_c in range(-32, 33):
                        if new_c == 0 or new_c == current_coeffs[idx]:
                            continue
                        old_c = current_coeffs[idx]
                        current_coeffs[idx] = new_c
                        
                        if not is_valid(current_coeffs, best_exponents):
                            current_coeffs[idx] = old_c
                            continue
                        
                        new_vals = fast_sq_vals(current_coeffs, best_exponents, dm)
                        new_nz = count_nz_from_vals(new_vals)
                        
                        if new_nz < current_nz:
                            current_nz = new_nz
                            current_vals = new_vals
                            improved = True
                            break
                        else:
                            current_coeffs[idx] = old_c
                    
                    if improved:
                        break
                
                if not improved:
                    break
            
            if is_valid(current_coeffs, best_exponents):
                update_best(current_coeffs, best_exponents, current_nz)
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    return best_result
