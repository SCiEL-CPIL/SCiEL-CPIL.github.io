import time
import random
import numpy as np
from itertools import combinations

T_START = time.time()
DEADLINE = 80.0  # Safety margin before 90s kill

def _square_nonzero_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeff list."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Use numpy convolution for speed
    arr = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _valid_candidate(coeffs):
    """Check all constraints on the candidate list."""
    if not isinstance(coeffs, list):
        return False
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    for v in coeffs:
        if v < -64 or v > 64:
            return False
    nnz = sum(1 for v in coeffs if v != 0)
    if nnz != 12:
        return False
    return True

def _try_polynomial(coeffs):
    """If valid, return (count, coeffs); else (None, None)."""
    if not _valid_candidate(coeffs):
        return None, None
    return _square_nonzero_count(coeffs), coeffs

def construct():
    best_count = 100
    best_coeffs = None
    end_time = T_START + DEADLINE

    def expired():
        return time.time() > end_time

    def build_poly(positions, values):
        coeffs = [0] * 41
        for p, v in zip(positions, values):
            if 0 <= p < 41:
                coeffs[p] = v
            else:
                return None
        k = 40
        while k >= 0 and coeffs[k] == 0:
            k -= 1
        if k < 0:
            return None
        return coeffs[:k+1]

    def update_best(coeffs):
        nonlocal best_count, best_coeffs
        if coeffs is None:
            return
        c, _ = _try_polynomial(coeffs)
        if c is not None and c < best_count:
            best_count = c
            best_coeffs = list(coeffs)

    # --- Phase 1: Additive combinatorics guided initialization ---
    # We want positions where the sumset {i+j : i,j in S} is small
    # and where coefficient choices can create cancellations
    
    # Strategy: try all subsets of a small structured universe
    # For 12 positions in [0, 40], the sumset has at most 81 elements
    # Minimum possible distinct sums for 12 elements: if they form an arithmetic
    # progression, sumset size is 2*12-1 = 23. With clever coefficients,
    # we can get cancellations.
    
    # Better: use positions that form a "difference set" structure
    # A (v, k, lambda) difference set in Z_v has the property that every
    # nonzero element is a difference of exactly lambda pairs.
    
    # For our purpose, we want the sumset to be small AND the coefficients
    # to create cancellations in the square.
    
    # Key insight: if P(x) = sum a_i x^{e_i}, then P(x)^2 = sum a_i^2 x^{2e_i} + 2*sum_{i<j} a_i*a_j x^{e_i+e_j}
    # Cancellations happen when multiple (i,j) pairs give the same e_i+e_j
    # and the coefficients sum to zero.
    
    # Let's try structured position sets with high additive energy
    
    position_candidates = []
    
    # Arithmetic progressions
    for gap in range(1, 8):
        for start in range(0, 41 - 11*gap + 1):
            pos = [start + i*gap for i in range(12)]
            position_candidates.append(pos)
    
    # Two-block structures: [a, a+1, ..., a+5] and [b, b+1, ..., b+5]
    for a in range(0, 20):
        for b in range(a+6, 41):
            pos = list(range(a, a+6)) + list(range(b, b+6))
            if max(pos) < 41:
                position_candidates.append(pos)
    
    # Symmetric positions
    for center in range(10, 31):
        for spread in range(1, 20):
            pos = []
            for k in range(1, 20):
                if center - k >= 0:
                    pos.append(center - k)
                if center + k < 41:
                    pos.append(center + k)
                if len(pos) >= 12:
                    break
            if len(pos) == 12:
                pos.sort()
                position_candidates.append(pos)
    
    # Random positions biased toward clusters
    random.seed(12345)
    for _ in range(200):
        # Generate 12 positions with some clustering
        base = random.randint(0, 30)
        pos = set()
        for _ in range(12):
            p = base + random.randint(-5, 5)
            if 0 <= p < 41:
                pos.add(p)
            else:
                pos.add(random.randint(0, 40))
        if len(pos) == 12:
            position_candidates.append(sorted(pos))
    
    # Deduplicate
    seen_pos = set()
    unique_pos = []
    for ps in position_candidates:
        key = tuple(ps)
        if key not in seen_pos:
            seen_pos.add(key)
            unique_pos.append(ps)
    position_candidates = unique_pos
    
    # For each position set, try to find good coefficients
    # Use a targeted search: try all sign patterns first, then local search
    
    sign_patterns = []
    # All +1
    sign_patterns.append([1]*12)
    # Alternating
    sign_patterns.append([1 if i%2==0 else -1 for i in range(12)])
    # Half +, half -
    sign_patterns.append([1]*6 + [-1]*6)
    # Various structured patterns
    for mask in range(0, 4096, 32):  # Sample 128 sign patterns
        sign_patterns.append([1 if (mask >> i) & 1 else -1 for i in range(12)])
    
    # Evaluate all position sets with sampled sign patterns
    for ps in position_candidates:
        if expired():
            break
        for signs in sign_patterns:
            if expired():
                break
            update_best(build_poly(ps, signs))
    
    # --- Phase 2: Genetic Algorithm ---
    # Population of (positions, values) pairs
    
    def encode(pos, vals):
        """Encode as a flat tuple for the GA."""
        return tuple(pos) + tuple(vals)
    
    def decode(ind):
        """Decode flat tuple to positions and values."""
        return list(ind[:12]), list(ind[12:24])
    
    def fitness(ind):
        """Lower is better. Return inf if invalid."""
        pos, vals = decode(ind)
        if len(set(pos)) != 12:
            return float('inf')
        for p in pos:
            if p < 0 or p >= 41:
                return float('inf')
        for v in vals:
            if v < -64 or v > 64 or v == 0:
                return float('inf')
        coeffs = build_poly(pos, vals)
        if coeffs is None:
            return float('inf')
        c, _ = _try_polynomial(coeffs)
        if c is None:
            return float('inf')
        return c
    
    def crossover(ind1, ind2):
        """Uniform crossover."""
        child = []
        for i in range(24):
            if random.random() < 0.5:
                child.append(ind1[i])
            else:
                child.append(ind2[i])
        return tuple(child)
    
    def mutate(ind):
        """Mutation: change one or two positions/values."""
        child = list(ind)
        n_mut = random.randint(1, 3)
        for _ in range(n_mut):
            gene = random.randint(0, 23)
            if gene < 12:
                # Position gene
                child[gene] = random.randint(0, 40)
            else:
                # Value gene
                child[gene] = random.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
        return tuple(child)
    
    # Initialize population
    pop_size = 200
    population = []
    
    # Add best found so far
    if best_coeffs:
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        population.append(encode(pos, vals))
    
    # Add diverse random individuals
    for _ in range(pop_size - 1):
        pos = random.sample(range(41), 12)
        pos.sort()
        vals = [random.choice([-1, 1, -2, 2, -3, 3]) for _ in range(12)]
        population.append(encode(pos, vals))
    
    # Add structured individuals
    for ps in position_candidates[:50]:
        vals = [1]*12
        population.append(encode(ps, vals))
        if len(population) >= pop_size:
            break
    
    # GA loop
    gen = 0
    max_gens = 500
    no_improve_gens = 0
    
    while gen < max_gens and not expired():
        gen += 1
        
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]
        
        # Track best
        min_fit = min(fitnesses)
        best_idx = fitnesses.index(min_fit)
        
        if min_fit < best_count:
            best_count = min_fit
            pos, vals = decode(population[best_idx])
            best_coeffs = build_poly(pos, vals)
            no_improve_gens = 0
        else:
            no_improve_gens += 1
        
        if no_improve_gens > 50:
            break
        
        # Selection: tournament selection
        def select():
            candidates = random.sample(range(len(population)), 5)
            best_c = min(candidates, key=lambda i: fitnesses[i])
            return population[best_c]
        
        # Create new population
        new_pop = []
        
        # Elitism: keep top 10
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i])
        for i in sorted_indices[:10]:
            new_pop.append(population[i])
        
        # Fill rest with crossover + mutation
        while len(new_pop) < pop_size:
            if expired():
                break
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            if random.random() < 0.7:
                child = mutate(child)
            new_pop.append(child)
        
        population = new_pop
    
    # --- Phase 3: Local search refinement on best ---
    if best_coeffs and not expired():
        def local_search(start_coeffs, max_steps=2000):
            nonlocal best_count, best_coeffs
            if start_coeffs is None:
                return
            
            arr = [0] * 41
            for i, v in enumerate(start_coeffs):
                if i < 41:
                    arr[i] = v
                    
            curr_arr = list(arr)
            
            def get_trimmed(a):
                k = 40
                while k >= 0 and a[k] == 0:
                    k -= 1
                if k < 0: return None
                return a[:k+1]
                
            curr_trimmed = get_trimmed(curr_arr)
            curr_count, _ = _try_polynomial(curr_trimmed)
            if curr_count is None:
                return
            
            no_improve = 0
            max_no_improve = 300
            
            for step in range(max_steps):
                if expired():
                    return
                if no_improve > max_no_improve:
                    break
                
                new_arr = list(curr_arr)
                move_type = random.randint(0, 4)
                
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz:
                    continue
                
                if move_type == 0:
                    # Flip sign
                    idx = random.choice(nz)
                    new_arr[idx] = -new_arr[idx]
                elif move_type == 1:
                    # Change coefficient
                    idx = random.choice(nz)
                    choices = [v for v in range(-64, 65) if v != 0 and v != new_arr[idx]]
                    new_arr[idx] = random.choice(choices)
                elif move_type == 2:
                    # Shift position
                    idx = random.choice(nz)
                    shift = random.choice([-2, -1, 1, 2])
                    new_idx = idx + shift
                    if 0 <= new_idx < 41 and new_arr[new_idx] == 0:
                        new_arr[new_idx] = new_arr[idx]
                        new_arr[idx] = 0
                    else:
                        continue
                elif move_type == 3:
                    # Swap values at two positions
                    if len(nz) < 2:
                        continue
                    idx1, idx2 = random.sample(nz, 2)
                    new_arr[idx1], new_arr[idx2] = new_arr[idx2], new_arr[idx1]
                else:
                    # Move to random empty position
                    idx = random.choice(nz)
                    empty = [i for i in range(41) if new_arr[i] == 0]
                    if not empty:
                        continue
                    new_idx = random.choice(empty)
                    new_arr[new_idx] = new_arr[idx]
                    new_arr[idx] = 0
                
                new_trimmed = get_trimmed(new_arr)
                if new_trimmed is None:
                    continue
                c, _ = _try_polynomial(new_trimmed)
                if c is not None:
                    if c < curr_count:
                        curr_count = c
                        curr_arr = list(new_arr)
                        no_improve = 0
                        if c < best_count:
                            best_count = c
                            best_coeffs = list(new_trimmed)
                    else:
                        no_improve += 1
        
        local_search(best_coeffs)
    
    # --- Phase 4: Exhaustive sign search on best positions ---
    if best_coeffs and not expired():
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        vals = [v for v in best_coeffs if v != 0]
        if len(pos) == 12:
            # Try all 2^12 sign combinations with magnitudes from original
            mags = [abs(v) for v in vals]
            for mask in range(4096):
                if expired():
                    break
                signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                new_vals = [mags[i] * signs[i] for i in range(12)]
                update_best(build_poly(pos, new_vals))
    
    if best_coeffs is None:
        best_coeffs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # Trim
        k = len(best_coeffs) - 1
        while k >= 0 and best_coeffs[k] == 0:
            k -= 1
        best_coeffs = best_coeffs[:k+1]

    return best_coeffs
