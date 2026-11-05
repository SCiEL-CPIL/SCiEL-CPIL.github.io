import random
import time
import math

def _count_nonzero_square_fast(coeffs):
    """Compute the square of a polynomial and count nonzero terms."""
    n = len(coeffs)
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
    return sum(1 for c in sq if c != 0)

def _square_nonzero_count(coeffs):
    """Same as above but also returns the squared coefficients for further analysis."""
    n = len(coeffs)
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
    return sum(1 for c in sq if c != 0), sq

def _pairwise_cancellation_score(exponents, signs):
    """
    For a given set of exponents and signs (±1), estimate how many cancellations
    occur in the square. Two terms c_i*x^i and c_j*x^j contribute to degree i+j.
    If multiple pairs (i,j) and (k,l) have i+j = k+l, their contributions may cancel.
    """
    n = len(exponents)
    # Map degree -> list of (sign_product, count)
    degree_map = {}
    for i in range(n):
        for j in range(n):
            deg = exponents[i] + exponents[j]
            sign = signs[i] * signs[j]
            if deg not in degree_map:
                degree_map[deg] = 0
            degree_map[deg] += sign
    nonzero = sum(1 for v in degree_map.values() if v != 0)
    return nonzero

def _generate_structured_exponents():
    """Generate candidate exponent sets with structure that promotes collisions in the square."""
    candidates = []
    
    # Type 1: Arithmetic progression blocks
    # P = sum of terms at positions that form symmetric patterns
    for base in range(0, 30):
        for step in range(1, 8):
            for block_size in range(4, 13):
                exponents = []
                for k in range(block_size):
                    exp = base + k * step
                    if exp >= 41:
                        break
                    exponents.append(exp)
                if len(exponents) == 12:
                    candidates.append(sorted(exponents))
    
    # Type 2: Symmetric around center
    for center in range(6, 35):
        for spread in range(1, 20):
            exponents = set()
            for k in range(-6, 7):
                exp = center + k * spread
                if 0 <= exp < 41:
                    exponents.add(exp)
            if len(exponents) == 12:
                candidates.append(sorted(exponents))
    
    # Type 3: Product structure (1+x^a+x^b+x^c)(1+x^d+x^e)
    for a in range(1, 15):
        for b in range(a+1, 15):
            for c in range(b+1, 15):
                for d in range(1, 15):
                    if d in (a, b, c):
                        continue
                    for e in range(1, 15):
                        if e in (a, b, c, d):
                            continue
                        exponents = set()
                        for x1 in [0, a, b, c]:
                            for x2 in [0, d, e]:
                                exp = x1 + x2
                                if exp < 41:
                                    exponents.add(exp)
                        if len(exponents) == 12:
                            candidates.append(sorted(exponents))
    
    # Type 4: Random but with small gaps (promotes collisions)
    for _ in range(2000):
        exponents = []
        current = random.randint(0, 5)
        while len(exponents) < 12 and current < 40:
            exponents.append(current)
            current += random.choice([1, 1, 2, 2, 3, 4])
        if len(exponents) == 12 and exponents[-1] < 41:
            candidates.append(sorted(exponents))
    
    return candidates

def _optimize_signs_for_exponents(exponents, max_iter=2000):
    """
    Given fixed exponents, find sign assignment (±1) that minimizes nonzero terms in square.
    Uses simulated annealing / hill climbing on the sign vector.
    """
    n = len(exponents)
    best_signs = [1] * n
    best_score = _pairwise_cancellation_score(exponents, best_signs)
    
    current_signs = [1] * n
    current_score = best_score
    
    temperature = 1.0
    cooling = 0.999
    
    for iteration in range(max_iter):
        # Pick a random index to flip
        idx = random.randint(0, n - 1)
        old_sign = current_signs[idx]
        current_signs[idx] = -old_sign
        
        new_score = _pairwise_cancellation_score(exponents, current_signs)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs[:]
        else:
            current_signs[idx] = old_sign
        
        temperature *= cooling
    
    return best_signs, best_score

def _genetic_search(exponents_pool, pop_size=50, generations=100, elite_size=5):
    """
    Genetic algorithm over sign assignments for a pool of exponent sets.
    Each individual: (exponents_index, signs_vector)
    """
    random.seed(42)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        exp_idx = random.randint(0, len(exponents_pool) - 1)
        exponents = exponents_pool[exp_idx]
        signs = [random.choice([-1, 1]) for _ in range(12)]
        score = _pairwise_cancellation_score(exponents, signs)
        population.append((exp_idx, signs, score))
    
    population.sort(key=lambda x: x[2])
    best_overall = population[0]
    
    for gen in range(generations):
        # Selection: keep elites
        elites = population[:elite_size]
        
        # Generate new population
        new_pop = list(elites)
        
        while len(new_pop) < pop_size:
            # Crossover
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # With some probability, also change the exponent set
            if random.random() < 0.2:
                child_exp_idx = random.randint(0, len(exponents_pool) - 1)
            else:
                child_exp_idx = parent1[0]
            
            # Crossover signs (bitwise)
            child_signs = []
            for i in range(12):
                if random.random() < 0.5:
                    child_signs.append(parent1[1][i])
                else:
                    child_signs.append(parent2[1][i])
            
            # Mutation
            for i in range(12):
                if random.random() < 0.1:
                    child_signs[i] = -child_signs[i]
            
            exponents = exponents_pool[child_exp_idx]
            score = _pairwise_cancellation_score(exponents, child_signs)
            new_pop.append((child_exp_idx, child_signs, score))
        
        population = new_pop
        population.sort(key=lambda x: x[2])
        
        if population[0][2] < best_overall[2]:
            best_overall = population[0]
        
        if best_overall[2] <= 5:
            break
    
    return best_overall

def _hill_climb_from_start(exponents, initial_signs, max_steps=5000):
    """Local search: flip individual signs to improve score."""
    n = len(exponents)
    current_signs = initial_signs[:]
    current_score = _pairwise_cancellation_score(exponents, current_signs)
    best_signs = current_signs[:]
    best_score = current_score
    
    for step in range(max_steps):
        improved = False
        for idx in range(n):
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            new_score = _pairwise_cancellation_score(exponents, current_signs)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                current_signs[idx] = old_sign
        
        if not improved:
            break
    
    return best_signs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    best_count = 999
    best_poly = None
    best_exponents = None
    best_signs = None
    
    random.seed(42)
    
    # Phase 1: Generate structured exponent candidates
    exponent_pool = _generate_structured_exponents()
    
    # Remove duplicates
    unique_exponents = []
    seen = set()
    for exps in exponent_pool:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_exponents.append(exps)
    
    # Ensure we have a reasonable pool
    if len(unique_exponents) < 100:
        # Add more random ones
        for _ in range(500):
            exponents = sorted(random.sample(range(0, 41), 12))
            key = tuple(exponents)
            if key not in seen:
                seen.add(key)
                unique_exponents.append(exponents)
    
    # Phase 2: For each exponent set, find optimal signs via hill climbing
    # But first, do a quick screening with random signs
    screened = []
    for exps in unique_exponents:
        # Try a few random sign assignments
        best_local_score = 999
        for _ in range(20):
            signs = [random.choice([-1, 1]) for _ in range(12)]
            score = _pairwise_cancellation_score(exps, signs)
            if score < best_local_score:
                best_local_score = score
        screened.append((best_local_score, exps))
    
    screened.sort(key=lambda x: x[0])
    
    # Phase 3: For the top candidates, do thorough sign optimization
    top_candidates = screened[:200]
    
    for score_est, exps in top_candidates:
        if time.time() > deadline:
            break
        
        # Try multiple starting points for hill climbing
        best_signs_local = None
        best_score_local = 999
        
        for restart in range(5):
            initial_signs = [random.choice([-1, 1]) for _ in range(12)]
            signs, score = _hill_climb_from_start(exps, initial_signs, max_steps=2000)
            if score < best_score_local:
                best_score_local = score
                best_signs_local = signs
        
        # Also try all-ones and alternating patterns
        for pattern in [[1]*12, [-1]*12, [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]]:
            signs, score = _hill_climb_from_start(exps, pattern[:], max_steps=2000)
            if score < best_score_local:
                best_score_local = score
                best_signs_local = signs
        
        if best_signs_local is not None and best_score_local < best_count:
            # Verify with actual computation
            coeffs = [0] * (exps[-1] + 1)
            for i, exp in enumerate(exps):
                coeffs[exp] = best_signs_local[i]
            
            actual_count, _ = _square_nonzero_count(coeffs)
            if actual_count < best_count:
                best_count = actual_count
                best_poly = coeffs[:]
                best_exponents = exps[:]
                best_signs = best_signs_local[:]
                
                if best_count <= 5:
                    return best_poly
    
    # Phase 4: Genetic algorithm over the exponent pool
    if time.time() < deadline - 20:
        ga_result = _genetic_search(unique_exponents[:100], pop_size=40, generations=50, elite_size=4)
        if ga_result is not None:
            exp_idx, signs, score_est = ga_result
            exps = unique_exponents[exp_idx]
            
            # Refine with hill climbing
            signs, score = _hill_climb_from_start(exps, signs[:], max_steps=3000)
            
            coeffs = [0] * (exps[-1] + 1)
            for i, exp in enumerate(exps):
                coeffs[exp] = signs[i]
            
            actual_count, _ = _square_nonzero_count(coeffs)
            if actual_count < best_count:
                best_count = actual_count
                best_poly = coeffs[:]
                best_exponents = exps[:]
                best_signs = signs[:]
    
    # Phase 5: If still not great, do targeted random search with small exponent gaps
    if time.time() < deadline - 5 and best_count > 8:
        for trial in range(50000):
            if time.time() > deadline:
                break
            
            # Generate exponents with small gaps to promote collisions
            exponents = []
            current = random.randint(0, 3)
            while len(exponents) < 12 and current < 40:
                exponents.append(current)
                gap = random.choice([1, 1, 1, 2, 2, 3])
                current += gap
            
            if len(exponents) != 12 or exponents[-1] >= 41:
                continue
            
            exponents = sorted(exponents)
            
            # Quick sign search
            best_local_score = 999
            best_local_signs = None
            for _ in range(10):
                signs = [random.choice([-1, 1]) for _ in range(12)]
                score = _pairwise_cancellation_score(exponents, signs)
                if score < best_local_score:
                    best_local_score = score
                    best_local_signs = signs
            
            if best_local_signs is None:
                continue
            
            # Refine
            signs, score = _hill_climb_from_start(exponents, best_local_signs[:], max_steps=1000)
            
            coeffs = [0] * (exponents[-1] + 1)
            for i, exp in enumerate(exponents):
                coeffs[exp] = signs[i]
            
            actual_count, _ = _square_nonzero_count(coeffs)
            if actual_count < best_count:
                best_count = actual_count
                best_poly = coeffs[:]
                best_exponents = exponents[:]
                best_signs = signs[:]
                
                if best_count <= 5:
                    break
    
    if best_poly is None:
        # Fallback: simple 12-term polynomial
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # Trim trailing zeros (but keep last coefficient nonzero)
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
    
    return best_poly
