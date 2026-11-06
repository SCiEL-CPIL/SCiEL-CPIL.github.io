import time
import random
import numpy as np

def _square_nonzero_count(poly):
    """
    Given poly as a list of (exponent, coeff) pairs,
    return the number of nonzero terms in poly^2.
    """
    if not poly:
        return 0
    max_deg = max(exp for exp, _ in poly)
    sq_deg = 2 * max_deg
    vec = np.zeros(sq_deg + 1, dtype=np.int64)
    for exp, coeff in poly:
        vec[exp] = coeff
    result = np.convolve(vec, vec)
    return int(np.count_nonzero(result))

def _square_nonzero_count_fast(exps, coeffs):
    """
    Faster evaluation using sparse representation.
    exps: sorted list of exponents
    coeffs: list of coefficients
    """
    n = len(exps)
    if n == 0:
        return 0
    
    sq_terms = {}
    for i in range(n):
        for j in range(i, n):
            deg = exps[i] + exps[j]
            if i == j:
                coeff = coeffs[i] * coeffs[i]
            else:
                coeff = 2 * coeffs[i] * coeffs[j]
            if coeff != 0:
                sq_terms[deg] = sq_terms.get(deg, 0) + coeff
    
    return sum(1 for v in sq_terms.values() if v != 0)

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    if any(c == 0 for _, c in poly):
        return False
    if any(c < -64 or c > 64 for _, c in poly):
        return False
    if any(e < 0 or not isinstance(e, int) for e, _ in poly):
        return False
    if any(e > 40 for e, _ in poly):
        return False
    exps = [e for e, _ in poly]
    if len(set(exps)) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 82
    
    best_poly = None
    best_score = float('inf')
    
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
        
    random.seed(42)
    np.random.seed(42)
    
    def _evaluate(exps, coeffs):
        return _square_nonzero_count_fast(exps, coeffs)

    def _random_individual():
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1, -2, 2, -3, 3]) for _ in range(12)]
        return exps, coeffs

    def _repair(exps, coeffs):
        # Ensure unique exponents
        # If duplicates, replace one with a random new one
        seen = set()
        new_exps = []
        new_coeffs = []
        for e, c in zip(exps, coeffs):
            if e in seen:
                # Find a new exponent
                while True:
                    ne = random.randint(0, 40)
                    if ne not in seen and ne not in new_exps:
                        e = ne
                        break
            seen.add(e)
            new_exps.append(e)
            new_coeffs.append(c)
            
        # Sort
        pairs = sorted(zip(new_exps, new_coeffs))
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def _crossover(p1, p2):
        e1, c1 = p1
        e2, c2 = p2
        child_exps = []
        child_coeffs = []
        for i in range(12):
            if random.random() < 0.5:
                child_exps.append(e1[i])
                child_coeffs.append(c1[i])
            else:
                child_exps.append(e2[i])
                child_coeffs.append(c2[i])
        return _repair(child_exps, child_coeffs)

    def _mutate(ind, rate=0.3):
        exps, coeffs = ind
        if random.random() < rate:
            i = random.randint(0, 11)
            new_e = random.randint(0, 40)
            exps[i] = new_e
            exps, coeffs = _repair(exps, coeffs)
        if random.random() < rate:
            i = random.randint(0, 11)
            coeffs[i] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5])
        return exps, coeffs

    def _ga():
        pop_size = 200
        generations = 500
        elite_size = 10
        
        population = [_random_individual() for _ in range(pop_size)]
        population[0] = ([e for e, _ in seed_poly], [c for _, c in seed_poly])
        
        best_exps, best_coeffs = [e for e, _ in seed_poly], [c for _, c in seed_poly]
        best_score = _evaluate(best_exps, best_coeffs)
        
        for gen in range(generations):
            if time.time() > deadline:
                break
                
            scored = []
            for ind in population:
                s = _evaluate(ind[0], ind[1])
                scored.append((s, ind[0], ind[1]))
                
            scored.sort(key=lambda x: x[0])
            
            if scored[0][0] < best_score:
                best_score = scored[0][0]
                best_exps = scored[0][1]
                best_coeffs = scored[0][2]
                
            if best_score <= 12:
                break
                
            new_pop = []
            for i in range(elite_size):
                new_pop.append((scored[i][1], scored[i][2]))
                
            while len(new_pop) < pop_size:
                # Tournament
                p1 = random.choice(scored[:50])
                p2 = random.choice(scored[:50])
                child = _crossover((p1[1], p1[2]), (p2[1], p2[2]))
                child = _mutate(child)
                new_pop.append(child)
                
            population = new_pop
            
        return best_exps, best_coeffs, best_score

    best_exps, best_coeffs, ga_score = _ga()
    
    if ga_score < best_score:
        best_score = ga_score
        best_poly = list(zip(best_exps, best_coeffs))
        
    # Local search
    if best_poly is not None and time.time() < deadline:
        cur_exps = [e for e, _ in best_poly]
        cur_coeffs = [c for _, c in best_poly]
        cur_score = _evaluate(cur_exps, cur_coeffs)
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            improved = False
            # Try all single exponent moves
            for i in range(12):
                old_e = cur_exps[i]
                # Try a few random new exponents
                for new_e in random.sample([e for e in range(41) if e != old_e], 10):
                    if new_e in cur_exps:
                        continue
                    t_exps = cur_exps.copy()
                    t_exps[i] = new_e
                    t_exps.sort()
                    # Realign coeffs
                    # This is tricky with simple sort. 
                    # Better to use pairs.
                    pairs = list(zip(cur_exps, cur_coeffs))
                    pairs[i] = (new_e, pairs[i][1])
                    pairs.sort(key=lambda x: x[0])
                    t_exps = [p[0] for p in pairs]
                    t_coeffs = [p[1] for p in pairs]
                    t_score = _evaluate(t_exps, t_coeffs)
                    if t_score < cur_score:
                        cur_exps = t_exps
                        cur_coeffs = t_coeffs
                        cur_score = t_score
                        improved = True
                        break
                if improved:
                    break
                    
            if not improved:
                # Try all single coeff moves
                for i in range(12):
                    old_c = cur_coeffs[i]
                    for new_c in [-1, 1, -2, 2, -3, 3, -4, 4]:
                        if new_c == old_c:
                            continue
                        t_coeffs = cur_coeffs.copy()
                        t_coeffs[i] = new_c
                        t_score = _evaluate(cur_exps, t_coeffs)
                        if t_score < cur_score:
                            cur_coeffs = t_coeffs
                            cur_score = t_score
                            improved = True
                            break
                    if improved:
                        break
                            
            if not improved:
                break
                
        if cur_score < best_score:
            best_score = cur_score
            best_poly = list(zip(cur_exps, cur_coeffs))
            
    if best_poly is None:
        best_poly = seed_poly
        
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
        
    return coeff_list
