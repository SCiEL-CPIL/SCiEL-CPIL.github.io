import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21
    
    best_score = fallback_score
    best_candidate = fallback

    # Pre-define small coefficient values to try
    COEFF_VALUES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    def compute_score(exps, coeffs):
        """
        Computes the number of non-zero terms in P(x)^2.
        exps: list of 12 exponents
        coeffs: list of 12 coefficients
        """
        if not exps or not coeffs:
            return float('inf')
        
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
            
        q = np.convolve(p, p)
        count = np.count_nonzero(q)
        return count

    def _update_best(exps, coeffs):
        """Update the global best candidate dense list."""
        nonlocal best_score, best_candidate
        score = compute_score(exps, coeffs)
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def local_search(exps, coeffs, max_iters=500):
        """
        Local search to minimize non-zero terms in square.
        Moves:
        1. Change coefficient value.
        2. Shift an exponent by +/- 1 (if valid and no collision).
        """
        nonlocal best_score, best_candidate
        
        current_score = compute_score(exps, coeffs)
        _update_best(exps, coeffs)
        
        no_improve_count = 0
        iterations = 0
        
        # Sort exps and coeffs together initially
        sorted_idx = np.argsort(exps)
        exps = [exps[i] for i in sorted_idx]
        coeffs = [coeffs[i] for i in sorted_idx]
        
        while iterations < max_iters and (time.time() < deadline - 2):
            iterations += 1
            improved = False
            
            # 1. Try changing coefficients
            for i in range(12):
                for new_c in COEFF_VALUES:
                    if new_c == coeffs[i]:
                        continue
                    old_c = coeffs[i]
                    coeffs[i] = new_c
                    new_score = compute_score(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        _update_best(exps, coeffs)
                    else:
                        coeffs[i] = old_c
                        
            if improved:
                continue
                
            # 2. Try shifting exponents
            for i in range(12):
                for delta in [-1, 1]:
                    new_e = exps[i] + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    if new_e in exps: # Collision check
                        continue
                        
                    old_e = exps[i]
                    exps[i] = new_e
                    
                    new_score = compute_score(exps, coeffs)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        _update_best(exps, coeffs)
                        idx = np.argsort(exps)
                        exps = [exps[j] for j in idx]
                        coeffs = [coeffs[j] for j in idx]
                    else:
                        exps[i] = old_e
                        
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    break
            else:
                no_improve_count = 0

        return exps, coeffs, current_score

    def genetic_algorithm(pop_size=20, generations=50):
        """
        Genetic algorithm with crossover and mutation to explore the space.
        """
        nonlocal best_score, best_candidate
        
        # Initialize population
        population = []
        
        # Seed with diverse candidates
        def generate_random_individual():
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice(COEFF_VALUES) for _ in range(12)]
            return exps, coeffs
        
        # Add structured candidates
        structured_candidates = []
        for start in range(0, 5):
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                for pattern in [
                    [1]*12,
                    [1 if i%2==0 else -1 for i in range(12)],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                ]:
                    structured_candidates.append((exps[:], pattern[:]))
        
        # Add random candidates
        for _ in range(pop_size - len(structured_candidates)):
            structured_candidates.append(generate_random_individual())
        
        population = structured_candidates[:pop_size]
        
        # Evaluate initial population
        for exps, coeffs in population:
            _update_best(exps, coeffs)
        
        for gen in range(generations):
            if time.time() > deadline - 5:
                break
            
            # Evaluate population
            scored_population = []
            for exps, coeffs in population:
                score = compute_score(exps, coeffs)
                scored_population.append((score, exps, coeffs))
            
            scored_population.sort(key=lambda x: x[0])
            
            # Elitism: keep top half
            elite_count = max(1, pop_size // 2)
            new_population = []
            for _, exps, coeffs in scored_population[:elite_count]:
                new_population.append((exps[:], coeffs[:]))
            
            # Generate offspring via crossover and mutation
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break
                
                # Tournament selection
                def tournament_select(pop, k=3):
                    candidates = random.sample(pop, min(k, len(pop)))
                    return min(candidates, key=lambda x: x[0])
                
                parent1 = tournament_select(scored_population)
                parent2 = tournament_select(scored_population)
                
                # Crossover: uniform crossover on exponents and coefficients
                child_exps = []
                child_coeffs = []
                for i in range(12):
                    if random.random() < 0.5:
                        child_exps.append(parent1[1][i])
                        child_coeffs.append(parent1[2][i])
                    else:
                        child_exps.append(parent2[1][i])
                        child_coeffs.append(parent2[2][i])
                
                # Ensure unique exponents
                if len(set(child_exps)) != 12:
                    # Fallback: use one parent's exponents
                    child_exps = parent1[1][:]
                    child_coeffs = parent1[2][:]
                
                # Mutation
                mutation_rate = 0.1
                for i in range(12):
                    if random.random() < mutation_rate:
                        # Mutate coefficient
                        child_coeffs[i] = random.choice(COEFF_VALUES)
                    if random.random() < mutation_rate:
                        # Mutate exponent
                        delta = random.choice([-2, -1, 1, 2])
                        new_e = child_exps[i] + delta
                        if 0 <= new_e <= 40 and new_e not in child_exps:
                            child_exps[i] = new_e
                
                # Validate
                if len(set(child_exps)) == 12 and all(0 <= e <= 40 for e in child_exps) and all(abs(c) <= 64 and c != 0 for c in child_coeffs):
                    new_population.append((child_exps, child_coeffs))
                
                # Local search on some offspring
                if random.random() < 0.3 and time.time() < deadline - 10:
                    child_exps, child_coeffs, _ = local_search(child_exps, child_coeffs, max_iters=50)
                    new_population[-1] = (child_exps, child_coeffs)
            
            population = new_population
        
        return best_score, best_candidate

    # Run genetic algorithm
    best_score, best_candidate = genetic_algorithm(pop_size=20, generations=50)
    
    # Final local search on the best candidate
    if best_candidate and time.time() < deadline - 5:
        max_deg = len(best_candidate) - 1
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [best_candidate[i] for i in exps]
        if len(exps) == 12:
            exps, coeffs, _ = local_search(exps, coeffs, max_iters=200)
            _update_best(exps, coeffs)

    return best_candidate

# EVOLVE-BLOCK-END
