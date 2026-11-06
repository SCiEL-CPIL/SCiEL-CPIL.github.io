import math
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    LIMIT = 10**18
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    k = len(primes)
    
    def calc_score_from_exponents(exps):
        """
        Calculate tau_perp(n) for n = prod p_i^e_i.
        Returns (score, n).
        """
        n = 1
        divisors = [1]
        
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            n *= (p ** e)
            if n > LIMIT:
                return -1, n
            
            # Update divisors
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
        
        if n < 2:
            return 0, n
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score, n

    def get_n_value(exps):
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= (primes[i] ** e)
        return n

    best_score = -1
    best_n = 1
    best_exps = [0] * k
    
    # Phase 1: Initial Candidates
    candidates = []
    
    # 1. Primorials (all 1s)
    candidates.append([1] * 15)
    
    # 2. Powers of 2
    for e in range(1, 60):
        candidates.append([e] + [0]*14)
        
    # 3. Mixed: Drop large primes, bump small
    base = [1] * 15
    for drop_idx in range(14, 5, -1):
        for bump_idx in range(0, 3):
            for b in range(1, 6):
                exps = base[:]
                exps[drop_idx] = 0
                exps[bump_idx] += b
                if get_n_value(exps) <= LIMIT:
                    candidates.append(exps)
    
    # Evaluate initial candidates
    for exps in candidates:
        if time.time() > DEADLINE: break
        score, n_val = calc_score_from_exponents(exps)
        if score > best_score:
            best_score = score
            best_n = n_val
            best_exps = exps[:]

    # Phase 2: Genetic Algorithm
    if time.time() < DEADLINE - 20:
        # Initialize population
        pop_size = 50
        population = [best_exps[:]] * pop_size
        # Add some diversity
        for i in range(1, pop_size):
            parent = random.choice(population)
            child = parent[:]
            # Random mutation
            for _ in range(3):
                idx = random.randint(0, k-1)
                delta = random.choice([-1, 1])
                new_exp = child[idx] + delta
                if 0 <= new_exp <= 20:
                    child[idx] = new_exp
            if get_n_value(child) <= LIMIT:
                population[i] = child
            else:
                population[i] = best_exps[:]
        
        # Precompute fitness for initial pop
        pop_fitness = []
        for exps in population:
            s, _ = calc_score_from_exponents(exps)
            pop_fitness.append(s)
        
        generations = 0
        while time.time() < DEADLINE - 10:
            generations += 1
            new_population = []
            new_fitness = []
            
            # Elitism
            best_idx = pop_fitness.index(max(pop_fitness))
            new_population.append(population[best_idx][:])
            new_fitness.append(pop_fitness[best_idx])
            
            while len(new_population) < pop_size:
                # Tournament Selection
                p1_idx = random.randint(0, pop_size-1)
                p2_idx = random.randint(0, pop_size-1)
                if pop_fitness[p1_idx] > pop_fitness[p2_idx]:
                    p1 = population[p1_idx]
                else:
                    p1 = population[p2_idx]
                
                p3_idx = random.randint(0, pop_size-1)
                p4_idx = random.randint(0, pop_size-1)
                if pop_fitness[p3_idx] > pop_fitness[p4_idx]:
                    p2 = population[p3_idx]
                else:
                    p2 = population[p4_idx]
                
                # Crossover
                child = p1[:]
                for i in range(k):
                    if random.random() < 0.5:
                        child[i] = p2[i]
                        
                # Mutation
                if random.random() < 0.8:
                    idx = random.randint(0, k-1)
                    delta = random.choice([-1, 1])
                    child[idx] += delta
                    if child[idx] < 0: child[idx] = 0
                    if child[idx] > 20: child[idx] = 20
                    
                # Constraint Check
                if get_n_value(child) <= LIMIT:
                    new_population.append(child)
                    s, _ = calc_score_from_exponents(child)
                    new_fitness.append(s)
                else:
                    # Fallback to parent if invalid
                    new_population.append(p1[:])
                    new_fitness.append(pop_fitness[best_idx]) # Approximate
                    
            population = new_population
            pop_fitness = new_fitness
            
            # Update Best
            max_fit = max(pop_fitness)
            if max_fit > best_score:
                best_score = max_fit
                best_idx = pop_fitness.index(max_fit)
                best_n = get_n_value(population[best_idx])
                best_exps = population[best_idx][:]
                
    # Phase 3: Local Refinement of Best
    if time.time() < DEADLINE - 2:
        current_exps = best_exps[:]
        current_score = best_score
        current_n = best_n
        
        # Simple Hill Climbing with random restarts
        for _ in range(50):
            if time.time() > DEADLINE - 2: break
            improved = True
            while improved:
                improved = False
                for i in range(k):
                    for delta in [-1, 1]:
                        new_exp = current_exps[i] + delta
                        if new_exp < 0 or new_exp > 20: continue
                        test_exps = current_exps[:]
                        test_exps[i] = new_exp
                        if get_n_value(test_exps) > LIMIT: continue
                        s, n_val = calc_score_from_exponents(test_exps)
                        if s > current_score:
                            current_score = s
                            current_n = n_val
                            current_exps = test_exps
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exps = current_exps[:]
            # Random restart
            if random.random() < 0.5:
                restart_exps = best_exps[:]
                for i in range(k):
                    if random.random() < 0.2:
                        restart_exps[i] = max(0, restart_exps[i] + random.randint(-2, 2))
                if get_n_value(restart_exps) <= LIMIT:
                    current_exps = restart_exps
                    s, n_val = calc_score_from_exponents(current_exps)
                    current_score = s
                    current_n = n_val

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
