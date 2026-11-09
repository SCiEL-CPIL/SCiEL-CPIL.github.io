"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin for 90s kill

    def count_nonzero_square(coeffs):
        """Calculate the number of non-zero coefficients in the square of the polynomial."""
        # coeffs is a list of 12 integers
        # Square has degree 22
        square_coeffs = [0] * 23
        for i in range(12):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, 12):
                aj = coeffs[j]
                if aj == 0:
                    continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    def is_valid_poly(poly):
        if len(poly) != 12:
            return False
        if any(abs(c) > 64 for c in poly):
            return False
        if poly[-1] == 0:
            return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        return True

    best_count = 23
    best_poly = None

    # Initial population: mix of random small integers and known good structures
    # The space of coefficients is [-64, 64], but small integers cancel more often.
    # We focus on {-3, -2, -1, 1, 2, 3} initially, expanding if needed.
    
    def random_poly():
        # Bias towards small values
        vals = [-3, -2, -1, 1, 2, 3]
        return [random.choice(vals) for _ in range(12)]
    
    # Ensure final coeff is non-zero
    def ensure_valid(poly):
        if poly[-1] == 0:
            poly[-1] = 1
        return poly

    population = []
    for _ in range(100):
        p = ensure_valid(random_poly())
        population.append(p)
        
    # Add some structured patterns
    # Alternating signs
    population.append([1 if i % 2 == 0 else -1 for i in range(12)])
    # Blocks
    population.append([1]*6 + [-1]*6)
    population.append([1]*4 + [-1]*4 + [1]*4)
    
    # Evaluate initial population
    for p in population:
        c = count_nonzero_square(p)
        if c < best_count:
            best_count = c
            best_poly = p[:]

    # Genetic Algorithm Loop
    # Parameters
    pop_size = 100
    generations = 0
    
    while time.time() < deadline - 10:  # Reserve 10s for final SA
        # 1. Selection: Keep top 50%
        scored_pop = [(count_nonzero_square(p), p) for p in population]
        scored_pop.sort(key=lambda x: x[0])
        survivors = [p for _, p in scored_pop[:pop_size//2]]
        
        # Update global best
        if scored_pop[0][0] < best_count:
            best_count = scored_pop[0][0]
            best_poly = scored_pop[0][1][:]
            
        # 2. Crossover: Uniform crossover with repair
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = [random.choice([p1[i], p2[i]]) for i in range(12)]
            child = ensure_valid(child)
            new_pop.append(child)
            
        # 3. Mutation: 
        # - Single point mutation: change one coeff
        # - Swap: swap two coeffs
        # - Invert sign: flip sign of one coeff
        for i in range(len(new_pop)):
            if random.random() < 0.7: # Mutation rate
                p = new_pop[i][:]
                mut_type = random.randint(0, 2)
                if mut_type == 0:
                    # Change value
                    idx = random.randint(0, 11)
                    # Try small range first
                    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                    p[idx] = random.choice(vals)
                    if p[idx] == 0:
                        p[idx] = 1
                elif mut_type == 1:
                    # Swap
                    i1, i2 = random.sample(range(12), 2)
                    p[i1], p[i2] = p[i2], p[i1]
                else:
                    # Sign flip
                    idx = random.randint(0, 11)
                    p[idx] = -p[idx]
                    if p[idx] == 0:
                        p[idx] = 1
                new_pop[i] = ensure_valid(p)
                
        population = new_pop
        generations += 1
        
    # Final Phase: Simulated Annealing from the best found so far
    if best_poly is None:
        return [1] * 12
        
    current = best_poly[:]
    current_cost = best_count
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    def sa_step():
        p = current[:]
        # Random mutation
        idx = random.randint(0, 11)
        old_val = p[idx]
        # Perturb value
        delta = random.choice([-2, -1, 1, 2])
        new_val = old_val + delta
        # Clamp to [-64, 64] and non-zero
        if new_val == 0:
            new_val = 1
        if new_val > 64: new_val = 64
        if new_val < -64: new_val = -64
        p[idx] = new_val
        return p

    while time.time() < deadline:
        if T < T_min:
            break
            
        new_p = sa_step()
        new_cost = count_nonzero_square(new_p)
        
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current = new_p
            current_cost = new_cost
            if current_cost < best_count:
                best_count = current_cost
                best_poly = current[:]
        
        T *= alpha

    # Final verification
    if best_poly is None:
        return [1] * 12
        
    # Ensure constraints are met
    if not is_valid_poly(best_poly):
        # Fallback to a known valid poly
        return [1] * 12
        
    return best_poly

# EVOLVE-BLOCK-END
