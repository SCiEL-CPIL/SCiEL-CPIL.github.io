import numpy as np
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True
    
    best_score = float('inf')
    best_poly = [1] * 12  # baseline
    
    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)
    
    # Known sparse square patterns - try structured constructions first
    
    # Pattern 1: Pairs of terms that cancel in cross terms
    # (a*x^i + b*x^j + c*x^k + d*x^l)^2 where cross terms cancel
    # Try symmetric patterns
    
    # Pattern 2: Sum of squares with orthogonal supports
    # If P = A + B where A*B has no overlapping terms with A^2 and B^2
    
    # Pattern 3: Cyclotomic-like polynomials
    # x^11 + x^10 + ... + 1 has known properties
    
    # Let's try systematic structured searches
    
    # Strategy: Use genetic algorithm with structured initial population
    
    # Initial population: various structured polynomials
    population = []
    
    # 1. Simple: all ones in various positions
    for _ in range(50):
        positions = sorted(random.sample(range(41), 12))
        coeffs = [0] * 41
        for p in positions:
            coeffs[p] = random.choice([-1, 1])
        # Ensure last nonzero is at highest position
        while coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) > 0 and coeffs[-1] != 0:
            population.append(coeffs)
    
    # 2. Symmetric patterns
    for _ in range(50):
        coeffs = [0] * 41
        # Pick 6 pairs that are symmetric
        possible_positions = list(range(41))
        random.shuffle(possible_positions)
        used = set()
        count = 0
        for p in possible_positions:
            if count >= 12:
                break
            if p in used:
                continue
            # Add symmetric partner
            partner = 40 - p
            if partner not in used:
                coeffs[p] = random.choice([-1, 1])
                coeffs[partner] = random.choice([-1, 1])
                used.add(p)
                used.add(partner)
                count += 2
            elif p not in used:
                coeffs[p] = random.choice([-1, 1])
                used.add(p)
                count += 1
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) > 0 and coeffs[-1] != 0 and sum(1 for c in coeffs if c != 0) == 12:
            population.append(coeffs)
    
    # 3. Try known sparse square constructions
    # (x^a + x^b + x^c + x^d + x^e + x^f)^2 type with cancellations
    
    # Evaluate initial population
    for poly in population:
        try_poly(poly)
    
    # Genetic algorithm
    def crossover(p1, p2):
        # Uniform crossover
        child = []
        for i in range(max(len(p1), len(p2))):
            if i < len(p1) and random.random() < 0.5:
                child.append(p1[i])
            elif i < len(p2):
                child.append(p2[i])
            else:
                child.append(0)
        # Fix to exactly 12 nonzero terms
        nonzero_indices = [i for i, c in enumerate(child) if c != 0]
        if len(nonzero_indices) > 12:
            # Keep 12 random ones
            keep = random.sample(nonzero_indices, 12)
            new_child = [0] * len(child)
            for i in keep:
                new_child[i] = child[i]
            child = new_child
        elif len(nonzero_indices) < 12:
            # Add random terms
            all_indices = list(range(len(child)))
            random.shuffle(all_indices)
            added = 0
            for i in all_indices:
                if added >= 12 - len(nonzero_indices):
                    break
                if i not in nonzero_indices:
                    child[i] = random.choice([-1, 1])
                    added += 1
        # Ensure last coefficient is nonzero
        while child and child[-1] == 0:
            child.pop()
        if len(child) == 0:
            child = [1]
        return child
    
    def mutate(poly, rate=0.3):
        child = list(poly)
        # Randomly flip signs
        for i in range(len(child)):
            if child[i] != 0 and random.random() < rate:
                child[i] = -child[i]
        # Randomly add/remove terms
        if random.random() < 0.3:
            # Remove a term
            nonzero_indices = [i for i, c in enumerate(child) if c != 0]
            if nonzero_indices:
                idx = random.choice(nonzero_indices)
                child[idx] = 0
        if random.random() < 0.3:
            # Add a term
            if len(child) < 41:
                pos = random.randint(0, len(child))
                child.insert(pos, random.choice([-1, 1]))
        # Fix to exactly 12 nonzero terms
        nonzero_indices = [i for i, c in enumerate(child) if c != 0]
        if len(nonzero_indices) > 12:
            keep = random.sample(nonzero_indices, 12)
            new_child = [0] * len(child)
            for i in keep:
                new_child[i] = child[i]
            child = new_child
        elif len(nonzero_indices) < 12:
            all_indices = list(range(len(child)))
            random.shuffle(all_indices)
            added = 0
            for i in all_indices:
                if added >= 12 - len(nonzero_indices):
                    break
                if i not in nonzero_indices:
                    child[i] = random.choice([-1, 1])
                    added += 1
        while child and child[-1] == 0:
            child.pop()
        if len(child) == 0:
            child = [1]
        return child
    
    # Run genetic algorithm with time budget
    max_generations = 1000
    pop_size = 100
    
    for gen in range(max_generations):
        if time.time() > deadline - 5:
            break
        
        # Select top half
        scored = [(evaluate(p), p) for p in population if is_valid(p)]
        scored.sort(key=lambda x: x[0])
        top_half = [p for _, p in scored[:pop_size // 2]]
        
        # Create next generation
        new_population = []
        while len(new_population) < pop_size:
            if random.random() < 0.1:
                # Elite: keep best
                new_population.append(random.choice(top_half))
            else:
                p1 = random.choice(top_half)
                p2 = random.choice(top_half)
                child = crossover(p1, p2)
                child = mutate(child)
                new_population.append(child)
        
        population = new_population
        
        # Evaluate and update best
        for poly in population:
            try_poly(poly)
        
        # If we found a very good score, do local refinement
        if best_score <= 16:
            # Try more targeted local search around best
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                perturbed = list(best_poly)
                # Randomly perturb
                for i in range(len(perturbed)):
                    if random.random() < 0.2:
                        perturbed[i] = random.choice([-1, 0, 1])
                # Fix to 12 nonzero
                nonzero_indices = [i for i, c in enumerate(perturbed) if c != 0]
                if len(nonzero_indices) > 12:
                    keep = random.sample(nonzero_indices, 12)
                    new_perturbed = [0] * len(perturbed)
                    for i in keep:
                        new_perturbed[i] = perturbed[i]
                    perturbed = new_perturbed
                elif len(nonzero_indices) < 12:
                    all_indices = list(range(len(perturbed)))
                    random.shuffle(all_indices)
                    added = 0
                    for i in all_indices:
                        if added >= 12 - len(nonzero_indices):
                            break
                        if i not in nonzero_indices:
                            perturbed[i] = random.choice([-1, 1])
                            added += 1
                while perturbed and perturbed[-1] == 0:
                    perturbed.pop()
                if len(perturbed) > 0 and perturbed[-1] != 0:
                    try_poly(perturbed)
    
    # Final local search around best
    if time.time() < deadline - 2:
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            perturbed = list(best_poly)
            # Single flip
            nonzero_indices = [i for i, c in enumerate(perturbed) if c != 0]
            if nonzero_indices:
                idx = random.choice(nonzero_indices)
                perturbed[idx] = -perturbed[idx]
            try_poly(perturbed)
            
            # Single move
            if len(best_poly) > 1:
                perturbed = list(best_poly)
                nonzero_indices = [i for i, c in enumerate(perturbed) if c != 0]
                if nonzero_indices:
                    idx = random.choice(nonzero_indices)
                    old_val = perturbed[idx]
                    new_pos = random.randint(0, len(perturbed) - 1)
                    perturbed[idx] = 0
                    if new_pos < len(perturbed):
                        perturbed[new_pos] += old_val
                    else:
                        perturbed.extend([0] * (new_pos - len(perturbed) + 1))
                        perturbed[new_pos] = old_val
                    # Fix to 12 nonzero
                    nonzero_indices2 = [i for i, c in enumerate(perturbed) if c != 0]
                    if len(nonzero_indices2) > 12:
                        keep = random.sample(nonzero_indices2, 12)
                        new_perturbed = [0] * len(perturbed)
                        for i in keep:
                            new_perturbed[i] = perturbed[i]
                        perturbed = new_perturbed
                    elif len(nonzero_indices2) < 12:
                        all_indices = list(range(len(perturbed)))
                        random.shuffle(all_indices)
                        added = 0
                        for i in all_indices:
                            if added >= 12 - len(nonzero_indices2):
                                break
                            if i not in nonzero_indices2:
                                perturbed[i] = random.choice([-1, 1])
                                added += 1
                    while perturbed and perturbed[-1] == 0:
                        perturbed.pop()
                    if len(perturbed) > 0 and perturbed[-1] != 0:
                        try_poly(perturbed)
    
    return best_poly
