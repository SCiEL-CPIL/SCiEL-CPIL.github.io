import time
import random
import numpy as np

def _score_mask(mask):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _score_set(s):
    mask = 0
    for x in s:
        mask |= (1 << x)
    return _score_mask(mask)

def _make_mask(s):
    mask = 0
    for x in s:
        mask |= (1 << x)
    return mask

def _mask_to_set(mask):
    s = set()
    for i in range(1, 513):
        if mask & (1 << i):
            s.add(i)
    return s

def _local_search(s, deadline):
    """Hill climbing with all single-swap moves."""
    current = set(s)
    current_mask = _make_mask(current)
    current_score = _score_mask(current_mask)
    best = set(current)
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(current):
            if time.time() > deadline:
                break
            for b in range(1, 513):
                if b in current:
                    continue
                new_mask = (current_mask & ~(1 << a)) | (1 << b)
                new_score = _score_mask(new_mask)
                if new_score > current_score:
                    current.discard(a)
                    current.add(b)
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best = set(current)
                        best_score = current_score
                    break
            if improved:
                break
    
    return best, best_score

def _iterated_local_search(s, deadline):
    """Iterated local search with random perturbations."""
    current = set(s)
    current_score = _score_set(current)
    best = set(current)
    best_score = current_score
    
    while time.time() < deadline:
        # Local search to convergence
        current, current_score = _local_search(current, deadline)
        if current_score > best_score:
            best = set(current)
            best_score = current_score
        
        if time.time() > deadline - 2:
            break
        
        # Random perturbation: swap 2-4 elements
        num_swaps = random.randint(2, 4)
        perturbed = set(current)
        for _ in range(num_swaps):
            if len(perturbed) != 32:
                break
            to_remove = random.choice(list(perturbed))
            to_add = random.randint(1, 512)
            if to_add in perturbed:
                continue
            perturbed.discard(to_remove)
            perturbed.add(to_add)
        
        current = perturbed
    
    return best, best_score

def _genetic_algorithm(pop_size, deadline):
    """Simple genetic algorithm."""
    # Initialize population
    population = []
    scores = []
    
    # First individual: interval
    s0 = set(range(1, 33))
    population.append(s0)
    scores.append(_score_set(s0))
    
    # Add random individuals
    while len(population) < pop_size:
        s = set(random.sample(range(1, 513), 32))
        population.append(s)
        scores.append(_score_set(s))
    
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = set(population[best_idx])
    best_score = scores[best_idx]
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best = set(population[0])
        
        # Create next generation
        new_pop = []
        new_scores = []
        
        # Elitism: keep top 2
        for i in range(min(2, len(population))):
            new_pop.append(set(population[i]))
            new_scores.append(scores[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size and time.time() < deadline:
            # Tournament selection
            def select():
                idxs = random.sample(range(len(population)), min(3, len(population)))
                best_i = max(idxs, key=lambda i: scores[i])
                return population[best_i]
            
            p1 = select()
            p2 = select()
            
            # Crossover: uniform
            child = set()
            for x in p1:
                if random.random() < 0.5:
                    child.add(x)
            for x in p2:
                if x not in child:
                    child.add(x)
            
            # Fix size
            if len(child) > 32:
                extras = list(child)
                random.shuffle(extras)
                for e in extras:
                    if len(child) <= 32:
                        break
                    child.discard(e)
            elif len(child) < 32:
                all_vals = set(range(1, 513)) - child
                if all_vals:
                    need = 32 - len(child)
                    extra = random.sample(list(all_vals), need)
                    child.update(extra)
            
            if len(child) != 32:
                child = set(random.sample(range(1, 513), 32))
            
            # Mutation
            if random.random() < 0.3:
                num_mut = random.randint(1, 3)
                for _ in range(num_mut):
                    to_remove = random.choice(list(child))
                    to_add = random.randint(1, 512)
                    if to_add not in child:
                        child.discard(to_remove)
                        child.add(to_add)
            
            new_pop.append(child)
            new_scores.append(_score_set(child))
        
        population = new_pop
        scores = new_scores
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Start with interval, do local search
    s0 = set(range(1, 33))
    best_s, best_score = _local_search(s0, deadline - 60)
    
    # Phase 2: Iterated local search from best
    if time.time() < deadline - 30:
        ils_s, ils_score = _iterated_local_search(best_s, deadline - 30)
        if ils_score > best_score:
            best_s, best_score = ils_s, ils_score
    
    # Phase 3: Genetic algorithm
    if time.time() < deadline - 5:
        ga_s, ga_score = _genetic_algorithm(20, deadline - 5)
        if ga_score > best_score:
            best_s, best_score = ga_s, ga_score
    
    # Final local search polish
    if time.time() < deadline:
        best_s, best_score = _local_search(best_s, deadline)
    
    return sorted(list(best_s))
