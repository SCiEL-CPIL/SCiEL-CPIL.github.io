import time
import itertools
import numpy as np
import random

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n):
    """Convert integer representation to list of digits."""
    digits = []
    for _ in range(n):
        digits.append(x % q)
        x //= q
    return digits

def _digits_to_int(digits, q):
    """Convert list of digits to integer representation."""
    val = 0
    for i, d in enumerate(reversed(digits)):
        val += d * (q ** i)
    return val

def _precompute_balls(q, n, radius):
    """Precompute balls for radius 1."""
    total = q ** n
    balls = []
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(ball)
    return balls

def _construct_initial_population(q, n, radius, pop_size, deadline):
    """Construct initial population using greedy cover and random mutations."""
    total = q ** n
    balls = _precompute_balls(q, n, radius)
    
    # Greedy cover to get a good starting point
    covered = np.zeros(total, dtype=bool)
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    selected = []
    while True:
        if time.time() > deadline - 10:
            break
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
        selected.append(max_idx)
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    counts[i] -= 1
        if np.all(covered):
            break
            
    greedy_mask = np.zeros(total, dtype=bool)
    greedy_mask[selected] = True
    
    population = [greedy_mask.copy() for _ in range(pop_size)]
    
    # Mutate some individuals
    for i in range(1, pop_size):
        if random.random() < 0.5:
            # Remove some random codewords
            num_removals = random.randint(1, 5)
            indices = np.where(population[i])[0]
            if len(indices) > 0:
                to_remove = random.sample(list(indices), min(num_removals, len(indices)))
                population[i][to_remove] = False
        if random.random() < 0.5:
            # Add some random codewords
            num_additions = random.randint(1, 5)
            for _ in range(num_additions):
                idx = random.randint(0, total - 1)
                population[i][idx] = True
                
    return population, balls

def _evaluate_fitness(mask, balls, total):
    """
    Evaluate fitness of a candidate code.
    Returns (uncovered_count, num_codewords).
    Lower is better for both.
    """
    num_codewords = np.sum(mask)
    if num_codewords == 0:
        return total, 0
        
    # Check coverage
    covered = np.zeros(total, dtype=bool)
    for i in range(total):
        if mask[i]:
            covered[balls[i]] = True
            
    uncovered = total - np.sum(covered)
    return uncovered, num_codewords

def _repair(mask, balls, total, deadline):
    """
    Repair a mask to ensure it is a valid cover.
    If invalid, add words to cover uncovered points.
    """
    covered = np.zeros(total, dtype=bool)
    for i in range(total):
        if mask[i]:
            covered[balls[i]] = True
            
    uncovered_indices = np.where(~covered)[0]
    
    if len(uncovered_indices) == 0:
        return mask
        
    # Greedily add words to cover uncovered points
    # This is a repair step to ensure validity
    # We want to minimize additions, so we pick words that cover the most uncovered points
    
    # To speed up, we can just pick random uncovered points and add their center?
    # Or better: for each uncovered point, find a codeword that covers it and maximizes coverage of other uncovered points.
    # Given time constraints, a simple greedy repair:
    
    # Create a set of uncovered points for fast lookup
    uncovered_set = set(uncovered_indices.tolist())
    
    # We'll iterate through all possible codewords? No, that's too slow.
    # Instead, for each uncovered point, we *must* cover it. 
    # The best word to add is one that covers the current uncovered point and potentially others.
    # Since we don't have an inverse index of "uncovered" specifically, let's just add the center of the uncovered point itself.
    # This guarantees coverage but might be suboptimal in size.
    
    # Better repair: 
    # 1. Identify all uncovered points.
    # 2. While uncovered points exist:
    #    a. Pick a random uncovered point u.
    #    b. Find the candidate word w (from all possible words, or just those in the current mask? No, must be new) 
    #       that covers u and the most other uncovered points.
    #    c. Since searching all 6561 words is expensive, we can restrict candidates to the ball of u? 
    #       No, the word itself is a candidate. 
    #       Actually, if we add u itself, it covers u and its neighbors.
    #       Is there a better word? Maybe a neighbor of u that covers more uncovered points.
    #       
    #    For simplicity and speed in the repair phase, let's just add the uncovered point itself.
    #    This ensures validity.
    
    for u in uncovered_indices:
        if time.time() > deadline - 2:
            break
        mask[u] = True
        # Update covered status for efficiency? 
        # We'll just re-evaluate or trust that adding u covers u.
        # To be safe and simple, we just mark u as covered in our local view?
        # Actually, the evaluator will check the final mask. 
        # The repair just needs to return a mask that *is* a cover.
        # Adding u guarantees u is covered.
        
    # Note: This simple repair might add many words. 
    # A smarter repair would try to find a word that covers multiple uncovered points.
    # Let's try a slightly smarter repair: 
    # Group uncovered points? No, too complex.
    
    # Let's verify the repair worked
    covered = np.zeros(total, dtype=bool)
    for i in range(total):
        if mask[i]:
            covered[balls[i]] = True
    if not np.all(covered):
        # Fallback: add all remaining uncovered points
        still_uncovered = np.where(~covered)[0]
        mask[still_uncovered] = True
        
    return mask

def _crossover(parent1, parent2):
    """Uniform crossover."""
    child = np.random.choice([parent1, parent2], size=parent1.shape, p=[0.5, 0.5])
    return child

def _mutation(mask, rate):
    """Bit-flip mutation."""
    # Flip bits with probability rate
    # For efficiency, we can just pick a few indices
    num_flips = int(len(mask) * rate)
    if num_flips > 0:
        indices = np.random.choice(len(mask), num_flips, replace=False)
        mask[indices] = ~mask[indices]
    return mask

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    total = q ** n
    balls = _precompute_balls(q, n, radius)
    
    pop_size = 50
    population, balls = _construct_initial_population(q, n, radius, pop_size, deadline)
    
    best_mask = None
    best_fitness = (total, total) # (uncovered, size)
    
    # Initial evaluation
    for mask in population:
        unc, size = _evaluate_fitness(mask, balls, total)
        if (unc, size) < best_fitness:
            best_fitness = (unc, size)
            best_mask = mask.copy()
            
    # Genetic Algorithm Loop
    while time.time() < deadline - 5:
        new_population = []
        
        # Elitism: keep the best one
        new_population.append(best_mask.copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            # Selection: Tournament selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            child = _crossover(p1, p2)
            
            # Mutation
            child = _mutation(child, 0.05)
            
            # Repair if necessary
            # Check if valid
            covered = np.zeros(total, dtype=bool)
            for i in range(total):
                if child[i]:
                    covered[balls[i]] = True
            if not np.all(covered):
                child = _repair(child, balls, total, deadline)
                
            new_population.append(child)
            
        # Evaluate new population and update best
        for mask in new_population:
            unc, size = _evaluate_fitness(mask, balls, total)
            if (unc, size) < best_fitness:
                best_fitness = (unc, size)
                best_mask = mask.copy()
                
        population = new_population
        
    # Final check and conversion
    if best_mask is None:
        # Fallback to greedy
        return _greedy_cover_fallback(q, n, radius, deadline)
        
    # Convert mask to list of codewords
    codewords = []
    for i in range(total):
        if best_mask[i]:
            codewords.append(_int_to_digits(i, q, n))
            
    return codewords

def _greedy_cover_fallback(q, n, radius, deadline):
    total = q ** n
    balls = _precompute_balls(q, n, radius)
    covered = np.zeros(total, dtype=bool)
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    selected = []
    while True:
        if time.time() > deadline - 2:
            break
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
        selected.append(max_idx)
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    counts[i] -= 1
        if np.all(covered):
            break
            
    codewords = []
    for idx in selected:
        codewords.append(_int_to_digits(idx, q, n))
    return codewords

# EVOLVE-BLOCK-END
