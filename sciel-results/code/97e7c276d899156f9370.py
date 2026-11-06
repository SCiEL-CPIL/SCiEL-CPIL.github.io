import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0  # safety margin before 90s kill

    # Precompute triple masks: for each x, list of masks of 3 elements that with x form a 4-AP
    triple_masks = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a, b, c = x + d, x + 2 * d, x + 3 * d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
        
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a, b, c = x - d, x + d, x + 2 * d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a, b, c = x - 2 * d, x - d, x + d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a, b, c = x - 3 * d, x - 2 * d, x - d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))

    def can_add(mask, x):
        for m in triple_masks[x]:
            if (mask & m) == m:
                return False
        return True

    def popcount(mask):
        return bin(mask).count('1')

    def randomized_greedy(rng, seed_val):
        mask = 0
        order = list(range(1, n + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def greedy_fill(mask, rng):
        """Try to add any element that fits, in random order."""
        candidates = [i for i in range(1, n + 1) if not ((mask >> i) & 1)]
        rng.shuffle(candidates)
        for x in candidates:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def find_violations(mask):
        """Find all 4-APs in the set. Returns list of (a,b,c,d) tuples."""
        violations = []
        for a in range(1, n + 1):
            if not ((mask >> a) & 1):
                continue
            for d in range(1, (n - a) // 3 + 1):
                b, c, dd = a + d, a + 2*d, a + 3*d
                if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> dd) & 1):
                    violations.append((a, b, c, dd))
        return violations

    def repair(mask, rng):
        """Remove elements to fix all 4-AP violations."""
        changed = True
        while changed:
            changed = False
            violations = find_violations(mask)
            if not violations:
                break
            # Pick a random violation and remove one random element from it
            viol = rng.choice(violations)
            x = rng.choice(viol)
            mask &= ~(1 << x)
            changed = True
        return mask

    def crossover(parent1, parent2, rng, num_blocks=16):
        """Block-based crossover: partition [1..n] into blocks, pick fitter parent's block."""
        child = 0
        size1 = popcount(parent1)
        size2 = popcount(parent2)
        block_size = n // num_blocks
        for i in range(num_blocks):
            start = i * block_size + 1
            end = start + block_size - 1
            if i == num_blocks - 1:
                end = n
            # Count bits in this block for each parent
            count1 = sum(1 for x in range(start, end + 1) if (parent1 >> x) & 1)
            count2 = sum(1 for x in range(start, end + 1) if (parent2 >> x) & 1)
            # Pick the parent with more bits in this block
            if count1 >= count2:
                for x in range(start, end + 1):
                    if (parent1 >> x) & 1:
                        child |= (1 << x)
            else:
                for x in range(start, end + 1):
                    if (parent2 >> x) & 1:
                        child |= (1 << x)
        return child

    def mutate_add(mask, rng):
        """Try to add random elements that don't violate constraints."""
        candidates = [i for i in range(1, n + 1) if not ((mask >> i) & 1)]
        rng.shuffle(candidates)
        for x in candidates:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    # Initialize population with diverse seeds
    rng = random.Random(12345)
    population = []
    
    # Base-3 Salem-Spencer set as one seed
    init_mask = 0
    powers = [1, 3, 9, 27, 81, 243]
    for i in range(1, 1 << len(powers)):
        val = 0
        for j, p in enumerate(powers):
            if (i >> j) & 1:
                val += p
        if 1 <= val <= n:
            init_mask |= (1 << val)
    population.append(init_mask)
    
    # Generate more diverse seeds
    num_seeds = 30
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit:
            break
        mask = randomized_greedy(rng, seed)
        population.append(mask)
    
    # Ensure population size
    while len(population) < 20:
        population.append(randomized_greedy(rng, len(population) * 1000))
    
    pop_size = len(population)
    best_mask = max(population, key=popcount)
    best_size = popcount(best_mask)
    
    # Genetic algorithm
    generation = 0
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort population by fitness
        population.sort(key=popcount, reverse=True)
        
        # Check for improvement
        current_best = popcount(population[0])
        if current_best > best_size:
            best_mask = population[0]
            best_size = current_best
            no_improve = 0
        else:
            no_improve += 1
        
        # If no improvement for many generations, do a kick
        if no_improve > 5:
            # Remove some random elements from best and refill
            kick_mask = best_mask
            elements_in_set = [i for i in range(1, n + 1) if (kick_mask >> i) & 1]
            k = rng.randint(3, 8)
            to_remove = rng.sample(elements_in_set, min(k, len(elements_in_set)))
            for x in to_remove:
                kick_mask &= ~(1 << x)
            kick_mask = greedy_fill(kick_mask, rng)
            population.append(kick_mask)
            no_improve = 0
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top 2
        for i in range(min(2, pop_size)):
            new_population.append(population[i])
        
        # Fill rest with crossover + mutation + repair
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit - 3:
                break
            
            # Tournament selection
            p1_idx = rng.randint(0, min(5, pop_size - 1))
            p2_idx = rng.randint(0, min(5, pop_size - 1))
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            
            # Crossover
            child = crossover(parent1, parent2, rng)
            
            # Mutation: try to add elements
            child = mutate_add(child, rng)
            
            # Repair: remove elements to fix violations
            child = repair(child, rng)
            
            # After repair, try to add more
            child = greedy_fill(child, rng)
            
            new_population.append(child)
        
        population = new_population
        
        # Occasionally inject fresh randomized greedy solutions
        if generation % 10 == 0 and time.time() - start_time < time_limit - 5:
            fresh = randomized_greedy(rng, generation * 7777)
            fresh = greedy_fill(fresh, rng)
            population[-1] = fresh
    
    # Final polish: try to add any remaining elements
    final_mask = best_mask
    candidates = [i for i in range(1, n + 1) if not ((final_mask >> i) & 1)]
    rng.shuffle(candidates)
    for x in candidates:
        if can_add(final_mask, x):
            final_mask |= (1 << x)
    
    # One more pass in different order
    candidates = [i for i in range(1, n + 1) if not ((final_mask >> i) & 1)]
    rng.shuffle(candidates)
    for x in candidates:
        if can_add(final_mask, x):
            final_mask |= (1 << x)
    
    # Convert to sorted list
    result = [i for i in range(1, n + 1) if (final_mask >> i) & 1]
    
    # Safety: ensure nonempty
    if not result:
        result = [1]
    
    return result
