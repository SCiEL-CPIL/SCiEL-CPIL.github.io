import random
import time

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for start in range(N - 3 * d):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    num_aps = len(aps)
    
    # Precompute for each AP the 4 indices
    ap_indices = aps  # list of tuples
    
    def count_violations_fast(colors):
        """Fast violation count using precomputed APs."""
        v = 0
        for ap in ap_indices:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if < 3 distinct
            if c0 == c1 == c2 == c3:
                v += 1
            else:
                # Check if exactly 2 distinct
                # Count distinct without set creation for speed
                distinct = 1
                if c1 != c0:
                    distinct += 1
                if c2 != c0 and c2 != c1:
                    distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2:
                    distinct += 1
                if distinct < 3:
                    v += 1
        return v
    
    def fitness(colors):
        return count_violations_fast(colors)
    
    def crossover(p1, p2):
        child = []
        for i in range(N):
            child.append(p1[i] if random.random() < 0.5 else p2[i])
        return child
    
    def mutate(colors, K):
        new = colors.copy()
        for i in range(N):
            if random.random() < 0.08:
                new[i] = random.randint(0, K - 1)
        return new
    
    start_time = time.time()
    time_limit = 55.0
    
    for K in range(3, 10):
        if time.time() - start_time > time_limit:
            break
        
        pop_size = 80
        # Initialize with diverse starting points
        population = []
        for _ in range(pop_size):
            ind = [random.randint(0, K - 1) for _ in range(N)]
            population.append(ind)
        
        # Initial evaluation
        fits = [fitness(ind) for ind in population]
        for i, f in enumerate(fits):
            if f == 0:
                return population[i]
        
        elite_size = 5
        
        while time.time() - start_time < time_limit:
            # Sort by fitness
            order = sorted(range(len(population)), key=lambda i: fits[i])
            population = [population[i] for i in order]
            fits = [fits[i] for i in order]
            
            if fits[0] == 0:
                return population[0]
            
            # New generation
            new_pop = []
            # Elites
            for i in range(min(elite_size, len(population))):
                new_pop.append(population[i].copy())
            
            # Fill rest
            while len(new_pop) < pop_size:
                # Tournament selection
                idx1 = random.randint(0, min(10, len(population) - 1))
                idx2 = random.randint(0, min(10, len(population) - 1))
                # Pick better
                if fits[idx1] <= fits[idx2]:
                    p1 = population[idx1]
                else:
                    p1 = population[idx2]
                if fits[idx1] <= fits[idx2]:
                    p2 = population[idx2]
                else:
                    p2 = population[idx1]
                
                child = crossover(p1, p2)
                child = mutate(child, K)
                new_pop.append(child)
            
            population = new_pop
            fits = [fitness(ind) for ind in population]
            
            for i, f in enumerate(fits):
                if f == 0:
                    return population[i]
        
        # Try next K

    # Fallback greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d >= 0:
                    c0 = color
                    c1 = colors[x - 1 - d]
                    c2 = colors[x - 1 - 2 * d]
                    c3 = colors[x - 1 - 3 * d]
                    if len({c0, c1, c2, c3}) < 3:
                        ok = False
                        break
            if not ok:
                continue
            # Also check that we don't create a 2-color pattern with existing
            # The above check covers all 4-APs ending at x
            # But we also need to ensure that APs where x is not the last element are fine
            # Those were already checked when those positions were assigned
            # So just checking APs ending at x is sufficient for greedy
            colors.append(color)
            break
    return colors
