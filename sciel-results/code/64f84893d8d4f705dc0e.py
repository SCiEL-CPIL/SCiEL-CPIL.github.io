import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m
    
    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def score_mask(mask):
        """
        Count distinct d > 0 such that there exists a with a, a+d, a+2d all in set.
        Using bitmask: check if (mask & (mask >> d) & (mask >> (2*d))) != 0
        """
        valid_diffs = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs
    
    # Generate initial population with diverse structures
    def generate_initial_population(size=50):
        pop = []
        
        # 4x8 grid construction (mentioned in observations as having score ~18)
        # Place elements in a grid pattern to maximize 3-AP differences
        grid_sets = []
        for offset in range(0, 4):
            base = 1 + offset * 2
            elements = []
            for row in range(8):
                for col in range(4):
                    val = base + row * 16 + col * 4
                    if 1 <= val <= MAX_VAL:
                        elements.append(val)
            if len(elements) >= N:
                grid_sets.append(elements[:N])
        
        # Add grid constructions
        for gs in grid_sets:
            if len(gs) == N:
                pop.append(gs)
        
        # Dense block at start
        pop.append(list(range(1, N + 1)))
        
        # Dense block at end
        pop.append(list(range(MAX_VAL - N + 1, MAX_VAL + 1)))
        
        # Two dense blocks
        pop.append(list(range(1, N // 2 + 1)) + list(range(MAX_VAL - N // 2 + 1, MAX_VAL + 1)))
        
        # Arithmetic progression with large step
        step = MAX_VAL // (N - 1)
        pop.append([1 + i * step for i in range(N)])
        
        # Random fills
        while len(pop) < size:
            pop.append(random.sample(range(1, MAX_VAL + 1), N))
        
        return pop[:size]
    
    # Initialize population
    population = generate_initial_population(50)
    
    # Evaluate initial population
    scored_pop = []
    for lst in population:
        mask = list_to_mask(lst)
        score = score_mask(mask)
        scored_pop.append((score, mask, lst))
    
    # Sort by score descending
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_pop[0][0]
    best_mask = scored_pop[0][1]
    best_list = scored_pop[0][2][:]
    
    # Genetic Algorithm
    pop_size = 30
    elite_count = 5
    crossover_rate = 0.7
    mutation_rate = 0.15
    T = 3.0
    cooling_rate = 0.9995
    
    def crossover(mask1, mask2):
        """Uniform crossover on bitmasks: for each bit position, randomly pick from parent1 or parent2"""
        child = 0
        for i in range(1, MAX_VAL + 1):
            if random.random() < 0.5:
                if (mask1 >> i) & 1:
                    child |= (1 << i)
            else:
                if (mask2 >> i) & 1:
                    child |= (1 << i)
        
        # Ensure exactly N elements
        lst = mask_to_list(child)
        if len(lst) > N:
            # Remove excess
            excess = len(lst) - N
            for _ in range(excess):
                idx = random.randint(0, len(lst) - 1)
                lst.pop(idx)
        elif len(lst) < N:
            # Add missing
            current_set = set(lst)
            while len(lst) < N:
                v = random.randint(1, MAX_VAL)
                if v not in current_set:
                    lst.append(v)
                    current_set.add(v)
        
        return list_to_mask(lst)
    
    def mutate(mask, rate):
        """Point mutation: flip random bits, maintaining exactly N elements"""
        lst = mask_to_list(mask)
        current_set = set(lst)
        
        num_mutations = int(N * rate) + random.randint(0, 2)
        
        for _ in range(num_mutations):
            if random.random() < 0.5 and lst:
                # Remove a random element
                idx = random.randint(0, len(lst) - 1)
                val = lst.pop(idx)
                current_set.remove(val)
            if len(lst) < N:
                # Add a random element
                while True:
                    v = random.randint(1, MAX_VAL)
                    if v not in current_set:
                        lst.append(v)
                        current_set.add(v)
                        break
        
        return list_to_mask(lst)
    
    # GA main loop
    current_pop = scored_pop[:pop_size]
    
    while time.time() < deadline:
        # Select parents (tournament selection)
        new_pop = []
        
        # Keep elites
        for i in range(min(elite_count, len(current_pop))):
            new_pop.append(current_pop[i])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection for two parents
            def tournament_select():
                candidates = random.sample(current_pop, min(3, len(current_pop)))
                return max(candidates, key=lambda x: x[0])
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if random.random() < crossover_rate:
                child_mask = crossover(parent1[1], parent2[1])
            else:
                child_mask = parent1[1]
            
            child_mask = mutate(child_mask, mutation_rate)
            child_score = score_mask(child_mask)
            child_list = mask_to_list(child_mask)
            
            new_pop.append((child_score, child_mask, child_list))
            
            # Update best
            if child_score > best_score:
                best_score = child_score
                best_mask = child_mask
                best_list = child_list[:]
        
        # Replace population
        current_pop = new_pop[:pop_size]
        current_pop.sort(key=lambda x: x[0], reverse=True)
        
        # Cool down
        T *= cooling_rate
    
    return best_list
