import random
import time

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses a Genetic Algorithm with bitset-based vectorized scoring.
    """
    
    N = 32
    MAX_VAL = 512
    
    def score_bitset(mask_int):
        """
        mask_int: Python integer where bit i is 1 if i is in the set.
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        count = 0
        # d can range from 1 to MAX_VAL // 2
        for d in range(1, MAX_VAL // 2 + 1):
            if (mask_int & (mask_int >> d) & (mask_int >> 2 * d)) != 0:
                count += 1
        return count

    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m

    # Initialize population
    pop_size = 120
    population = []
    
    # Seed with random sets
    for _ in range(pop_size):
        s = set(random.sample(range(1, MAX_VAL + 1), N))
        population.append(s)

    # Calculate initial scores
    scored_pop = []
    for s in population:
        m = set_to_mask(s)
        sc = score_bitset(m)
        scored_pop.append((sc, s, m))
    
    # Sort by score descending
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_pop[0][0]
    best_set = scored_pop[0][1]
    best_mask = scored_pop[0][2]
    
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        new_pop = []
        
        # Elitism: keep the top few
        elite_count = 10
        for i in range(elite_count):
            new_pop.append(scored_pop[i])
            
        # Generate offspring
        while len(new_pop) < pop_size:
            # Selection: Tournament
            t1 = random.choice(scored_pop)
            t2 = random.choice(scored_pop)
            parent1 = t1 if t1[0] > t2[0] else t2
            
            t3 = random.choice(scored_pop)
            t4 = random.choice(scored_pop)
            parent2 = t3 if t3[0] > t4[0] else t4
            
            s1 = parent1[1]
            s2 = parent2[1]
            
            # Crossover: Uniform
            child_set = set()
            for x in s1:
                if random.random() < 0.5:
                    child_set.add(x)
            for x in s2:
                if random.random() < 0.5:
                    child_set.add(x)
            
            # Ensure size is N
            if len(child_set) > N:
                # Remove random excess
                excess = list(child_set)
                random.shuffle(excess)
                for x in excess[N:]:
                    child_set.remove(x)
            elif len(child_set) < N:
                # Add random missing
                while len(child_set) < N:
                    x = random.randint(1, MAX_VAL)
                    if x not in child_set:
                        child_set.add(x)
            
            # Mutation
            # Random swap
            if random.random() < 0.8:
                if child_set:
                    to_remove = random.choice(list(child_set))
                    to_add = random.randint(1, MAX_VAL)
                    while to_add in child_set:
                        to_add = random.randint(1, MAX_VAL)
                    child_set.remove(to_remove)
                    child_set.add(to_add)
            
            # Calculate score
            m = set_to_mask(child_set)
            sc = score_bitset(m)
            
            new_pop.append((sc, child_set, m))
            
            if sc > best_score:
                best_score = sc
                best_set = child_set
                best_mask = m
        
        # Update population
        scored_pop = new_pop
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
    return sorted(best_set)
