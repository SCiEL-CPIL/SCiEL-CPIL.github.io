import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    random.seed(12345)

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Generate initial diverse population
    def gen_initial_population(pop_size):
        pop = []
        seen = set()
        # Structured seeds
        # Interval
        lst = list(range(1, 33))
        m = list_to_mask(lst)
        pop.append((lst, m, score_mask(m)))
        # APs
        for step in range(5, 16):
            lst = [1 + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                m = list_to_mask(lst)
                pop.append((lst, m, score_mask(m)))
        # Sumset structures
        for mult in [2, 3, 4, 5]:
            for trial in range(30):
                sizeA = random.randint(3, 8)
                sizeB = random.randint(3, 8)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    key = tuple(sorted(lst))
                    if key not in seen:
                        seen.add(key)
                        pop.append((lst, m, s))
        # Random
        while len(pop) < pop_size:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                pop.append((lst, m, s))
        return pop

    def tournament_select(pop, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(pop)
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def crossover(parent1, parent2):
        """Uniform crossover: each position picks from parent1 or parent2."""
        child = []
        for i in range(N):
            if random.random() < 0.5:
                child.append(parent1[0][i])
            else:
                child.append(parent2[0][i])
        # Resolve duplicates
        seen = set()
        final = []
        for v in child:
            if v not in seen:
                seen.add(v)
                final.append(v)
        # Fill remaining with random values
        while len(final) < N:
            v = random.randint(1, MAX_VAL)
            if v not in seen:
                seen.add(v)
                final.append(v)
        m = list_to_mask(final)
        s = score_mask(m)
        return (final, m, s)

    def mutate(individual, rate=0.15):
        lst = individual[0][:]
        s_set = set(lst)
        for i in range(N):
            if random.random() < rate:
                old = lst[i]
                new = random.randint(1, MAX_VAL)
                attempts = 0
                while new in s_set and attempts < 50:
                    new = random.randint(1, MAX_VAL)
                    attempts += 1
                if new not in s_set:
                    s_set.discard(old)
                    s_set.add(new)
                    lst[i] = new
        m = list_to_mask(lst)
        s = score_mask(m)
        return (lst, m, s)

    def local_search(individual, max_iter=200):
        """Hill climbing on a single individual."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        for _ in range(max_iter):
            improved = False
            for i in range(N):
                old = lst[i]
                best_new = old
                best_s = s
                # Try a few random replacements
                for _ in range(5):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    new_m = m ^ (1 << old) ^ (1 << new)
                    new_s = score_mask(new_m)
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= (1 << old) ^ (1 << best_new)
                    s = best_s
                    improved = True
            if not improved:
                break
        return (lst, m, s)

    # Initialize population
    pop_size = 40
    population = gen_initial_population(pop_size)
    
    # Sort by fitness
    population.sort(key=lambda x: x[2], reverse=True)
    best_individual = population[0]
    best_score = best_individual[2]
    best_list = best_individual[0][:]
    best_mask = best_individual[1]
    
    print(f"Initial best: {best_score}", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    no_improve = 0
    elite_count = 5
    
    while time.time() < deadline:
        generation += 1
        if generation % 10 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)
        
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)
            
            if random.random() < 0.3:
                # Crossover
                child = crossover(p1, p2)
            else:
                # Mutation of best parent
                child = mutate(p1, rate=0.1)
            
            # Local search on child (limited iterations)
            if time.time() < deadline - 5:
                child = local_search(child, max_iter=50)
            
            new_population.append(child)
        
        # Update population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # Track best
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1
        
        # Restart mechanism if stuck
        if no_improve > 30:
            # Add fresh random individuals to maintain diversity
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            # Also try structured restarts
            for mult in [2, 3]:
                sizeA = random.randint(3, 7)
                sizeB = random.randint(3, 7)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0
    
    # Final polish: exhaustive single-swap local search on best
    if time.time() < deadline - 2:
        lst = best_list[:]
        m = best_mask
        s = best_score
        s_set = set(lst)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                old = lst[i]
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    new_m = m ^ (1 << old) ^ (1 << new)
                    new_s = score_mask(new_m)
                    if new_s > s:
                        s_set.discard(old)
                        s_set.add(new)
                        lst[i] = new
                        m = new_m
                        s = new_s
                        improved = True
                        break
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END
