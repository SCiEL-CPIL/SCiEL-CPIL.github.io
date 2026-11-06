import time
import random

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 58.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    
    def build_next(trans):
        """Precompute subset transitions for a single letter."""
        size = 1 << N
        next_sub = [0] * size
        # next_sub[0] is 0
        for mask in range(1, size):
            # Use lowest set bit trick
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            next_sub[mask] = next_sub[rest] | (1 << trans[idx])
        return next_sub

    def compute_threshold(next_a, next_b):
        """BFS on subset automaton to find shortest reset word length."""
        size = 1 << N
        dist = [-1] * size
        dist[FULL_MASK] = 0
        queue = [FULL_MASK]
        head = 0
        
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            # Try 'a'
            n1 = next_a[s]
            if n1 == 1: # Single state 0
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
            # Try 'b'
            n2 = next_b[s]
            if n2 == 1:
                return d + 1
            if dist[n2] == -1:
                dist[n2] = d + 1
                queue.append(n2)
                
        return 0 # Not synchronizing

    def evaluate(fa, fb):
        next_a = build_next(fa)
        next_b = build_next(fb)
        return compute_threshold(next_a, next_b)

    rng = random.Random(20240517)
    
    def pinwheel_b(fa):
        """Create a 'b' transition that works well with a cycle 'a'."""
        fb = [0] * N
        for i in range(N):
            fb[i] = (i - 1) % N
        fb[0] = 0
        return fb

    def random_permutation():
        p = list(range(N))
        rng.shuffle(p)
        return p

    def random_b():
        return [rng.randrange(N) for _ in range(N)]

    # Genetic Algorithm Parameters
    POP_SIZE = 50
    ELITISM = 5
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.3
    
    # Initialize Population
    population = []
    
    # Add some known good structures
    fa1 = list(range(N))
    fb1 = pinwheel_b(fa1)
    population.append((fa1, fb1, evaluate(fa1, fb1)))
    
    fa2 = list(range(N))
    rng.shuffle(fa2)
    fb2 = pinwheel_b(fa2)
    population.append((fa2, fb2, evaluate(fa2, fb2)))
    
    # Fill rest with randoms
    while len(population) < POP_SIZE:
        fa = random_permutation()
        fb = random_b()
        score = evaluate(fa, fb)
        population.append((fa, fb, score))
        
    # Sort by score descending
    population.sort(key=lambda x: x[2], reverse=True)
    
    best_score = population[0][2]
    best_fa = population[0][0][:]
    best_fb = population[0][1][:]
    
    def tournament_select(pop):
        selected = rng.sample(pop, TOURNAMENT_SIZE)
        return max(selected, key=lambda x: x[2])
        
    def crossover(parent1, parent2):
        """
        Crossover for (fa, fb).
        fa is a permutation. We perform uniform crossover on fa but ensure it remains a permutation?
        Actually, uniform crossover on permutations breaks the permutation property.
        Strategy: 
        1. For fa (permutation): Swap-based crossover or just mutate. 
           Given the "cycle" heuristic, maybe we should just mutate fa slightly.
           Let's try: 50% chance to keep parent1's fa, 50% parent2's fa, then mutate.
           Or, perform a "permutation crossover" like OX (Order Crossover) or simply swap two elements.
           Simpler: Uniform crossover on indices, then repair to permutation if needed? 
           Repair is expensive. 
           Alternative: Since 'a' being a permutation is a strong prior, let's do:
           - 20%: keep parent1 fa
           - 20%: keep parent2 fa
           - 60%: Swap two random values in a copy of parent1 fa (or parent2).
           
        2. For fb (arbitrary): Standard uniform crossover.
        """
        # Crossover fa
        r = rng.random()
        if r < 0.2:
            child_fa = parent1[0][:]
        elif r < 0.4:
            child_fa = parent2[0][:]
        else:
            # Swap-based crossover: Pick two indices and swap in parent1
            child_fa = parent1[0][:]
            i, j = rng.sample(range(N), 2)
            child_fa[i], child_fa[j] = child_fa[j], child_fa[i]
            
        # Crossover fb
        child_fb = []
        for i in range(N):
            if rng.random() < 0.5:
                child_fb.append(parent1[1][i])
            else:
                child_fb.append(parent2[1][i])
                
        return child_fa, child_fb

    def mutate(fa, fb):
        """
        Mutate fa and fb.
        fa: Swap two elements (keep permutation).
        fb: Change one or two destinations.
        """
        # Mutate fa
        if rng.random() < MUTATION_RATE:
            i, j = rng.sample(range(N), 2)
            fa[i], fa[j] = fa[j], fa[i]
            
        # Mutate fb
        if rng.random() < MUTATION_RATE:
            i = rng.randrange(N)
            fb[i] = rng.randrange(N)
            
        return fa, fb

    # Main GA Loop
    while time.time() - start_time < DEADLINE:
        new_population = []
        
        # Elitism
        for i in range(ELITISM):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            
            child_fa, child_fb = crossover(p1, p2)
            child_fa, child_fb = mutate(child_fa, child_fb)
            
            # Evaluate
            score = evaluate(child_fa, child_fb)
            
            if score > best_score:
                best_score = score
                best_fa = child_fa[:]
                best_fb = child_fb[:]
                
            new_population.append((child_fa, child_fb, score))
            
        # Replace population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # If we are very close to deadline, do a final hill climb on the best individual
        # to squeeze out any last improvements
        if time.time() - start_time > DEADLINE - 5:
             # Simple local search on the best
             curr_fa = best_fa[:]
             curr_fb = best_fb[:]
             curr_score = best_score
             next_a = build_next(curr_fa)
             
             for _ in range(1000):
                 if time.time() - start_time > DEADLINE - 1:
                     break
                 
                 # Randomly mutate b
                 old_b = curr_fb[:]
                 state = rng.randrange(N)
                 new_dest = rng.randrange(N)
                 curr_fb[state] = new_dest
                 
                 next_b = build_next(curr_fb)
                 new_score = compute_threshold(next_a, next_b)
                 
                 if new_score > curr_score:
                     curr_score = new_score
                     if curr_score > best_score:
                         best_score = curr_score
                         best_fa = curr_fa[:]
                         best_fb = curr_fb[:]
                 else:
                     curr_fb = old_b

    return best_fa, best_fb
