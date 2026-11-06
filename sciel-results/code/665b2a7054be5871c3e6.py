import time
import random
import copy

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    size = SIZE
    # Precompute transitions for all subsets
    # Note: In a GA, we call this frequently. 
    # Optimizing this is critical.
    # We can cache subset transitions if the automaton doesn't change, 
    # but here the automaton changes every generation.
    # However, we can optimize the inner loop.
    
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    for subset in range(size):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # 1. Initialization: Generate a population of "Pinwheel-like" automata
    # The Pinwheel automaton is known to have high reset thresholds.
    # Standard Pinwheel: a is a cycle, b maps i to i+1 except for a specific "break".
    # We will create diverse variants of this structure to seed the GA.
    
    def generate_pinwheel_variant():
        a = [(i + 1) % N for i in range(N)]
        b = a[:]
        # Randomly break the cycle for b in a few places
        # To ensure it's not just a cycle, we need to map some states to "lower" indices or create a sink-like structure
        # A common high-threshold structure is:
        # a: i -> i+1 mod N
        # b: i -> i-1 mod N for i>0, 0 -> 0 (or similar)
        
        # Let's use a mix of "almost cycles"
        # Pick 1 or 2 positions to change in b
        num_changes = random.randint(1, 2)
        for _ in range(num_changes):
            idx = random.randint(0, N - 1)
            # Map to a "random" state, but bias towards creating a "trap" or "slow" path
            # Often mapping to 0 or a small index helps synchronization eventually
            target = random.randint(0, N - 1)
            b[idx] = target
            
        # Ensure it synchronizes. If not, we'll repair it later.
        return a, b

    def repair(a, b):
        """
        Attempts to make the automaton synchronizing by adding a 'reset' letter effect.
        Strategy: Pick a state r. Force b to map many states to r, or create a path to a singleton.
        A robust way: 
        1. Check if synchronizing.
        2. If not, pick a random state 'sink'. 
        3. Modify b to map a random subset of states to 'sink'.
        4. Iterate until synchronizing or limit reached.
        """
        for _ in range(50):
            l = compute_reset_length(a, b)
            if l != -1:
                return True
            
            # Modification: Pick a random state and force some transitions to it
            # This effectively creates a "reset" behavior for those states
            target_state = random.randint(0, N - 1)
            num_forced = random.randint(1, 3)
            for _ in range(num_forced):
                src = random.randint(0, N - 1)
                if random.random() < 0.5:
                    b[src] = target_state
                else:
                    a[src] = target_state
        return False

    # Population
    pop_size = 40
    population = []
    scores = []
    
    # Seed with a few known good structures if possible, otherwise random pinwheels
    # Let's start with a specific known high-threshold candidate:
    # a: 0->1->...->11->0
    # b: 0->0, 1->0, 2->1, ..., 11->10
    # This is the "standard" pinwheel variant.
    
    a_std = [(i + 1) % N for i in range(N)]
    b_std = [0] * N
    for i in range(1, N):
        b_std[i] = i - 1
        
    population.append((a_std[:], b_std[:]))
    
    for _ in range(pop_size - 1):
        a, b = generate_pinwheel_variant()
        if repair(a, b):
            population.append((a, b))
        else:
            # Fallback to standard if repair fails
            population.append((a_std[:], b_std[:]))
            
    # Evaluate initial population
    for i, (a, b) in enumerate(population):
        scores.append(compute_reset_length(a, b))
        
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_a = population[best_idx][0][:]
    best_b = population[best_idx][1][:]
    best_len = scores[best_idx]
    
    # 2. Genetic Algorithm Loop
    # Parameters
    mutation_rate = 0.1
    crossover_rate = 0.7
    elite_count = 2
    
    generation = 0
    no_improve_generations = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Selection: Tournament selection
        def select():
            i1 = random.randint(0, pop_size - 1)
            i2 = random.randint(0, pop_size - 1)
            if scores[i1] > scores[i2]:
                return population[i1]
            else:
                return population[i2]
        
        new_population = []
        
        # Elitism
        # Sort population by score
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(population[idx])
            
        # Generate offspring
        while len(new_population) < pop_size:
            if random.random() < crossover_rate:
                p1 = select()
                p2 = select()
                # Crossover: Uniform crossover on the transition lists
                a1, b1 = p1
                a2, b2 = p2
                a_child = []
                b_child = []
                for i in range(N):
                    if random.random() < 0.5:
                        a_child.append(a1[i])
                        b_child.append(b1[i])
                    else:
                        a_child.append(a2[i])
                        b_child.append(b2[i])
            else:
                p = select()
                a_child, b_child = p[0][:], p[1][:]
                
            # Mutation
            # Mutate a few transitions
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                if random.random() < 0.5:
                    idx = random.randint(0, N - 1)
                    a_child[idx] = random.randint(0, N - 1)
                else:
                    idx = random.randint(0, N - 1)
                    b_child[idx] = random.randint(0, N - 1)
            
            # Repair if necessary
            if not repair(a_child, b_child):
                # If repair fails, discard or replace with a random valid one
                # To save time, just skip if it takes too long, but repair is capped.
                # If it still fails, use a random pinwheel
                a_child, b_child = generate_pinwheel_variant()
                repair(a_child, b_child)
                
            new_population.append((a_child, b_child))
            
        # Evaluate new population
        for i in range(pop_size):
            a, b = new_population[i]
            new_population[i] = (a, b) # Ensure tuple
            scores[i] = compute_reset_length(a, b)
            
        population = new_population
        
        # Update best
        current_best_idx = max(range(pop_size), key=lambda i: scores[i])
        current_best_len = scores[current_best_idx]
        
        if current_best_len > best_len:
            best_len = current_best_len
            best_a = population[current_best_idx][0][:]
            best_b = population[current_best_idx][1][:]
            no_improve_generations = 0
        else:
            no_improve_generations += 1
            # If stuck for too long, inject diversity
            if no_improve_generations > 20:
                for i in range(pop_size // 2):
                    a, b = generate_pinwheel_variant()
                    if repair(a, b):
                        population[i] = (a, b)
                        scores[i] = compute_reset_length(a, b)
                no_improve_generations = 0
                
    return [best_a, best_b]
