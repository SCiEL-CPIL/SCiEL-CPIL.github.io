import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    target_n = 34
    
    # Precompute triples and their indices
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4s = list(combinations(range(target_n), 4))
    k5s = list(combinations(range(target_n), 5))
    
    # Build reverse mappings: triple_idx -> list of k4_indices, list of k5_indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for k4_idx, k4 in enumerate(k4s):
        t1 = (k4[0], k4[1], k4[2])
        t2 = (k4[0], k4[1], k4[3])
        t3 = (k4[0], k4[2], k4[3])
        t4 = (k4[1], k4[2], k4[3])
        triple_to_k4s[triple_to_idx[t1]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t2]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t3]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t4]].append(k4_idx)
        
    for k5_idx, k5 in enumerate(k5s):
        v = list(k5)
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = (v[i], v[j], v[k])
                    triple_to_k5s[triple_to_idx[t]].append(k5_idx)

    # Helper to initialize counts
    def init_counts(colors):
        k4_red_counts = [0] * len(k4s)
        k5_blue_counts = [0] * len(k5s)
        
        for k4_idx, k4 in enumerate(k4s):
            cnt = 0
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0: cnt += 1
            if colors[triple_to_idx[t2]] == 0: cnt += 1
            if colors[triple_to_idx[t3]] == 0: cnt += 1
            if colors[triple_to_idx[t4]] == 0: cnt += 1
            k4_red_counts[k4_idx] = cnt
            
        for k5_idx, k5 in enumerate(k5s):
            cnt = 0
            v = list(k5)
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] == 1: cnt += 1
            k5_blue_counts[k5_idx] = cnt
            
        return k4_red_counts, k5_blue_counts

    # Helper to calculate violations
    def get_violations(k4_red_counts, k5_blue_counts):
        v4 = sum(1 for c in k4_red_counts if c == 4)
        v5 = sum(1 for c in k5_blue_counts if c == 10)
        return v4 + v5

    # Repair Heuristic: Try to fix violations by flipping edges in violating cliques
    def repair(colors, k4_red_counts, k5_blue_counts, max_iters=5000):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            v4 = sum(1 for c in k4_red_counts if c == 4)
            v5 = sum(1 for c in k5_blue_counts if c == 10)
            if v4 == 0 and v5 == 0:
                return colors, k4_red_counts, k5_blue_counts, True

            # Pick a random violation
            if v4 > 0 and (v5 == 0 or random.random() < v4 / (v4 + v5)):
                # Find a red K4
                k4_idx = random.randint(0, len(k4s) - 1)
                while k4_red_counts[k4_idx] != 4:
                    k4_idx = random.randint(0, len(k4s) - 1)
                # Flip a random edge in this K4 to Blue
                k4 = k4s[k4_idx]
                edges = [
                    (k4[0], k4[1], k4[2]),
                    (k4[0], k4[1], k4[3]),
                    (k4[0], k4[2], k4[3]),
                    (k4[1], k4[2], k4[3])
                ]
                t = random.choice(edges)
                i = triple_to_idx[t]
                
                # Flip Red -> Blue
                for k4_i in triple_to_k4s[i]:
                    k4_red_counts[k4_i] -= 1
                for k5_i in triple_to_k5s[i]:
                    k5_blue_counts[k5_i] += 1
                colors[i] = 1
            else:
                # Find a blue K5
                k5_idx = random.randint(0, len(k5s) - 1)
                while k5_blue_counts[k5_idx] != 10:
                    k5_idx = random.randint(0, len(k5s) - 1)
                # Flip a random edge in this K5 to Red
                k5 = k5s[k5_idx]
                v = list(k5)
                edges = [(v[i], v[j], v[k]) for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
                t = random.choice(edges)
                i = triple_to_idx[t]
                
                # Flip Blue -> Red
                for k4_i in triple_to_k4s[i]:
                    k4_red_counts[k4_i] += 1
                for k5_i in triple_to_k5s[i]:
                    k5_blue_counts[k5_i] -= 1
                colors[i] = 0
                
        return colors, k4_red_counts, k5_blue_counts, False

    # Genetic Algorithm Parameters
    pop_size = 50
    best_violations = float('inf')
    best_colors = None
    best_counts = None
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        k4_red_counts, k5_blue_counts = init_counts(colors)
        pop_v = get_violations(k4_red_counts, k5_blue_counts)
        population.append((pop_v, colors, k4_red_counts, k5_blue_counts))
        
        if pop_v == 0:
            return {'n': target_n, 'colors': colors}
            
        if pop_v < best_violations:
            best_violations = pop_v
            best_colors = colors.copy()
            best_counts = (k4_red_counts.copy(), k5_blue_counts.copy())

    # Main GA Loop
    while time.time() < deadline - 10:
        # Selection: Keep top half, replace bottom half with offspring
        population.sort(key=lambda x: x[0])
        
        new_population = population[:pop_size // 2]
        
        while len(new_population) < pop_size and time.time() < deadline - 5:
            # Tournament Selection
            def select():
                a, b = random.sample(range(len(population)), 2)
                return population[a] if population[a][0] < population[b][0] else population[b]
            
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform
            child_colors = [p1[1][i] if random.random() < 0.5 else p2[1][i] for i in range(num_triples)]
            
            # Mutation: Flip a few bits
            for _ in range(5):
                i = random.randint(0, num_triples - 1)
                child_colors[i] = 1 - child_colors[i]
            
            # Initialize counts for child
            k4_red_counts, k5_blue_counts = init_counts(child_colors)
            
            # Repair
            child_colors, k4_red_counts, k5_blue_counts, solved = repair(child_colors, k4_red_counts, k5_blue_counts, max_iters=2000)
            
            if solved:
                return {'n': target_n, 'colors': child_colors}
                
            child_v = get_violations(k4_red_counts, k5_blue_counts)
            new_population.append((child_v, child_colors, k4_red_counts, k5_blue_counts))
            
            if child_v < best_violations:
                best_violations = child_v
                best_colors = child_colors.copy()
                best_counts = (k4_red_counts.copy(), k5_blue_counts.copy())
                
        population = new_population

    # Final check on best found
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
        
    # Fallback to seed
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    target_n = 34
    
    # Precompute triples and their indices
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4s = list(combinations(range(target_n), 4))
    k5s = list(combinations(range(target_n), 5))
    
    # Build reverse mappings: triple_idx -> list of k4_indices, list of k5_indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for k4_idx, k4 in enumerate(k4s):
        t1 = (k4[0], k4[1], k4[2])
        t2 = (k4[0], k4[1], k4[3])
        t3 = (k4[0], k4[2], k4[3])
        t4 = (k4[1], k4[2], k4[3])
        triple_to_k4s[triple_to_idx[t1]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t2]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t3]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t4]].append(k4_idx)
        
    for k5_idx, k5 in enumerate(k5s):
        v = list(k5)
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = (v[i], v[j], v[k])
                    triple_to_k5s[triple_to_idx[t]].append(k5_idx)

    # Helper to initialize counts
    def init_counts(colors):
        k4_red_counts = [0] * len(k4s)
        k5_blue_counts = [0] * len(k5s)
        
        for k4_idx, k4 in enumerate(k4s):
            cnt = 0
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0: cnt += 1
            if colors[triple_to_idx[t2]] == 0: cnt += 1
            if colors[triple_to_idx[t3]] == 0: cnt += 1
            if colors[triple_to_idx[t4]] == 0: cnt += 1
            k4_red_counts[k4_idx] = cnt
            
        for k5_idx, k5 in enumerate(k5s):
            cnt = 0
            v = list(k5)
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] == 1: cnt += 1
            k5_blue_counts[k5_idx] = cnt
            
        return k4_red_counts, k5_blue_counts

    # Helper to calculate violations
    def get_violations(k4_red_counts, k5_blue_counts):
        v4 = sum(1 for c in k4_red_counts if c == 4)
        v5 = sum(1 for c in k5_blue_counts if c == 10)
        return v4 + v5

    # Repair Heuristic: Try to fix violations by flipping edges in violating cliques
    def repair(colors, k4_red_counts, k5_blue_counts, max_iters=5000):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            v4 = sum(1 for c in k4_red_counts if c == 4)
            v5 = sum(1 for c in k5_blue_counts if c == 10)
            if v4 == 0 and v5 == 0:
                return colors, k4_red_counts, k5_blue_counts, True

            # Pick a random violation
            if v4 > 0 and (v5 == 0 or random.random() < v4 / (v4 + v5)):
                # Find a red K4
                k4_idx = random.randint(0, len(k4s) - 1)
                while k4_red_counts[k4_idx] != 4:
                    k4_idx = random.randint(0, len(k4s) - 1)
                # Flip a random edge in this K4 to Blue
                k4 = k4s[k4_idx]
                edges = [
                    (k4[0], k4[1], k4[2]),
                    (k4[0], k4[1], k4[3]),
                    (k4[0], k4[2], k4[3]),
                    (k4[1], k4[2], k4[3])
                ]
                t = random.choice(edges)
                i = triple_to_idx[t]
                
                # Flip Red -> Blue
                for k4_i in triple_to_k4s[i]:
                    k4_red_counts[k4_i] -= 1
                for k5_i in triple_to_k5s[i]:
                    k5_blue_counts[k5_i] += 1
                colors[i] = 1
            else:
                # Find a blue K5
                k5_idx = random.randint(0, len(k5s) - 1)
                while k5_blue_counts[k5_idx] != 10:
                    k5_idx = random.randint(0, len(k5s) - 1)
                # Flip a random edge in this K5 to Red
                k5 = k5s[k5_idx]
                v = list(k5)
                edges = [(v[i], v[j], v[k]) for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
                t = random.choice(edges)
                i = triple_to_idx[t]
                
                # Flip Blue -> Red
                for k4_i in triple_to_k4s[i]:
                    k4_red_counts[k4_i] += 1
                for k5_i in triple_to_k5s[i]:
                    k5_blue_counts[k5_i] -= 1
                colors[i] = 0
                
        return colors, k4_red_counts, k5_blue_counts, False

    # Genetic Algorithm Parameters
    pop_size = 50
    best_violations = float('inf')
    best_colors = None
    best_counts = None
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        k4_red_counts, k5_blue_counts = init_counts(colors)
        pop_v = get_violations(k4_red_counts, k5_blue_counts)
        population.append((pop_v, colors, k4_red_counts, k5_blue_counts))
        
        if pop_v == 0:
            return {'n': target_n, 'colors': colors}
            
        if pop_v < best_violations:
            best_violations = pop_v
            best_colors = colors.copy()
            best_counts = (k4_red_counts.copy(), k5_blue_counts.copy())

    # Main GA Loop
    while time.time() < deadline - 10:
        # Selection: Keep top half, replace bottom half with offspring
        population.sort(key=lambda x: x[0])
        
        new_population = population[:pop_size // 2]
        
        while len(new_population) < pop_size and time.time() < deadline - 5:
            # Tournament Selection
            def select():
                a, b = random.sample(range(len(population)), 2)
                return population[a] if population[a][0] < population[b][0] else population[b]
            
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform
            child_colors = [p1[1][i] if random.random() < 0.5 else p2[1][i] for i in range(num_triples)]
            
            # Mutation: Flip a few bits
            for _ in range(5):
                i = random.randint(0, num_triples - 1)
                child_colors[i] = 1 - child_colors[i]
            
            # Initialize counts for child
            k4_red_counts, k5_blue_counts = init_counts(child_colors)
            
            # Repair
            child_colors, k4_red_counts, k5_blue_counts, solved = repair(child_colors, k4_red_counts, k5_blue_counts, max_iters=2000)
            
            if solved:
                return {'n': target_n, 'colors': child_colors}
                
            child_v = get_violations(k4_red_counts, k5_blue_counts)
            new_population.append((child_v, child_colors, k4_red_counts, k5_blue_counts))
            
            if child_v < best_violations:
                best_violations = child_v
                best_colors = child_colors.copy()
                best_counts = (k4_red_counts.copy(), k5_blue_counts.copy())
                
        population = new_population

    # Final check on best found
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
        
    # Fallback to seed
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
# EVOLVE-BLOCK-END
