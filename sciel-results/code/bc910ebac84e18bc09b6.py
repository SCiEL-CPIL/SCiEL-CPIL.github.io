import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)

    # Precompute for each index, which APs it belongs to
    # ap_list[i] is a list of (ap_id, other_indices_tuple)
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta(colors, idx):
        """
        Calculate change in mono count if we flip colors[idx].
        """
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def local_search(colors):
        """
        Deterministic local search: repeatedly find the best single flip that reduces the count.
        Continue until no improvement is possible.
        """
        current_count = count_mono(colors)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_idx = -1
            
            for idx in range(N):
                d = get_delta(colors, idx)
                if d < best_delta:
                    best_delta = d
                    best_idx = idx
            
            if best_idx != -1:
                colors[best_idx] = 1 - colors[best_idx]
                current_count += best_delta
                improved = True
        return colors, current_count

    # Genetic Algorithm Setup
    pop_size = 40
    mutation_rate = 0.05
    
    # Initialize population with random colorings
    population = []
    for _ in range(pop_size):
        colors = [random.randint(0, 1) for _ in range(N)]
        colors, count = local_search(colors)
        population.append((count, colors))
    
    # Sort by count
    population.sort(key=lambda x: x[0])
    
    best_count = population[0][0]
    best_colors = population[0][1][:]
    
    # GA Loop
    while time.time() < deadline - 2.0:
        # Selection: Tournament selection
        def select_tournament():
            # Pick 3 random individuals, return the best
            i1 = random.randint(0, len(population)-1)
            i2 = random.randint(0, len(population)-1)
            i3 = random.randint(0, len(population)-1)
            
            best = population[i1]
            if population[i2][0] < best[0]:
                best = population[i2]
            if population[i3][0] < best[0]:
                best = population[i3]
            return best[1]

        # Generate new population
        new_population = []
        
        # Elitism: Keep the best 20%
        elite_size = pop_size // 5
        for i in range(elite_size):
            new_population.append(population[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = select_tournament()
            parent2 = select_tournament()
            
            # Uniform Crossover
            child = [0] * N
            for i in range(N):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Local Search Refinement
            child, count = local_search(child)
            new_population.append((count, child))
        
        # Sort new population
        new_population.sort(key=lambda x: x[0])
        
        # Update best
        if new_population[0][0] < best_count:
            best_count = new_population[0][0]
            best_colors = new_population[0][1][:]
        
        population = new_population
        
        # Early stopping if we find a very good solution
        if best_count <= 10:
            break

    return best_colors
