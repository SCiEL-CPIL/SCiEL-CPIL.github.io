"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Primary: Paley tournament on p=67 (known valid).
    Secondary: attempts a genetic algorithm search for smaller n (31-66) within time budget.
    """
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill

    # 1. Precompute the known valid Paley tournament on 67 vertices
    p_best = 67
    squares_best = [False] * p_best
    for x in range(1, p_best):
        squares_best[(x * x) % p_best] = True

    orientation_best = []
    for i in range(p_best):
        for j in range(i + 1, p_best):
            diff = (j - i) % p_best
            if squares_best[diff]:
                orientation_best.append(0)
            else:
                orientation_best.append(1)

    best_result = {'n': p_best, 'orientation': orientation_best}

    # 2. Check time remaining
    if time.time() - start_time > deadline - 5:
        return best_result

    # 3. Attempt search for smaller n using efficient numpy-based evaluation
    
    def build_adj_matrix(n, orientation):
        """Build adjacency matrix from orientation list. adj[i,j]=1 means i->j."""
        adj = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1  # i -> j
                else:
                    adj[j, i] = 1  # j -> i
                idx += 1
        return adj

    def count_uncovered_fast(n, adj):
        """
        Count 4-sets without a common dominator using numpy.
        A vertex v dominates set {a,b,c,d} if v has edges to all 4.
        For a 4-set S, it's uncovered if no vertex dominates it.
        Equivalently: for each 4-set, check if intersection of in-neighborhoods is empty.
        
        Alternative: for each vertex v, count 4-sets it dominates. A 4-set is covered
        if at least one vertex dominates it. Use inclusion-exclusion or just check directly.
        
        More efficient: For each 4-set {a,b,c,d}, it has a common dominator iff
        there exists v such that adj[v,a]=adj[v,b]=adj[v,c]=adj[v,d]=1.
        
        We can precompute for each pair (a,b): the set of vertices that dominate both a and b.
        Then for 4-set {a,b,c,d}: check if any vertex dominates all four.
        
        Using bitsets: represent in-neighborhood of each vertex as a Python int bitmask.
        """
        # Build in-neighborhood bitmasks
        in_neigh = [0] * n
        for v in range(n):
            mask = 0
            for u in range(n):
                if adj[u, v] == 1:  # u -> v
                    mask |= (1 << u)
            in_neigh[v] = mask
        
        uncovered = 0
        for a in range(n):
            for b in range(a + 1, n):
                ab = in_neigh[a] & in_neigh[b]
                if ab == 0:
                    # No vertex dominates both a and b, so any 4-set containing a,b is uncovered
                    # Count remaining pairs (c,d) with c,d > b
                    rem = n - b - 1
                    uncovered += rem * (rem - 1) // 2
                    continue
                for c in range(b + 1, n):
                    abc = ab & in_neigh[c]
                    if abc == 0:
                        rem = n - c - 1
                        uncovered += rem
                        continue
                    for d in range(c + 1, n):
                        if (abc & in_neigh[d]) == 0:
                            uncovered += 1
        return uncovered

    def random_orientation(n):
        m = n * (n - 1) // 2
        return list(np.random.randint(0, 2, size=m))

    def flip_bits(orientation, n, num_flips):
        new_orient = orientation[:]
        m = n * (n - 1) // 2
        for _ in range(num_flips):
            idx = random.randint(0, m - 1)
            new_orient[idx] = 1 - new_orient[idx]
        return new_orient

    # Search from smaller n to larger, but focus effort where it's most likely to succeed
    # Lower bound is 31, upper is 66. Start from higher values where solutions are denser.
    # Actually, let's try a range of n values with limited time each.
    
    # Allocate time budget per n value
    # We'll try n values from 50 down to 35, spending more time on mid-range values.
    
    n_values = list(range(55, 30, -1))  # 55, 54, ..., 31
    
    for n in n_values:
        if time.time() - start_time > deadline - 3:
            break
        
        # Time budget for this n: proportional to how close to 67 (harder for smaller n)
        # Give more time to n closer to 67
        time_budget = min(5.0, max(0.5, (n - 30) * 0.15))
        n_start = time.time()
        
        pop_size = 50
        max_gens = 100
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            orient = random_orientation(n)
            adj = build_adj_matrix(n, orient)
            fitness = count_uncovered_fast(n, adj)
            population.append((fitness, orient))
        
        population.sort(key=lambda x: x[0])
        
        if population[0][0] == 0:
            return {'n': n, 'orientation': population[0][1]}
        
        best_fitness = population[0][0]
        
        for g in range(max_gens):
            if time.time() - n_start > time_budget:
                break
            if time.time() - start_time > deadline - 3:
                break
            
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = max(2, pop_size // 5)
            for i in range(elite_count):
                new_population.append(population[i])
            
            # Fill rest with mutated offspring
            while len(new_population) < pop_size:
                # Tournament selection: pick 3 random, take best
                candidates = random.sample(range(pop_size), min(3, pop_size))
                parent_idx = min(candidates, key=lambda i: population[i][0])
                parent_orient = population[parent_idx][1]
                
                # Mutation: flip 1-4 bits
                num_flips = random.randint(1, 4)
                child_orient = flip_bits(parent_orient, n, num_flips)
                
                # Evaluate
                adj = build_adj_matrix(n, child_orient)
                fitness = count_uncovered_fast(n, adj)
                
                new_population.append((fitness, child_orient))
                
                if fitness == 0:
                    return {'n': n, 'orientation': child_orient}
            
            population = new_population
            population.sort(key=lambda x: x[0])
            
            if population[0][0] == 0:
                return {'n': n, 'orientation': population[0][1]}
            
            # If we've made no progress in several generations, inject some diversity
            if g > 0 and g % 20 == 0 and population[0][0] == best_fitness:
                # Add some random individuals
                for _ in range(pop_size // 4):
                    orient = random_orientation(n)
                    adj = build_adj_matrix(n, orient)
                    fitness = count_uncovered_fast(n, adj)
                    population.append((fitness, orient))
                population.sort(key=lambda x: x[0])
                population = population[:pop_size]
                best_fitness = population[0][0]

    # If search didn't find anything better, return the best known (67)
    return best_result

# EVOLVE-BLOCK-END
