"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    return d1 == d2 or d1 == d3 or d2 == d3


def construct():
    """
    Find a minimal maximal no-isosceles set in [0, N)^2 using Genetic Algorithm.
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute isosceles triangles and coverers
    # triangle_list: list of (i, j, k)
    # coverers[v]: list of (u, w) such that v,u,w is isosceles
    triangle_list = []
    coverers = [[] for _ in range(num_points)]
    
    # To avoid duplicates, we can just iterate all triples? 
    # O(N^6) is too slow.
    # Iterate pairs (u, w) and find v?
    # Or iterate v and pairs (u, w)?
    
    # Efficient way:
    # For each pair (u, w), find all v such that triangle v,u,w is isosceles.
    # There are O(N^2) pairs. For each pair, finding v is O(N^2). Total O(N^4).
    # N=16, N^4 = 65536. Very fast.
    
    for u in range(num_points):
        for w in range(u + 1, num_points):
            # Find all v such that v,u,w is isosceles
            # Isosceles conditions:
            # 1. dist(v,u) == dist(v,w)  -> v on perpendicular bisector of uw
            # 2. dist(v,u) == dist(u,w)  -> v on circle centered u radius dist(u,w)
            # 3. dist(v,w) == dist(u,w)  -> v on circle centered w radius dist(u,w)
            
            duw2 = _dist2(points[u], points[w])
            
            # We can just check all v
            for v in range(num_points):
                if v == u or v == w:
                    continue
                if _is_isosceles(points[v], points[u], points[w]):
                    triangle_list.append((v, u, w))
                    coverers[v].append((u, w))
                    # Note: We might add duplicates if we're not careful, but 
                    # since we iterate unique pairs (u,w) and unique v, 
                    # each triangle (v,u,w) is added once per permutation?
                    # No, (v,u,w) is unique for this loop iteration.
                    # But isosceles triangles are symmetric. 
                    # Does it matter? 
                    # For fitness, we just need to know if ANY triangle is fully in S.
                    # Duplicates just increase the penalty weight slightly, which is fine.
                    # For coverers, duplicates mean we check the same pair multiple times, 
                    # which is just a slight performance hit.
    
    # Remove duplicates from triangle_list and coverers to be safe/clean
    triangle_list = list(set(triangle_list))
    for i in range(num_points):
        coverers[i] = list(set(coverers[i]))
        
    def get_fitness(mask):
        # mask is an integer
        p_iso = 0
        for i, j, k in triangle_list:
            if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
                p_iso += 1
                
        p_max = 0
        for v in range(num_points):
            if (mask >> v) & 1:
                continue # v is in S, doesn't count for maximality check
            blocked = False
            for u, w in coverers[v]:
                if ((mask >> u) & 1) and ((mask >> w) & 1):
                    blocked = True
                    break
            if not blocked:
                p_max += 1
                
        size = bin(mask).count('1')
        # We want to minimize size, but penalize violations heavily
        # If p_iso > 0, it's not a valid no-isosceles set.
        # If p_max > 0, it's not maximal.
        fitness = size + 10000 * p_iso + 100 * p_max
        return fitness, p_iso, p_max

    # GA Parameters
    pop_size = 50
    generations = 200
    mutation_rate = 0.1
    tournament_size = 5
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        # Random subset
        mask = 0
        for i in range(num_points):
            if random.random() < 0.5:
                mask |= (1 << i)
        population.append(mask)
        
    # Also add a few heuristic seeds
    # Greedy seed
    greedy_mask = 0
    random.seed(42)
    pts_shuffled = list(range(num_points))
    random.shuffle(pts_shuffled)
    for p in pts_shuffled:
        # Check if p can be added
        can_add = True
        for i, j, k in triangle_list:
            # If triangle involves p, check if other 2 are in greedy_mask
            if p == i:
                if (greedy_mask >> j) & 1 and (greedy_mask >> k) & 1:
                    can_add = False
                    break
            elif p == j:
                if (greedy_mask >> i) & 1 and (greedy_mask >> k) & 1:
                    can_add = False
                    break
            elif p == k:
                if (greedy_mask >> i) & 1 and (greedy_mask >> j) & 1:
                    can_add = False
                    break
        if can_add:
            greedy_mask |= (1 << p)
    population.append(greedy_mask)
    
    best_mask = 0
    best_fitness = float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = []
        for m in population:
            f, p_iso, p_max = get_fitness(m)
            fitnesses.append(f)
            if f < best_fitness:
                best_fitness = f
                best_mask = m
                # If we found a valid maximal set, we can try to improve size
                if p_iso == 0 and p_max == 0:
                    # It's a valid solution. We want to minimize size.
                    # The fitness function already handles this.
                    pass
        
        # Selection
        new_population = []
        # Elitism: keep best 2
        indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
        for i in indices[:2]:
            new_population.append(population[i])
            
        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                tournament = random.sample(range(pop_size), tournament_size)
                winner = min(tournament, key=lambda i: fitnesses[i])
                parents.append(population[winner])
                
            # Crossover
            child = 0
            for i in range(num_points):
                if random.random() < 0.5:
                    child |= (parents[0] >> i) & 1
                else:
                    child |= (parents[1] >> i) & 1
            child = child # Ensure int
            
            # Mutation
            for i in range(num_points):
                if random.random() < mutation_rate:
                    child ^= (1 << i)
                    
            new_population.append(child)
            
        population = new_population
        
    # Convert best_mask to list of points
    result = []
    for i in range(num_points):
        if (best_mask >> i) & 1:
            result.append(points[i])
            
    return result
# EVOLVE-BLOCK-END
