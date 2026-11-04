import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3 with genetic algorithm.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3:
        return _construct_petersen_lift_ga(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift_ga(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages.
    Uses genetic algorithm to find valid voltage assignments.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    # Build numpy matrix for vectorized evaluation
    if len(coeff_list) > 0:
        coeff_matrix = np.array(coeff_list, dtype=np.int64)
        n_constraints = len(coeff_matrix)
    else:
        coeff_matrix = np.zeros((0, 15), dtype=np.int64)
        n_constraints = 0
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = []
    candidate = 2
    while len(primes) < 100:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    def fitness(voltages, m):
        """Count number of violated constraints (zero voltage sum mod m)."""
        if n_constraints == 0:
            return 0
        sums = (coeff_matrix @ voltages) % m
        return int(np.sum(sums == 0))
    
    def ga_search(m, max_time=20.0):
        """
        Genetic algorithm to find voltage assignment with zero violations.
        Returns (voltages, fitness) or None if not found.
        """
        pop_size = 50
        elite_size = 5
        mutation_rate = 0.3
        crossover_rate = 0.7
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            vol = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
            population.append(vol)
        
        # Evaluate initial fitness
        fitnesses = [fitness(population[i], m) for i in range(pop_size)]
        
        # If any has zero fitness, return it
        for i in range(pop_size):
            if fitnesses[i] == 0:
                return population[i], 0
        
        best_fitness = min(fitnesses)
        best_vol = population[fitnesses.index(best_fitness)]
        
        gen = 0
        start_ga = time.time()
        
        while time.time() - start_ga < max_time:
            gen += 1
            
            # Sort by fitness
            sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])
            
            # Elitism: keep best individuals
            new_population = [population[sorted_idx[i]].copy() for i in range(elite_size)]
            new_fitnesses = [fitnesses[sorted_idx[i]] for i in range(elite_size)]
            
            # If best is zero, return
            if new_fitnesses[0] == 0:
                return new_population[0], 0
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def select():
                    idxs = random.sample(range(pop_size), 3)
                    best_i = min(idxs, key=lambda i: fitnesses[i])
                    return population[best_i]
                
                if random.random() < crossover_rate:
                    p1 = select()
                    p2 = select()
                    # Uniform crossover
                    mask = np.random.random(15) < 0.5
                    child = np.where(mask, p1, p2)
                else:
                    p1 = select()
                    child = p1.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    n_mut = max(1, int(15 * random.random() * 0.5))
                    mut_idx = random.sample(range(15), n_mut)
                    for idx in mut_idx:
                        child[idx] = random.randint(1, m - 1)
                
                # Ensure no zero voltages
                child = np.where(child == 0, 1, child)
                
                new_population.append(child)
                new_fitnesses.append(fitness(child, m))
            
            # Update population
            population = new_population
            fitnesses = new_fitnesses
            
            # Track best
            best_i = fitnesses.index(min(fitnesses))
            if fitnesses[best_i] < best_fitness:
                best_fitness = fitnesses[best_i]
                best_vol = population[best_i].copy()
                if best_fitness == 0:
                    return best_vol, 0
        
        return best_vol, best_fitness
    
    best_n = float('inf')
    best_edges = None
    
    # Try primes in increasing order, larger m gives more room for valid assignments
    # But n = 10*m, so smaller m is better
    for m in primes:
        if time.time() > deadline - 10:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Try GA search
        vol, fit = ga_search(m, max_time=15.0)
        
        if fit == 0:
            edges = _build_lifted_graph(pet_edges, vol.tolist(), m)
            best_n = n_candidate
            best_edges = edges
            # Found a valid graph, try to find smaller one
            continue
        
        # If GA didn't find zero, try some random restarts as backup
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
            if fitness(voltages, m) == 0:
                edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                best_n = n_candidate
                best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
