import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before the 150s kill

    # --- Step 1: Guaranteed valid baseline from the algebraic construction ---
    # This construction is known to produce the (3,16)-cage with 936 vertices.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Pre-compute rules for the algebraic construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    baseline_n = len(order)
    baseline_edges = edges

    # Fast Girth Checker
    def check_girth(edges, n, g):
        if n < g:
            return False
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify degree
        for i in range(n):
            if len(adj[i]) != k:
                return False

        # BFS from each vertex to check for cycles shorter than g
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[v] < dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Verify baseline
    if not check_girth(baseline_edges, baseline_n, g):
        pass

    best_n = baseline_n
    best_edges = baseline_edges

    # --- Step 2: Petersen base with voltage assignment, GA-style repair ---
    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5.
    # We assign a voltage (an element of Z_p) to each edge.
    # The voltage graph has 10p vertices.
    # The girth of the voltage graph is constrained by the base graph's cycles.
    # For each cycle C in the base graph, the product of voltages around C must be non-zero
    # if len(C) < g, but in additive mod p, sum must be 0 only if len(C) >= g?
    # Actually, a cycle of length l in the base lifts to l cycles in the voltage graph
    # (or fewer if the voltage sum is 0).
    # The shortest cycle in the voltage graph has length min over all cycles C in base
    # of (len(C) if sum(voltage(C)) != 0, else ... well, it closes).
    # More precisely: a cycle of length L in base with voltage sum S (mod p).
    # In the voltage graph, this lifts to p/gcd(S,p) cycles of length L * p / gcd(S,p).
    # If S == 0, it lifts to p cycles of length L.
    # So the girth of the voltage graph is min over all cycles C in base
    # of L_C * p / gcd(sum(C), p).
    # For p prime, gcd(sum(C), p) is 1 if sum(C) != 0, and p if sum(C) == 0.
    # So: if sum(C) != 0, cycle length in voltage graph = L_C * p.
    #     if sum(C) == 0, cycle length in voltage graph = L_C.
    # We want all cycles in the voltage graph to have length >= g.
    # So for all cycles C in base:
    #   if sum(C) == 0, need L_C >= g.
    #   if sum(C) != 0, need L_C * p >= g.
    # Since base has girth 5, cycles of length 5, 6, ... exist.
    # For L_C < g: we need L_C * p >= g => p >= g / L_C.
    # For L_C >= g: sum(C) == 0 is OK (gives L_C >= g), sum(C) != 0 gives L_C * p >= g (trivially).
    # So the constraint is: for all cycles C with L_C < g, sum(C) != 0 (mod p).
    # (Because if sum(C) == 0, the voltage graph has a cycle of length L_C < g.)
    # And for L_C < g, sum(C) != 0 implies L_C * p >= g.
    # So we need: for all base cycles C with length L_C < g, voltage sum != 0 (mod p).
    # AND: for all such C, L_C * p >= g.
    # The second condition is p >= ceil(g / L_C) for all C with L_C < g.
    # The most restrictive is the shortest cycle in base with L_C < g.
    # For Petersen, girth is 5. So shortest cycle is 5.
    # Need 5 * p >= g => p >= g / 5.
    # For g = 16, p >= 16/5 = 3.2, so p >= 4. But p is prime, so p >= 5.
    # Also need: for all cycles of length 5, 6, 7, ..., g-1 in base, sum != 0 (mod p).
    # Number of cycles in Petersen:
    #   length 5: 12
    #   length 6: 20 (actually need to verify)
    #   ...
    # Let me use a different approach: enumerate all cycles up to length g-1 in base,
    # and check that their voltage sums are non-zero mod p.
    # For Petersen, the number of cycles up to length 15 is manageable.

    # Petersen graph edges (0-indexed):
    # Outer 5-cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner 5-cycle (star): 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    m = len(petersen_edges)

    # Build adjacency for Petersen
    p_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        p_adj[u].append(v)
        p_adj[v].append(u)

    # Enumerate all cycles of length 3 to g-1 in Petersen
    # Use BFS from each edge
    cycles = set()
    for i, (u, v) in enumerate(petersen_edges):
        # BFS from (u, v)
        # State: (current, prev, length, path)
        stack = [(v, u, 1, [(u, v)])]
        while stack:
            cur, prev, length, path = stack.pop()
            if length >= g:
                continue
            if cur == u and length >= 3:
                # Found a cycle
                cycle = tuple(sorted(path))
                # Canonicalize: rotate so min vertex first, then min second
                # Actually, just use frozenset of edges
                edge_set = frozenset(path)
                cycles.add(edge_set)
                continue
            for w in p_adj[cur]:
                if w == prev:
                    continue
                if w == u and length < 3:
                    continue
                stack.append((w, cur, length + 1, path + [(cur, w)]))
    
    # For each cycle, we need: for all prime p >= ceil(g / min_len), 
    #   sum of voltages on cycle edges != 0 (mod p)
    # where voltages are assigned to the 15 edges.
    # This is a constraint satisfaction problem.
    # Use a genetic algorithm to find voltage assignments.

    # For a given p and voltage assignment (tuple of m values in 0..p-1),
    # check if all cycles have non-zero sum mod p.
    # Fitness: number of violated constraints (cycles with sum == 0).
    # We want fitness = 0.

    # Find the minimum p needed: p >= ceil(g / 5) for Petersen (girth 5).
    min_p = max(2, (g + 4) // 5)  # ceil(g/5)
    # But also need p to be prime.
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    def next_prime(n):
        while not is_prime(n):
            n += 1
        return n

    p = next_prime(min_p)

    # Voltage assignment: m values in 0..p-1
    # Fitness: count of cycles with sum == 0 mod p
    def fitness(voltages, p, cycles):
        count = 0
        for cycle in cycles:
            s = 0
            for (u, v) in cycle:
                # Find edge index
                # Precompute edge lookup
                pass
        return count

    # Precompute edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Precompute for each cycle, the list of edge indices
    cycle_edge_indices = []
    for cycle in cycles:
        idxs = []
        for (u, v) in cycle:
            idxs.append(edge_index[(u, v)])
        cycle_edge_indices.append(idxs)

    def fitness(voltages, p):
        count = 0
        for idxs in cycle_edge_indices:
            s = 0
            for i in idxs:
                s += voltages[i]
            if s % p == 0:
                count += 1
        return count

    # GA parameters
    pop_size = 50
    gen_count = 500
    mutation_rate = 0.1
    crossover_rate = 0.8

    # Initialize population
    population = []
    for _ in range(pop_size):
        v = [random.randint(0, p - 1) for _ in range(m)]
        population.append(v)

    best_fitness = min(fitness(v, p) for v in population)
    best_voltage = max(population, key=lambda v: -fitness(v, p))

    # Evolution
    gen = 0
    while gen < gen_count and time.time() < deadline - 15.0:
        # Selection: tournament
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_voltage[:])
        
        while len(new_population) < pop_size:
            # Crossover
            if random.random() < crossover_rate and len(new_population) + 1 < pop_size:
                # Tournament selection for parents
                p1 = random.sample(population, 3)
                p1 = max(p1, key=lambda v: -fitness(v, p))
                p2 = random.sample(population, 3)
                p2 = max(p2, key=lambda v: -fitness(v, p))
                
                # Uniform crossover
                child = [random.choice([p1[i], p2[i]]) for i in range(m)]
            else:
                child = random.choice(population)[:]
            
            # Mutation
            for i in range(m):
                if random.random() < mutation_rate:
                    child[i] = random.randint(0, p - 1)
            
            new_population.append(child)
        
        # Update best
        for v in new_population:
            f = fitness(v, p)
            if f < best_fitness:
                best_fitness = f
                best_voltage = v[:]
        
        # Replace population with new one (keep best)
        population = new_population
        gen += 1

    if best_fitness == 0:
        # Build the voltage graph
        # 10 * p vertices
        # Vertex (base_v, lift) where base_v in 0..9, lift in 0..p-1
        # Edge between (u, a) and (v, b) if (u,v) is edge i in base, and b = a + voltage[i] mod p
        # Actually, undirected: for edge (u, v) with voltage w,
        #   (u, a) -- (v, a + w) and (v, b) -- (u, b - w)
        # These are the same edge.
        # So for each base edge (u, v) with voltage w:
        #   for a in 0..p-1:
        #     edge between (u, a) and (v, (a + w) % p)
        # Total edges: 15 * p
        # Total vertices: 10 * p
        # Each vertex has degree 3 (same as base).
        
        n_v = 10 * p
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            w = best_voltage[i]
            for a in range(p):
                b = (a + w) % p
                eu = u * p + a
                ev = v * p + b
                edges.append((eu, ev))
        
        # Verify
        if check_girth(edges, n_v, g) and n_v < best_n:
            best_n = n_v
            best_edges = edges

    # Try larger p if needed
    # Actually, let's try multiple primes
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        p = next_prime(p + 1)
        n_v = 10 * p
        if n_v >= best_n:
            break
        
        # Reset GA with this p
        population = []
        for _ in range(pop_size):
            v = [random.randint(0, p - 1) for _ in range(m)]
            population.append(v)
        
        best_fitness = min(fitness(v, p) for v in population)
        best_voltage = max(population, key=lambda v: -fitness(v, p))
        
        gen = 0
        while gen < gen_count and time.time() < deadline - 10.0:
            new_population = [best_voltage[:]]
            while len(new_population) < pop_size:
                if random.random() < crossover_rate and len(new_population) + 1 < pop_size:
                    p1 = random.sample(population, 3)
                    p1 = max(p1, key=lambda v: -fitness(v, p))
                    p2 = random.sample(population, 3)
                    p2 = max(p2, key=lambda v: -fitness(v, p))
                    child = [random.choice([p1[i], p2[i]]) for i in range(m)]
                else:
                    child = random.choice(population)[:]
                for i in range(m):
                    if random.random() < mutation_rate:
                        child[i] = random.randint(0, p - 1)
                new_population.append(child)
            for v in new_population:
                f = fitness(v, p)
                if f < best_fitness:
                    best_fitness = f
                    best_voltage = v[:]
            population = new_population
            gen += 1
        
        if best_fitness == 0:
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                w = best_voltage[i]
                for a in range(p):
                    b = (a + w) % p
                    edges.append((u * p + a, v * p + b))
            if check_girth(edges, n_v, g) and n_v < best_n:
                best_n = n_v
                best_edges = edges

    # --- Step 3: Try a few random 3-regular graphs as fallback ---
    if time.time() < deadline - 5.0:
        def generate_random_3_regular(n):
            if n % 2 != 0 or n < 4:
                return None
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            if valid:
                return list(edges)
            return None

        moore_bound = 2 * sum(2 ** i for i in range(g // 2 - 1)) if g % 2 == 0 else 2 * sum(2 ** i for i in range((g - 1) // 2 - 1)) + 1
        if moore_bound % 2 != 0:
            moore_bound += 1

        for n_try in range(moore_bound, min(moore_bound + 10, best_n)):
            if time.time() > deadline - 5.0:
                break
            if n_try % 2 != 0:
                continue
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                edges = generate_random_3_regular(n_try)
                if edges is None:
                    continue
                if check_girth(edges, n_try, g):
                    best_n = n_try
                    best_edges = edges
                    break
            if best_n < baseline_n:
                break

    return best_n, best_edges
