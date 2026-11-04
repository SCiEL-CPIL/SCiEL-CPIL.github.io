import random
import itertools
import math

def construct(k, g):
    # Try to find a Cayley graph with n < 512
    # Moore bound 243
    # Known best 320
    
    # Helper functions for abelian group operations
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def generate_random_abelian_dims(n, max_dims=4):
        factors = factorize(n)
        if not factors:
            return (1,)
        primes = list(factors.keys())
        prime_list = []
        for p, e in factors.items():
            prime_list.extend([p] * e)
        
        k = random.randint(1, min(max_dims, len(prime_list)))
        bins = [[] for _ in range(k)]
        for p in prime_list:
            bins[random.randint(0, k-1)].append(p)
        
        dims = [1] * k
        for i in range(k):
            for p in bins[i]:
                dims[i] *= p
        dims.sort()
        return tuple(dims)

    def add_g(g1, g2, dims):
        return tuple((g1[i] + g2[i]) % dims[i] for i in range(len(dims)))

    def find_cayley_graph(n, trials=200):
        for _ in range(trials):
            dims = generate_random_abelian_dims(n)
            g1 = tuple(random.randint(0, d-1) for d in dims)
            g2 = tuple(random.randint(0, d-1) for d in dims)
            
            neg_g1 = tuple((-x) % d for x, d in zip(g1, dims))
            neg_g2 = tuple((-x) % d for x, d in zip(g2, dims))
            
            gens = [g1, neg_g1, g2, neg_g2]
            if len(set(gens)) < 4:
                continue
            
            start = tuple(0 for _ in dims)
            visited = {start: None}
            queue = [start]
            depth = 0
            valid = True
            
            while queue and depth <= 4:
                current = queue.pop(0)
                for gen in gens:
                    v = add_g(current, gen, dims)
                    if v in visited:
                        parent = visited[current]
                        if parent != v and visited[v] != current:
                            valid = False
                            break
                    else:
                        if depth < 4:
                            visited[v] = current
                            queue.append(v)
                if not valid:
                    break
                depth += 1
                
            if valid:
                # Build edges
                ranges = [range(d) for d in dims]
                all_vertices = list(itertools.product(*ranges))
                vid = {v: i for i, v in enumerate(all_vertices)}
                edges = []
                for v in all_vertices:
                    for gen in gens:
                        u = add_g(v, gen, dims)
                        if vid[v] < vid[u]:
                            edges.append((vid[v], vid[u]))
                return n, edges
        return None

    # Search
    for n in range(243, 321):
        res = find_cayley_graph(n)
        if res:
            return res

    # Fallback to seed
    return _seed_construct(k, g)

def _seed_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
    return len(order), edges
