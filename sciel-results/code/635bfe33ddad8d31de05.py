import random
import time
import math

def construct():
    deadline = time.time() + 80.0
    N = 14
    M = 35
    MAXV = 10000

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            a = values[i]
            b = values[j]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))

    def gen_value_sets():
        sets = []
        sets.append(list(range(1, 15)))
        sets.append(list(range(2, 29, 2)))
        sets.append([1 << i for i in range(14)])
        primes = [2, 3, 5, 7, 11]
        grid = []
        for p in primes:
            for q in primes:
                v = p * q
                if v <= MAXV and v not in grid:
                    grid.append(v)
        if len(grid) >= 14:
            sets.append(grid[:14])
            sets.append(random.sample(grid, 14))
        sets.append(random.sample(range(1, 100), 14))
        sets.append(random.sample(range(1, 1000), 14))
        sets.append(random.sample(range(1, MAXV + 1), 14))
        for _ in range(3):
            c = random.randint(10, 2000)
            d = random.randint(3, 30)
            vals = []
            for k in range(-d, d + 1):
                v = c + k
                if 1 <= v <= MAXV and v not in vals:
                    vals.append(v)
            while len(vals) < 14:
                v = random.randint(1, MAXV)
                if v not in vals:
                    vals.append(v)
            sets.append(vals[:14])
        return sets

    def gen_edges():
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        return [list(e) for e in all_edges[:M]]

    def neighbor(vals, edges):
        r = random.random()
        if r < 0.35:
            new_vals = vals[:]
            idx = random.randrange(N)
            new_val = random.randint(1, MAXV)
            if new_val not in new_vals:
                new_vals[idx] = new_val
            return new_vals, edges
        elif r < 0.5:
            new_vals = vals[:]
            idx = random.randrange(N)
            delta = random.choice([-1, 1]) * random.randint(1, 20)
            new_val = new_vals[idx] + delta
            if 1 <= new_val <= MAXV and new_val not in new_vals:
                new_vals[idx] = new_val
            return new_vals, edges
        elif r < 0.65:
            new_vals = vals[:]
            i, j = random.sample(range(N), 2)
            new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
            return new_vals, edges
        elif r < 0.85:
            new_edges = [e[:] for e in edges]
            ei = random.randrange(M)
            present = set(tuple(sorted(e)) for e in edges)
            all_edges = [(a, b) for a in range(N) for b in range(a + 1, N)]
            candidates = [e for e in all_edges if tuple(sorted(e)) not in present]
            if candidates:
                ne = random.choice(candidates)
                new_edges[ei] = list(ne)
            return vals, new_edges
        else:
            new_vals = vals[:]
            for _ in range(random.randint(2, 4)):
                idx = random.randrange(N)
                new_val = random.randint(1, MAXV)
                if new_val not in new_vals:
                    new_vals[idx] = new_val
            return new_vals, edges

    best = None
    best_val = float('inf')

    def try_update(vals, edges, val):
        nonlocal best, best_val
        if val < best_val:
            best_val = val
            best = (vals[:], [e[:] for e in edges])

    while time.time() < deadline:
        vals = random.choice(gen_value_sets())
        edges = gen_edges()
        cur_val = evaluate(vals, edges)
        try_update(vals, edges, cur_val)

        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        while T > T_min and time.time() < deadline:
            nvals, nedges = neighbor(vals, edges)
            nval = evaluate(nvals, nedges)
            d = nval - cur_val
            if d < 0 or random.random() < math.exp(-d / max(T, 1e-9)):
                vals, edges, cur_val = nvals, nedges, nval
                try_update(vals, edges, cur_val)
            T *= cooling

    if best is not None:
        vals, edges = best[0][:], [e[:] for e in best[1]]
        cur_val = evaluate(vals, edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(500):
                if time.time() > deadline:
                    break
                nvals, nedges = neighbor(vals, edges)
                nval = evaluate(nvals, nedges)
                if nval < cur_val:
                    vals, edges, cur_val = nvals, nedges, nval
                    improved = True
                    try_update(vals, edges, cur_val)

    vals, edges = best[0], best[1]
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    return {'values': vals, 'edges': edges}
