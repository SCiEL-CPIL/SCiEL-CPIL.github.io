"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    n = P['n']
    m = P['m']
    max_val = P['maximum']

    all_pairs = list(combinations(range(n), 2))
    n_pairs = len(all_pairs)  # 91

    best_score = float('inf')
    best_values = None
    best_edges = None

    def _valid_values(vals):
        if len(vals) != n:
            return False
        if len(set(vals)) != n:
            return False
        if not all(isinstance(v, int) and 1 <= v <= max_val for v in vals):
            return False
        return True

    def _valid_edges(edges):
        if len(edges) != m:
            return False
        seen = set()
        for u, v in edges:
            if not (isinstance(u, int) and isinstance(v, int)):
                return False
            if not (0 <= u < n and 0 <= v < n):
                return False
            if u == v:
                return False
            t = (min(u, v), max(u, v))
            if t in seen:
                return False
            seen.add(t)
        return True

    def gen_product_collision_values():
        """Generate values designed to create product collisions.
        Strategy: pick a base set of small primes, then create values as
        products of pairs from this set so that a*b = c*d for many pairs."""
        small_primes = [2, 3, 5, 7, 11, 13]
        vals = set()
        candidates = []
        for p1 in small_primes:
            for p2 in small_primes:
                if p1 <= p2:
                    v = p1 * p2
                    if v <= max_val:
                        candidates.append(v)
                    v2 = p1 * p2 * random.choice(small_primes)
                    if v2 <= max_val:
                        candidates.append(v2)
        random.shuffle(candidates)
        for c in candidates:
            if len(vals) >= n:
                break
            if c not in vals:
                vals.add(c)
        while len(vals) < n:
            base = random.choice(small_primes)
            v = base * random.randint(1, max_val // base)
            vals.add(v)
        return list(vals)

    def gen_geometric_values():
        """Geometric progression creates sum structure."""
        base = random.choice([2, 3, 4])
        start = random.randint(1, 10)
        vals = set()
        for i in range(n):
            v = start * (base ** i)
            if v <= max_val:
                vals.add(v)
            else:
                vals.add(start * random.randint(1, max_val // start))
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)

    def gen_factorial_values():
        """Values based on factorial-like structure for rich products."""
        vals = set()
        for a in range(0, 14):
            for b in range(0, 6):
                for c in range(0, 4):
                    v = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= v <= max_val:
                        vals.add(v)
                    if len(vals) >= n:
                        break
                if len(vals) >= n:
                    break
            if len(vals) >= n:
                break
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)

    def gen_paired_product_values():
        """Create values where specific pairs have equal products."""
        base = random.choice([6, 10, 12, 15, 20, 24, 30, 36])
        multipliers = random.sample(range(1, max_val // base + 1), n)
        vals = [base * mult for mult in multipliers]
        if len(set(vals)) < n:
            vals = random.sample(range(1, max_val + 1), n)
        return vals

    def gen_arithmetic_values():
        """Arithmetic progression for sum collisions."""
        step = random.choice([1, 2, 3, 4, 5, 6, 8, 10])
        start = random.randint(1, max(1, max_val - step * (n - 1)))
        vals = [start + i * step for i in range(n)]
        if not _valid_values(vals):
            vals = random.sample(range(1, max_val + 1), n)
        return vals

    def gen_random_values():
        return random.sample(range(1, max_val + 1), n)

    def gen_mixed_values():
        """Mix of structured and random values."""
        half = n // 2
        structured = gen_product_collision_values()[:half]
        random_part = [v for v in random.sample(range(1, max_val + 1), n - half) if v not in structured]
        while len(random_part) < n - half:
            random_part.append(random.randint(1, max_val))
        vals = structured + random_part[:n - half]
        if len(set(vals)) < n:
            vals = random.sample(range(1, max_val + 1), n)
        return vals

    def gen_edges_greedy_collision(values):
        """Select m edges that maximize sum/product collisions."""
        edge_set = set()
        edges = []
        sums_count = defaultdict(int)
        prods_count = defaultdict(int)

        pair_data = {}
        for i, j in all_pairs:
            a, b = values[i], values[j]
            s = a + b
            p = a * b
            pair_data[(i, j)] = (s, p)

        all_sums = [pair_data[e][0] for e in all_pairs]
        all_prods = [pair_data[e][1] for e in all_pairs]
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s in all_sums:
            sum_freq[s] += 1
        for p in all_prods:
            prod_freq[p] += 1

        candidates = []
        for i, j in all_pairs:
            s, p = pair_data[(i, j)]
            score = sum_freq[s] + prod_freq[p]
            candidates.append((score, i, j))

        candidates.sort(key=lambda x: -x[0])

        for score, i, j in candidates:
            if len(edges) >= m:
                break
            t = (min(i, j), max(i, j))
            if t not in edge_set:
                edge_set.add(t)
                edges.append([i, j])

        if len(edges) < m:
            remaining = [e for e in all_pairs if (min(e[0], e[1]), max(e[0], e[1])) not in edge_set]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= m:
                    break
                edges.append(list(e))
                edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

        return edges

    def local_search_values_fast(values, edges, max_iter=30):
        """Fast local search: try changing one value at a time."""
        current_score = _score(values, edges)
        values_set = set(values)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            for i in range(n):
                old_val = values[i]
                values_set.discard(old_val)

                candidates = set()
                for _ in range(10):
                    candidates.add(random.randint(1, max_val))
                for v in values:
                    for delta in [-3, -2, -1, 1, 2, 3, -5, 5]:
                        nv = v + delta
                        if 1 <= nv <= max_val:
                            candidates.add(nv)
                for factor in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                    nv = factor * random.randint(1, max_val // factor)
                    candidates.add(nv)

                best_new_val = old_val
                best_new_score = current_score

                for new_val in candidates:
                    if new_val == old_val or new_val in values_set:
                        continue
                    values[i] = new_val
                    values_set.add(new_val)
                    new_score = _score(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    values[i] = old_val
                    values_set.discard(new_val)

                if best_new_val != old_val:
                    values[i] = best_new_val
                    values_set.discard(old_val)
                    values_set.add(best_new_val)
                    current_score = best_new_score
                    improved = True
                else:
                    values_set.add(old_val)

            if not improved:
                break

        return values, current_score

    def local_search_edges_fast(values, edges, max_iter=20):
        """Fast edge local search: swap one edge at a time."""
        current_score = _score(values, edges)
        edge_set = set(tuple(sorted(e)) for e in edges)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            for idx in range(m):
                if time.time() > deadline:
                    break
                e_remove = tuple(sorted(edges[idx]))

                candidate_edges = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]

                scored = []
                for e in candidate_edges:
                    u, v = e
                    a, b = values[u], values[v]
                    s = a + b
                    p = a * b
                    coll = 0
                    for eu, ev in edges:
                        if tuple(sorted((eu, ev))) == e_remove:
                            continue
                        ea, eb = values[eu], values[ev]
                        if ea + eb == s:
                            coll += 1
                        if ea * eb == p:
                            coll += 1
                    scored.append((coll, e))

                scored.sort(key=lambda x: -x[0])
                top = [e for _, e in scored[:15]]
                random.shuffle(top)

                best_e_add = None
                best_new_score = current_score
                for e_add in top:
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _score(values, new_edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_e_add = e_add

                if best_e_add is not None:
                    edges = edges[:idx] + edges[idx+1:] + [list(best_e_add)]
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(best_e_add)))
                    current_score = best_new_score
                    improved = True

            if not improved:
                break

        return edges, current_score

    def simulated_annealing(values, edges, max_iter=80):
        """SA with joint perturbations."""
        current_score = _score(values, edges)
        best_score_sa = current_score
        best_values_sa = values.copy()
        best_edges_sa = edges.copy()

        temperature = 4.0
        cooling = 0.93

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            new_values = values.copy()
            new_edges = edges.copy()

            if random.random() < 0.6:
                i = random.randint(0, n - 1)
                old = new_values[i]
                new_val = random.randint(1, max_val)
                while new_val == old or new_val in set(new_values[:i] + new_values[i+1:]):
                    new_val = random.randint(1, max_val)
                new_values[i] = new_val

            if random.random() < 0.6:
                idx = random.randint(0, m - 1)
                edge_set = set(tuple(sorted(e)) for e in new_edges)
                available = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]
                if available:
                    new_edges[idx] = list(random.choice(available))

            new_score = _score(new_values, new_edges)
            delta = new_score - current_score

            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.1)):
                values = new_values
                edges = new_edges
                current_score = new_score

                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_values_sa = values.copy()
                    best_edges_sa = edges.copy()

            temperature *= cooling

        return best_values_sa, best_edges_sa, best_score_sa

    # Strategy list
    strategies = [
        gen_product_collision_values,
        gen_geometric_values,
        gen_factorial_values,
        gen_paired_product_values,
        gen_arithmetic_values,
        gen_random_values,
        gen_mixed_values,
        gen_product_collision_values,
        gen_factorial_values,
    ]

    # Initialize with best strategy
    for strat in strategies[:5]:
        if time.time() > deadline - 5:
            break
        cur_values = strat()
        cur_edges = gen_edges_greedy_collision(cur_values)
        cur_score = _score(cur_values, cur_edges)
        if cur_score < best_score:
            best_score = cur_score
            best_values = cur_values.copy()
            best_edges = cur_edges.copy()

    if best_score == float('inf'):
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_pairs[:m]]
        best_score = _score(best_values, best_edges)

    # Main optimization loop
    for iteration in range(50):
        if time.time() > deadline - 5:
            break

        if iteration % 8 == 0:
            strat = random.choice(strategies)
            cur_values = strat()
            cur_edges = gen_edges_greedy_collision(cur_values)
        else:
            cur_values = best_values.copy()
            cur_edges = best_edges.copy()

        for _ in range(3):
            if time.time() > deadline - 5:
                break

            cur_values, vscore = local_search_values_fast(cur_values, cur_edges, max_iter=20)

            if time.time() > deadline - 5:
                break

            cur_edges, escore = local_search_edges_fast(cur_values, cur_edges, max_iter=15)

            if time.time() > deadline - 5:
                break

            cur_score = _score(cur_values, cur_edges)
            if cur_score < best_score:
                best_score = cur_score
                best_values = cur_values.copy()
                best_edges = cur_edges.copy()

        if iteration % 4 == 0 and time.time() < deadline - 15:
            sa_values, sa_edges, sa_score = simulated_annealing(
                best_values.copy(), best_edges.copy(), max_iter=60)
            if sa_score < best_score:
                best_score = sa_score
                best_values = sa_values.copy()
                best_edges = sa_edges.copy()

        if best_score <= 14:
            break

    # Final polish
    if time.time() < deadline - 3:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            best_edges, escore = local_search_edges_fast(best_values, best_edges, max_iter=10)
            new_score = _score(best_values, best_edges)
            if new_score < best_score:
                best_score = new_score

    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)

    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
