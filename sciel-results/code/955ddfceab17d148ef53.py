import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 82.0
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)

    def full_score(vals, edge_list):
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    def make_scoring(vals):
        sv = vals[:]
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        def add(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 0:
                if which == 's':
                    distinct_sum += 1
                else:
                    distinct_prod += 1
            cnt[x] = c + 1

        def rem(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 1:
                del cnt[x]
                if which == 's':
                    distinct_sum -= 1
                else:
                    distinct_prod -= 1
            else:
                cnt[x] = c - 1

        def swap(out, inn):
            u, v = out
            rem(sv[u] + sv[v], cnt_sum, 's')
            rem(sv[u] * sv[v], cnt_prod, 'p')
            u, v = inn
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def add_edge(e):
            u, v = e
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def raw():
            return max(distinct_sum, distinct_prod)

        def init_from(edges):
            for e in edges:
                add_edge(e)

        return init_from, swap, add_edge, raw, (cnt_sum, cnt_prod)

    def generate_value_sets():
        sets = []
        # 1. Consecutive small
        for start_v in range(1, 8):
            sets.append(list(range(start_v, start_v + n)))
        # 2. Multiples of small k
        for k in range(2, 12):
            sets.append([k * i for i in range(1, n + 1)])
        # 3. Powers
        sets.append([2**i for i in range(1, n + 1)])
        sets.append([3**i for i in range(1, n + 1)])
        # 4. Squares
        sets.append([i**2 for i in range(1, n + 1)])
        # 5. Random small
        for _ in range(30):
            vals = random.sample(range(1, 40), n)
            sets.append(vals)
        # 6. Random with structure
        for _ in range(30):
            base = random.randint(1, 5)
            vals = []
            seen = set()
            for i in range(n):
                v = base + random.randint(0, 30)
                while v in seen or v < 1:
                    v = random.randint(1, 60)
                seen.add(v)
                vals.append(v)
            sets.append(vals)
        # 7. Clustered
        for _ in range(20):
            center = random.randint(5, 30)
            spread = random.randint(2, 10)
            vals = []
            seen = set()
            for i in range(n):
                v = center + random.randint(-spread, spread)
                while v in seen or v < 1:
                    v = random.randint(1, 100)
                seen.add(v)
                vals.append(v)
            sets.append(vals)
        # 8. Dense small ranges
        for hi in (18, 22, 28, 36, 48, 64, 90):
            pool = list(range(1, hi + 1))
            if len(pool) >= n:
                sets.append(random.sample(pool, n))
        # 9. Arithmetic progressions
        for start_v in range(1, 15):
            for step in [2, 3, 4, 5]:
                pool = [start_v + i * step for i in range(30) if start_v + i * step <= 150]
                if len(pool) >= n:
                    sets.append(random.sample(pool, n))
        # 10. Composite-rich values
        composite_pool = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 49, 54, 56, 60, 64, 72, 80, 81, 90, 96, 100, 108, 120, 128, 144, 150, 160, 162, 168, 180, 196, 200]
        if len(composite_pool) >= n:
            sets.append(random.sample(composite_pool, n))
        # 11. Powers of 2 and 3 mix
        pow_pool = []
        for i in range(1, 12):
            pow_pool.append(2**i)
            pow_pool.append(3**i)
        pow_pool = [v for v in pow_pool if v <= 10000]
        if len(pow_pool) >= n:
            sets.append(random.sample(pow_pool, n))
        return sets

    def greedy_edges(vals):
        """Greedy edge selection to maximize collisions."""
        best_chosen = None
        best_score = float('inf')
        
        # Strategy 1: Sort by sum
        order1 = sorted(all_edges, key=lambda e: vals[e[0]] + vals[e[1]])
        # Strategy 2: Sort by product
        order2 = sorted(all_edges, key=lambda e: vals[e[0]] * vals[e[1]])
        # Strategy 3: Random order
        order3 = all_edges[:]
        random.shuffle(order3)
        # Strategy 4: Sort by sum then product
        order4 = sorted(all_edges, key=lambda e: (vals[e[0]] + vals[e[1]], vals[e[0]] * vals[e[1]]))
        
        for order in [order1, order2, order3, order4]:
            chosen = []
            cnt_sum = {}
            cnt_prod = {}
            distinct_sum = 0
            distinct_prod = 0
            
            for e in order:
                if len(chosen) == m:
                    break
                u, v = e
                s = vals[u] + vals[v]
                p = vals[u] * vals[v]
                
                if s not in cnt_sum:
                    distinct_sum += 1
                cnt_sum[s] = cnt_sum.get(s, 0) + 1
                
                if p not in cnt_prod:
                    distinct_prod += 1
                cnt_prod[p] = cnt_prod.get(p, 0) + 1
                
                chosen.append(e)
            
            score = max(distinct_sum, distinct_prod)
            if score < best_score:
                best_score = score
                best_chosen = chosen
        
        return best_chosen, best_score

    def polish_edges(vals, chosen, steps, deadline):
        """Local search on edges with incremental scoring."""
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(chosen)
        cur = raw()
        best = cur
        best_chosen = list(chosen)
        chosen_set = set(chosen)
        no_improve = 0
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            i = random.randrange(m)
            e_out = chosen[i]
            
            best_e_in = None
            best_r = float('inf')
            
            # Try multiple candidates
            for _ in range(8):
                e_in = random.choice(all_edges)
                if e_in in chosen_set:
                    continue
                
                swap(e_out, e_in)
                r = raw()
                
                if r < best_r:
                    best_r = r
                    best_e_in = e_in
                
                swap(e_in, e_out)
            
            if best_e_in is None:
                no_improve += 1
                if no_improve > 10000:
                    break
                continue
            
            # Apply best swap
            swap(e_out, best_e_in)
            r = raw()
            
            if r <= cur:
                cur = r
                chosen[i] = best_e_in
                chosen_set.discard(e_out)
                chosen_set.add(best_e_in)
                if r < best:
                    best = r
                    best_chosen = list(chosen)
                no_improve = 0
            else:
                swap(best_e_in, e_out)
                no_improve += 1
        
        return best_chosen, best

    def joint_search(vals0, edges0, steps, deadline):
        """Joint value and edge local search."""
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        no_improve = 0
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            if random.random() < 0.6:
                # Edge move
                i = random.randrange(m)
                e_out = edges[i]
                
                best_e_in = None
                best_r = float('inf')
                
                for _ in range(5):
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue
                    
                    swap(e_out, e_in)
                    r = raw()
                    
                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                    
                    swap(e_in, e_out)
                
                if best_e_in is None:
                    no_improve += 1
                    continue
                
                swap(e_out, best_e_in)
                r = raw()
                
                if r <= cur:
                    cur = r
                    edges[i] = best_e_in
                    edges_set.discard(e_out)
                    edges_set.add(best_e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    swap(best_e_in, e_out)
                    no_improve += 1
            else:
                # Value move
                idx = random.randrange(n)
                old = vals[idx]
                
                candidates = []
                for _ in range(3):
                    candidates.append(random.randint(1, 30))
                for _ in range(2):
                    candidates.append(random.randint(5, 50))
                for _ in range(2):
                    other = random.randrange(n)
                    if other != idx:
                        candidates.append(vals[other] + random.randint(-3, 3))
                
                best_nv = None
                best_r = float('inf')
                
                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue
                    
                    vals[idx] = nv
                    r = full_score(vals, edges)
                    
                    if r < best_r:
                        best_r = r
                        best_nv = nv
                    
                    vals[idx] = old
                
                if best_nv is None:
                    no_improve += 1
                    continue
                
                vals[idx] = best_nv
                r = best_r
                
                if r <= cur:
                    cur = r
                    init_from, swap, add_edge, raw, _ = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1
        
        return best, best_vals, best_edges

    def final_polish(vals, edges, deadline):
        """High-precision final polish with more aggressive local search."""
        vals = list(vals)
        edges = list(edges)
        edges_set = set(edges)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        no_improve = 0
        
        while time.time() < deadline:
            # Try edge swaps
            i = random.randrange(m)
            e_out = edges[i]
            
            best_e_in = None
            best_r = float('inf')
            
            for _ in range(12):
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                
                swap(e_out, e_in)
                r = raw()
                
                if r < best_r:
                    best_r = r
                    best_e_in = e_in
                
                swap(e_in, e_out)
            
            if best_e_in is not None:
                swap(e_out, best_e_in)
                r = raw()
                
                if r <= cur:
                    cur = r
                    edges[i] = best_e_in
                    edges_set.discard(e_out)
                    edges_set.add(best_e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    swap(best_e_in, e_out)
                    no_improve += 1
            else:
                no_improve += 1
            
            # Occasionally try value moves
            if random.random() < 0.3:
                idx = random.randrange(n)
                old = vals[idx]
                
                candidates = []
                for _ in range(5):
                    candidates.append(random.randint(1, 40))
                for _ in range(3):
                    other = random.randrange(n)
                    if other != idx:
                        candidates.append(vals[other] + random.randint(-5, 5))
                        candidates.append(vals[other] * random.randint(1, 2))
                
                best_nv = None
                best_r = float('inf')
                
                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue
                    
                    vals[idx] = nv
                    r = full_score(vals, edges)
                    
                    if r < best_r:
                        best_r = r
                        best_nv = nv
                    
                    vals[idx] = old
                
                if best_nv is not None:
                    vals[idx] = best_nv
                    r = best_r
                    
                    if r <= cur:
                        cur = r
                        init_from, swap, add_edge, raw, _ = make_scoring(vals)
                        init_from(edges)
                        if r < best:
                            best = r
                            best_vals = list(vals)
                            best_edges = list(edges)
                        no_improve = 0
                    else:
                        vals[idx] = old
                        no_improve += 1
            
            if no_improve > 20000:
                break
        
        return best, best_vals, best_edges

    # Main driver
    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Try structured value sets
    value_sets = generate_value_sets()
    random.shuffle(value_sets)
    
    for vals in value_sets:
        if time.time() > deadline - 10.0:
            break
        
        chosen, sc = greedy_edges(vals)
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(chosen)
        
        # Polish edges
        polished, psc = polish_edges(vals, chosen, 5000, deadline - 8.0)
        if psc < best_score:
            best_score = psc
            best_vals = list(vals)
            best_edges = list(polished)
    
    # Phase 2: Joint search from best found
    if best_vals is not None:
        sc, vals, edges = joint_search(best_vals, best_edges, 30000, deadline - 15.0)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3.0 and best_vals is not None:
        polished, psc = final_polish(best_vals, best_edges, deadline - 2.0)
        if psc < best_score:
            best_score = psc
            best_vals = polished[1]
            best_edges = polished[2]
    
    # Fallback
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)

    # Type conversion
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Validate
    seen = set()
    clean = []
    for e in final_edges:
        a, b = e
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean.append([a, b])
    
    if len(clean) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean.append([e[0], e[1]])
                if len(clean) == m:
                    break
    
    final_edges = clean[:m]
    
    return {'values': final_values, 'edges': final_edges}
