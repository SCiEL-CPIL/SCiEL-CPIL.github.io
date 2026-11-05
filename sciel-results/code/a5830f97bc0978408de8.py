import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def can_add(S, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S:
            if (a + x) in S or (x - a) in S:
                return False
        if (2 * x) in S:
            return False
        return True

    def build_conflict_graph(V_set):
        """Precompute: for each x in V, which other elements in V conflict with x."""
        # conflict[x] = set of y in V such that x and y cannot both be in a sum-free set
        # x and y conflict if: x+y in V, x-y in V, or y-x in V (i.e., some pair sums to a third)
        # More precisely: x,y,z in V with x+y=z means x conflicts with y (if z also present)
        # Actually: x and y are in conflict if there exists z in V such that x+y=z,
        # OR if x and z are in conflict because y+z=x, etc.
        # Simpler: x and y conflict if x+y in V, or x-y in V, or y-x in V
        # Wait, that's not quite right either. Let me think...
        # a+b=c is a violation. So if we have a,b,c in S, that's bad.
        # x and y conflict if: (x+y) in V, or (x-y) in V, or (y-x) in V
        # Because if x,y,x+y all in S -> violation. If x,y,x-y in S -> x = y+(x-y) violation.
        # Actually: if x,y in S and x+y in S, violation. If x,y in S and x-y in S, then x = y + (x-y), violation.
        # So x and y are "in tension" if x+y in V or x-y in V or y-x in V.
        # But we only care about conflicts within the candidate set V.
        conflict = {}
        for x in V_set:
            c = set()
            for y in V_set:
                if x == y:
                    if 2 * x in V_set:
                        c.add(y)
                else:
                    if (x + y) in V_set or (x - y) in V_set or (y - x) in V_set:
                        c.add(y)
            conflict[x] = c
        return conflict

    def conflict_count_fast(conflict, S, x):
        """Count how many elements in S conflict with x using precomputed graph."""
        if x not in conflict:
            return len(S)  # shouldn't happen
        return len(conflict[x] & S)

    def greedy_from_order(order, V_set, S0=None):
        S = set(S0) if S0 is not None else set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def try_add_remaining(S, V_set, deadline, conflict=None):
        added = True
        while added and time.time() < deadline:
            added = False
            remaining = [x for x in V_set if x not in S]
            if not remaining:
                break
            if conflict is not None:
                remaining.sort(key=lambda x: (conflict_count_fast(conflict, S, x), abs(x), x))
            else:
                remaining.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            for x in remaining:
                if time.time() >= deadline:
                    break
                if x not in S and can_add(S, x):
                    S.add(x)
                    added = True
        return S

    def greedy_with_priority(V_set, S0=None, max_passes=3, conflict=None):
        """Greedy that repeatedly picks the element with fewest conflicts."""
        S = set(S0) if S0 is not None else set()
        candidates = [x for x in V_set if x not in S]
        for _ in range(max_passes):
            if not candidates:
                break
            if conflict is not None:
                candidates.sort(key=lambda x: (conflict_count_fast(conflict, S, x), abs(x), x))
            else:
                candidates.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            added_any = False
            new_candidates = []
            for x in candidates:
                if x not in S and can_add(S, x):
                    S.add(x)
                    added_any = True
                elif x not in S:
                    new_candidates.append(x)
            candidates = new_candidates
            if not added_any:
                break
        return S

    def is_sum_free(S):
        s = set(S)
        L = list(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(len(L)):
                b = L[j]
                if (a + b) in s:
                    return False
        return True

    def local_search_1opt(best, V_set, budget, rng, conflict=None):
        """1-opt: remove one element, then greedily fill to maximize size."""
        t_start = time.time()
        improved = True
        best_size = len(best)
        while improved and time.time() - t_start < budget:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for a in S_list:
                if time.time() - t_start > budget:
                    break
                trial = set(best)
                trial.discard(a)
                # try to fill with conflict-aware ordering
                remaining = [x for x in V_set if x not in trial]
                if conflict is not None:
                    remaining.sort(key=lambda x: (conflict_count_fast(conflict, trial, x), abs(x), x))
                else:
                    remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                for x in remaining:
                    if x not in trial and can_add(trial, x):
                        trial.add(x)
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
                    improved = True
                    break
        return best

    def local_search_2opt(best, V_set, budget, rng, conflict=None):
        """2-opt: remove two elements, then greedily fill to maximize size."""
        t_start = time.time()
        improved = True
        best_size = len(best)
        while improved and time.time() - t_start < budget:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for i in range(len(S_list)):
                if time.time() - t_start > budget:
                    break
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() - t_start > budget:
                        break
                    b = S_list[j]
                    trial = set(best)
                    trial.discard(a)
                    trial.discard(b)
                    remaining = [x for x in V_set if x not in trial]
                    if conflict is not None:
                        remaining.sort(key=lambda x: (conflict_count_fast(conflict, trial, x), abs(x), x))
                    else:
                        remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                    for x in remaining:
                        if x not in trial and can_add(trial, x):
                            trial.add(x)
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
        return best

    def hypergraph_greedy(V_set, seed_off, budget):
        """
        Model as max independent set on 3-uniform hypergraph.
        Each hyperedge (a,b,c) with a+b=c. We want max subset with no hyperedge fully contained.
        Strategy: iteratively remove the vertex that is in the most hyperedges,
        but also consider the "potential" gain.
        """
        t_start = time.time()
        rng = random.Random(seed_off)
        
        # Build hyperedges
        V_list = list(V_set)
        V_set_local = set(V_list)
        hyperedges = []
        for i, a in enumerate(V_list):
            for j, b in enumerate(V_list):
                c = a + b
                if c in V_set_local:
                    hyperedges.append((a, b, c))
        
        # Count degree of each vertex (how many hyperedges it participates in)
        degree = {x: 0 for x in V_list}
        for (a, b, c) in hyperedges:
            degree[a] += 1
            degree[b] += 1
            degree[c] += 1
        
        # Greedy: start with all vertices, remove the one with highest degree
        # but only if removing it "frees up" more than it costs
        active = set(V_list)
        remaining_edges = list(hyperedges)
        
        # Actually, let's try a different approach: start empty, add vertices
        # in order of lowest degree (fewest conflicts)
        # This is similar to what we already do, but let's try removing high-degree vertices first
        
        # Alternative: start with all, iteratively remove vertex that is in most remaining hyperedges
        # until no hyperedge is fully in the set
        S = set(V_list)
        # Find all violated hyperedges
        violated = [(a, b, c) for (a, b, c) in hyperedges if a in S and b in S and c in S]
        
        while violated and time.time() - t_start < budget * 0.5:
            # For each vertex in violated edges, count how many violated edges it's in
            edge_count = {x: 0 for x in S}
            for (a, b, c) in violated:
                edge_count[a] = edge_count.get(a, 0) + 1
                edge_count[b] = edge_count.get(b, 0) + 1
                edge_count[c] = edge_count.get(c, 0) + 1
            
            # Remove the vertex with highest edge count
            max_count = -1
            remove_x = None
            for x in S:
                cnt = edge_count.get(x, 0)
                if cnt > max_count:
                    max_count = cnt
                    remove_x = x
            
            if remove_x is None or max_count == 0:
                break
            
            S.discard(remove_x)
            # Recompute violated
            violated = [(a, b, c) for (a, b, c) in hyperedges if a in S and b in S and c in S]
        
        return S

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        best = set()
        best_size = 0
        rng = random.Random(seed_off)

        # Precompute conflict graph for speed
        conflict = build_conflict_graph(V_set)

        # Candidate structural orders
        orders = []

        # A. By value descending (largest first) — upper interval
        orders.append(sorted(V, key=lambda x: x, reverse=True))
        # B. By value ascending
        orders.append(sorted(V, key=lambda x: x))
        # C. By |x| descending
        orders.append(sorted(V, key=lambda x: abs(x), reverse=True))
        # D. Odds first, then evens by |x|
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        orders.append(sorted(odds, key=lambda x: abs(x), reverse=True) + sorted(evens, key=lambda x: abs(x), reverse=True))
        # E. Positives by value desc, then negatives by value asc (most neg)
        pos = sorted([x for x in V if x > 0], reverse=True)
        neg = sorted([x for x in V if x < 0])
        orders.append(pos + neg)
        # F. Negatives first (most negative), then positives
        neg2 = sorted([x for x in V if x < 0], reverse=True)
        pos2 = sorted([x for x in V if x > 0])
        orders.append(neg2 + pos2)
        # G. Upper value interval only: top 60% by value
        sv = sorted(V)
        k = int(n * 0.6)
        orders.append(sv[n-k:])
        # H. Top 60% by |x|
        sm = sorted(V, key=lambda x: abs(x), reverse=True)
        orders.append(sm[:k])
        # I. Random orders
        for _ in range(5):
            rand_order = V[:]
            rng.shuffle(rand_order)
            orders.append(rand_order)
        # J. By degree (conflict count) ascending - fewest conflicts first
        orders.append(sorted(V, key=lambda x: (len(conflict[x]), abs(x), x)))
        # K. Odds by degree ascending
        odds_deg = sorted([x for x in V if x % 2 != 0], key=lambda x: (len(conflict[x]), abs(x), x))
        evens_deg = sorted([x for x in V if x % 2 == 0], key=lambda x: (len(conflict[x]), abs(x), x))
        orders.append(odds_deg + evens_deg)

        # Run greedy on each order
        for order in orders:
            if time.time() - t_start > budget:
                break
            S = greedy_from_order(order, V_set)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Also try conflict-aware greedy from scratch
        if time.time() - t_start < budget:
            S = greedy_with_priority(V_set, max_passes=5, conflict=conflict)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Hypergraph-based greedy: remove high-degree vertices
        if time.time() - t_start < budget:
            S = hypergraph_greedy(V_set, seed_off, budget)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Local improvement: try adding any remaining element
        best = try_add_remaining(best, V_set, t_start + budget, conflict=conflict)
        best_size = len(best)

        # 1-opt swaps
        t_swap = time.time()
        if time.time() - t_start < budget * 0.8:
            best = local_search_1opt(best, V_set, budget * 0.2, rng, conflict=conflict)
            best_size = len(best)
            best = try_add_remaining(best, V_set, t_start + budget, conflict=conflict)

        # 2-opt swaps
        if time.time() - t_start < budget:
            best = local_search_2opt(best, V_set, budget * 0.15, rng, conflict=conflict)
            best_size = len(best)
            best = try_add_remaining(best, V_set, t_start + budget, conflict=conflict)

        return list(best)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    # time budgets, leave margin
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            # fallback: odds
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        # verify
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
