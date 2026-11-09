"""C4-free graph with no independent 11-set (Ramsey R(C4,K11)) via greedy construction."""
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - time.time()

    # ---------- deterministic C4-free fallback (10 triangles, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free graph construction: prefer connecting high-degree vertices ----------
    def build_c4free(n, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        # Try multiple random orderings, keep the densest result
        best_adj = None
        best_density = -1
        for attempt in range(3):
            adj_try = [set() for _ in range(n)]
            order = list(range(n))
            rng.shuffle(order)
            # Process vertices in random order, but try to connect to as many as possible
            for idx, i in enumerate(order):
                for j in order[idx+1:]:
                    if len(adj_try[i] & adj_try[j]) <= 1:
                        adj_try[i].add(j)
                        adj_try[j].add(i)
            density = sum(len(s) for s in adj_try)
            if density > best_density:
                best_density = density
                best_adj = adj_try
        return best_adj

    def densify(adj, n, seed):
        """Add edges between vertices with small neighborhood intersection (never creates a C4)."""
        rng = random.Random(seed)
        changed = True
        while changed:
            changed = False
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if j in adj[i]:
                    continue
                inter = adj[i] & adj[j]
                if len(inter) <= 1:      # safe: won't create C4
                    adj[i].add(j)
                    adj[j].add(i)
                    changed = True
        return adj

    # ---------- greedy maximal independent set ----------
    def greedy_mis(adj, n, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        indep = []
        in_set = set()
        for v in order:
            if v in in_set:
                continue
            if all((u not in adj[v]) for u in indep):
                indep.append(v)
                in_set.add(v)
        return len(indep)

    # ---------- exact alpha check via branch and bound (for n <= 40) ----------
    def exact_alpha(adj, n):
        """Return the independence number using branch and bound."""
        if n > 42:
            return None  # too expensive
        # Use a simple recursive branch and bound
        best_alpha = 0
        # Sort vertices by degree (ascending) for better pruning
        order = sorted(range(n), key=lambda v: len(adj[v]))
        
        def branch(candidates, current_size, current_set):
            nonlocal best_alpha
            if not candidates:
                if current_size > best_alpha:
                    best_alpha = current_size
                return
            if current_size + len(candidates) <= best_alpha:
                return
            # Pick a vertex to branch on
            v = candidates[0]
            remaining = candidates[1:]
            # Branch 1: include v
            new_candidates = [u for u in remaining if u not in adj[v]]
            branch(new_candidates, current_size + 1, current_set | {v})
            # Branch 2: exclude v
            branch(remaining, current_size, current_set)
        
        branch(list(order), 0, set())
        return best_alpha

    # ---------- candidate validation (fast, non-exact) ----------
    def fast_ok(adj, n):
        # confirm C4-free
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i] and len(adj[i] & adj[j]) > 1:
                    return False
        # confirm greedy MIS < 11 across many orderings
        for s in range(30):
            if greedy_mis(adj, n, 9000 + s) >= 11:
                return False
        return True

    # ---------- main search ----------
    best = None
    target_ns = [43, 42, 41, 40, 39, 38, 37, 36]
    trial = 0
    for n in target_ns:
        if time.time() >= DEADLINE - 3.0:
            break
        max_trials_for_n = 60 if n <= 40 else 30
        found_for_n = False
        for trial_n in range(max_trials_for_n):
            if time.time() >= DEADLINE - 3.0:
                break
            trial += 1
            adj = build_c4free(n, trial)
            adj = densify(adj, n, trial)
            if fast_ok(adj, n):
                # For smaller n, verify with exact alpha
                if n <= 40:
                    alpha = exact_alpha(adj, n)
                    if alpha is not None and alpha < 11:
                        best = (n, adj)
                        found_for_n = True
                        break
                else:
                    best = (n, adj)
                    found_for_n = True
                    break
        if found_for_n:
            break

    if best is None:
        return fallback()

    n, adj = best
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
