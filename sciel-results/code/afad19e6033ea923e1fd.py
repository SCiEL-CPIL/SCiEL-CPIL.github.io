import time
import sys
from random import Random
import math

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency of original graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        deg = [len(neigh[i]) for i in range(m)]
        
        # DSATUR for initial coloring
        color = [-1] * m
        order = sorted(range(m), key=lambda i: -deg[i])
        for v in order:
            used = set()
            for u in neigh[v]:
                if color[u] != -1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        
        current_K = max(color) + 1 if color else 0
        best_color = color[:]
        best_K = current_K
        
        # Simulated Annealing
        rng = Random(seed + 12345)
        
        # Precompute conflict counts for current best
        def get_conflicts(col):
            cnt = 0
            for i in range(m):
                ci = col[i]
                for j in neigh[i]:
                    if j > i and col[j] == ci:
                        cnt += 1
            return cnt

        # We want to minimize K. 
        # Strategy: 
        # 1. Find a valid coloring with K0 (DSATUR).
        # 2. Try to reduce K by 1, 2, ... using SA.
        
        # Let's try to find the minimum K possible.
        # Lower bound: max degree of conflict graph? Or just try down from current_K.
        
        # We will run SA for each target K from current_K down to some lower bound.
        # To save time, we can just run one long SA that allows K to change? 
        # No, standard SA for graph coloring usually fixes K and minimizes conflicts.
        # Here we want to minimize K.
        
        # Approach:
        # Start with K = current_K.
        # While K > lower_bound:
        #   Try to find a valid coloring with K-1.
        #   If found, K = K-1.
        #   Else, break.
        
        # How to try K-1?
        # Initialize coloring with K-1 colors (random or greedy).
        # Run SA to minimize conflicts.
        # If conflicts == 0, success.
        
        lower_bound = max(deg) + 1 # Chromatic number of conflict graph is at least max degree + 1? No, that's for greedy. Lower bound is clique number.
        # Actually, for strong edge coloring, the lower bound is related to the maximum degree.
        # The conjecture says 5d^2/4.
        # Let's just try to reduce K as much as possible.
        
        # To be efficient, we'll run a global SA that minimizes (conflicts, K).
        # But K is discrete.
        
        # Let's stick to the iterative reduction.
        # Time budget per attempt.
        
        def try_reduce_K(target_K, time_limit):
            # Initialize
            if target_K == 0:
                return [0]*m, True
            
            # Greedy init for target_K
            # If greedy fails, use random init
            col = [-1] * m
            # Sort by degree
            order = sorted(range(m), key=lambda i: -deg[i])
            for v in order:
                used = set()
                for u in neigh[v]:
                    if col[u] != -1:
                        used.add(col[u])
                # Find smallest available color < target_K
                c = 0
                while c < target_K and c in used:
                    c += 1
                if c < target_K:
                    col[v] = c
                else:
                    col[v] = rng.randrange(target_K)
            
            # SA
            T = 10.0
            T_min = 0.01
            cooling = 0.95
            
            # Calculate initial conflicts
            def calc_conflicts(col):
                cnt = 0
                for i in range(m):
                    ci = col[i]
                    for j in neigh[i]:
                        if j > i and col[j] == ci:
                            cnt += 1
                return cnt
                
            cur_conf = calc_conflicts(col)
            best_conf = cur_conf
            best_col = col[:]
            
            start_time = time.time()
            
            while T > T_min and time.time() < time_limit:
                # Move: pick random vertex, change color
                v = rng.randrange(m)
                old_c = col[v]
                new_c = rng.randrange(target_K)
                if new_c == old_c:
                    continue
                
                # Calculate delta
                delta = 0
                for u in neigh[v]:
                    if col[u] == old_c:
                        delta -= 1
                    if col[u] == new_c:
                        delta += 1
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    col[v] = new_c
                    cur_conf += delta
                    if cur_conf < best_conf:
                        best_conf = cur_conf
                        best_col = col[:]
                        if best_conf == 0:
                            return best_col, True
                else:
                    # Revert
                    pass # col[v] is still old_c
                
                T *= cooling
                
            return best_col, (best_conf == 0)

        # Try to reduce K
        for target_K in range(current_K - 1, max(3, current_K // 2), -1):
            if time.time() > deadline - 2.0:
                break
            
            # Give some time to each attempt
            # If we have lots of time, use more.
            # Let's say 5 seconds per attempt?
            attempt_deadline = min(time.time() + 5.0, deadline - 2.0)
            
            res_col, ok = try_reduce_K(target_K, attempt_deadline)
            if ok:
                best_color = res_col
                best_K = target_K
            else:
                # If we fail, maybe try again with different random seed?
                # Or just break?
                # If we can't reduce by 1, we likely can't reduce by more.
                # But sometimes skipping a K helps? Unlikely.
                break
        
        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
