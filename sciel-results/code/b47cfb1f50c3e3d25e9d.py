import time
import sys
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # Safety margin
    
    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # Generate a random graph with max degree d
        # The prompt implies a specific generator "bounded_degree_graph" but provides a seed.
        # I will replicate the generator from the current program as it is likely the standard one.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Conflict: edges i, j conflict if they share a vertex or are distance 1.
        # i=(u,v), j=(x,y).
        # Conflict if {u,v} & {x,y} != empty OR (u in adj[x] or u in adj[y] or v in adj[x] or v in adj[y])
        
        v_to_e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v_to_e[u].append(i)
            v_to_e[v].append(i)
            
        conf = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            # Edges sharing u or v
            for j in v_to_e[u]:
                if j != i:
                    conf[i].add(j)
            for j in v_to_e[v]:
                if j != i:
                    conf[i].add(j)
            # Edges connected by an edge
            for x in adj.get(u, []):
                for j in v_to_e[x]:
                    if j != i:
                        conf[i].add(j)
            for x in adj.get(v, []):
                for j in v_to_e[x]:
                    if j != i:
                        conf[i].add(j)
        
        # Convert to list of lists for faster iteration, and sort
        conf_list = [sorted(list(c)) for c in conf]
        return conf_list

    def dsatur_coloring(conf, m, deadline, rng=None):
        col = [-1] * m
        forbidden = [set() for _ in range(m)]
        remaining = set(range(m))
        deg = [len(c) for c in conf]
        
        while remaining and time.time() < deadline:
            best_edge = None
            best_key = None
            for e in remaining:
                cnt = len(forbidden[e])
                key = (-cnt, -deg[e], e)
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(remaining))
            
            c = 0
            while c in forbidden[best_edge]:
                c += 1
            
            col[best_edge] = c
            remaining.discard(best_edge)
            
            for j in conf[best_edge]:
                if j in remaining:
                    forbidden[j].add(c)
        
        return col

    def k_of(col):
        if not col:
            return 0
        return max(col) + 1

    def local_improve(col, conf, m, deadline, rng, max_guard=2000):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < max_guard:
            improved = False
            guard += 1
            cls = defaultdict(set)
            for i, c in enumerate(col):
                cls[c].add(i)
            
            # Merge pass
            merged = False
            cols = sorted(cls.keys())
            for idx, c1 in enumerate(cols):
                if c1 not in cls: continue
                s1 = cls[c1]
                for c2 in cols[idx + 1:]:
                    if c2 not in cls: continue
                    s2 = cls[c2]
                    ok = True
                    for e in s2:
                        for j in conf[e]:
                            if j in s1:
                                ok = False
                                break
                        if not ok: break
                    if ok:
                        for e in s2:
                            col[e] = c1
                        s1.update(s2)
                        del cls[c2]
                        merged = True
                        break
                if merged: break
            if merged:
                improved = True
                continue

            # Single edge move
            moved = False
            for e in range(m):
                c = col[e]
                if c == 0: continue
                for new_c in range(c):
                    valid = True
                    for j in conf[e]:
                        if col[j] == new_c:
                            valid = False
                            break
                    if valid:
                        col[e] = new_c
                        moved = True
                        break
                if moved: break
            if moved:
                improved = True
                continue

            # 2-swap
            swapped = False
            for _ in range(2000):
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2: continue
                c1, c2 = col[e1], col[e2]
                if c1 == c2: continue
                valid = True
                for j in conf[e1]:
                    if j != e2 and col[j] == c2:
                        valid = False
                        break
                if valid:
                    for j in conf[e2]:
                        if j != e1 and col[j] == c1:
                            valid = False
                            break
                if valid:
                    col[e1], col[e2] = c2, c1
                    swapped = True
                    break
            if swapped:
                improved = True

        # Reindex
        present = sorted(set(col))
        if not present: return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def solve_instance(n, d, seed, time_budget):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0: return []
        
        conf = build_conflict(edges)
        
        # 1. Get initial solution
        rng = Random(seed + 42)
        col = dsatur_coloring(conf, m, t0 + time_budget, rng)
        col = local_improve(col, conf, m, t0 + time_budget, rng)
        best_k = k_of(col)
        best_col = col[:]
        
        # 2. Try to improve with B&B
        # Order vertices by degree descending
        deg = [len(c) for c in conf]
        order = sorted(range(m), key=lambda x: -deg[x])
        
        # Precompute conflicts in the new order? No, just use index.
        # B&B State:
        # color_assignment: list of length m, -1 if uncolored
        # next_idx: index in 'order'
        # current_k: number of colors used so far
        
        # Lower Bound function:
        # For remaining vertices, greedy coloring to estimate min colors needed.
        def greedy_lb(start_idx, color_assignment):
            # Greedy color the remaining vertices in order
            # Returns min colors needed
            # This is expensive, so use a simpler bound:
            # Max degree in remaining subgraph + 1?
            # Or just 1.
            # Let's use a fast greedy on the remaining set.
            rem = [order[i] for i in range(start_idx, m)]
            if not rem: return 0
            
            # Simple greedy
            rem_col = {v: -1 for v in rem}
            rem_conf = {v: set() for v in rem}
            for v in rem:
                for u in conf[v]:
                    if u in rem_conf:
                        rem_conf[v].add(u)
            
            used = 0
            for v in rem:
                # Find smallest color not in neighbors
                nb_colors = {rem_col[u] for u in rem_conf[v] if rem_col[u] != -1}
                c = 0
                while c in nb_colors:
                    c += 1
                rem_col[v] = c
                if c >= used:
                    used = c + 1
            return used

        # B&B DFS
        sys.setrecursionlimit(10000)
        
        def bb_dfs(idx, current_k, color_assignment):
            nonlocal best_k, best_col
            
            if time.time() > t0 + time_budget:
                return False # Timeout
            
            if idx == m:
                if current_k < best_k:
                    best_k = current_k
                    best_col = color_assignment[:]
                return True
            
            v = order[idx]
            
            # Try existing colors 0..current_k-1
            for c in range(current_k):
                # Check if c is valid for v
                valid = True
                for u in conf[v]:
                    if color_assignment[u] == c:
                        valid = False
                        break
                if valid:
                    color_assignment[v] = c
                    if bb_dfs(idx + 1, current_k, color_assignment):
                        color_assignment[v] = -1
                        return True
                    color_assignment[v] = -1
            
            # Try new color current_k
            # Symmetry breaking: Only try new color if current_k == 0 OR 
            # this vertex is the first one to use a new color in this branch?
            # Standard: Try new color.
            # Lower bound check before trying new color?
            # If current_k + greedy_lb(idx+1, color_assignment) >= best_k:
            #    return False # Prune
            
            # Optimization: Only try new color if we haven't exhausted existing ones?
            # Yes.
            
            # Check if new color is needed
            # If we use a new color, current_k becomes current_k + 1
            if current_k + 1 < best_k:
                # Pruning with LB
                # lb = greedy_lb(idx + 1, color_assignment) # Too slow
                # Simple bound: 1
                if current_k + 1 + 1 <= best_k: # Rough
                    color_assignment[v] = current_k
                    if bb_dfs(idx + 1, current_k + 1, color_assignment):
                        color_assignment[v] = -1
                        return True
                    color_assignment[v] = -1
            
            return False

        # Run B&B with a time limit
        # We can't easily interrupt recursion in Python without exceptions or global flags.
        # Use a global flag.
        stop_bb = False
        original_deadline = t0 + time_budget
        bb_deadline = original_deadline - 2.0 # Leave time for final polish
        
        def bb_dfs_timed(idx, current_k, color_assignment):
            nonlocal best_k, best_col, stop_bb
            if stop_bb:
                return False
            if time.time() > bb_deadline:
                stop_bb = True
                return False
            
            if idx == m:
                if current_k < best_k:
                    best_k = current_k
                    best_col = color_assignment[:]
                return True
            
            v = order[idx]
            
            # Try existing colors
            for c in range(current_k):
                valid = True
                for u in conf[v]:
                    if color_assignment[u] == c:
                        valid = False
                        break
                if valid:
                    color_assignment[v] = c
                    if bb_dfs_timed(idx + 1, current_k, color_assignment):
                        color_assignment[v] = -1
                        return True
                    color_assignment[v] = -1
            
            # Try new color
            if current_k + 1 < best_k:
                color_assignment[v] = current_k
                if bb_dfs_timed(idx + 1, current_k + 1, color_assignment):
                    color_assignment[v] = -1
                    return True
                color_assignment[v] = -1
                
            return False

        # Initialize for B&B
        color_assignment = [-1] * m
        # Start with best_k from heuristic
        # If B&B finds a better one, it updates best_k
        # Note: B&B explores the space. It might not find the optimal if it times out.
        # But it guarantees that if it finishes, best_k is optimal.
        
        # To make B&B effective, we should start with the heuristic solution's k as the bound.
        # The B&B will try to find a solution with < best_k colors.
        
        # Run B&B
        bb_dfs_timed(0, 0, color_assignment)
        
        # Final Polish on best_col found
        if best_col:
            final_col = local_improve(best_col, conf, m, t0 + time_budget, rng, max_guard=1000)
            final_k = k_of(final_col)
            if final_k < best_k:
                best_k = final_k
                best_col = final_col
        
        return best_col

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    total_time = 118.0
    weights = [n * d for n, d, s in specs]
    total_w = sum(weights)
    budgets = [total_time * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(specs):
        if time.time() > t0 + DEADLINE - 5:
            # Fallback: simple greedy
            edges = gen_instance(*spec)
            m = len(edges)
            conf = build_conflict(edges)
            col = dsatur_coloring(conf, m, t0 + DEADLINE)
            out.append(col)
        else:
            out.append(solve_instance(*spec, budgets[i]))
    
    return out
