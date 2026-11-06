import time
import math
import random
from random import Random

def construct():
    def build_instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(6 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        m = len(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Conflict if edges share a vertex OR are joined by an edge
        # For edge i=(a,b) and edge j=(c,d):
        # Conflict if a==c or a==d or b==c or b==d (share vertex)
        # OR if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b] (joined)
        # Note: if they share a vertex, the "joined" check is redundant but harmless.
        
        # Precompute for speed:
        # For each edge i, store the set of vertices it "touches" in the strong sense?
        # No, the condition is specific.
        
        # Let's use a bitset approach for the conflict graph if m is small enough, or just sets.
        # m is at most n*d/2. For n=80, d=6, m <= 240.
        # 240 bits fits in 4 64-bit words or a Python int.
        
        conflict = [0] * m  # Bitset of conflicts for each edge
        
        # To speed up, we can group edges by their endpoints.
        # But m is small enough that O(m^2) is fine. 240^2 = 57600 checks.
        
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                if a == c or a == d or b == c or b == d:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    # Check if joined
                    if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
        
        return conflict, adj

    def solve_instance(n, d, seed, deadline):
        edges = build_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict, adj = build_conflict(edges, n)
        
        # Lower bound: max degree in conflict graph + 1
        # Or clique number lower bound.
        # Let's just use greedy to get an upper bound first.
        
        # Greedy Coloring (DSATUR-like)
        def greedy_dsatur():
            color = [-1] * m
            uncolored = set(range(m))
            
            while uncolored:
                # Pick vertex with max saturation (distinct colors in neighbors)
                # Tie-break by degree in uncolored subgraph? Or total degree?
                best_v = -1
                best_sat = -1
                best_deg = -1
                
                for v in uncolored:
                    # Count distinct colors in neighbors
                    neigh_colors = set()
                    bits = conflict[v]
                    temp = bits
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        if color[u] != -1:
                            neigh_colors.add(color[u])
                        temp ^= lsb
                    sat = len(neigh_colors)
                    
                    # Degree in uncolored
                    deg = 0
                    temp = bits
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        if color[u] == -1:
                            deg += 1
                        temp ^= lsb
                        
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                # Assign smallest available color
                forbidden = set()
                temp = conflict[best_v]
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    if color[u] != -1:
                        forbidden.add(color[u])
                    temp ^= lsb
                
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                uncolored.discard(best_v)
            
            return color

        best_color = greedy_dsatur()
        best_K = max(best_color) + 1
        
        # Local Search: Simulated Annealing / Tabu to reduce K
        # We will try to reduce K by 1, 2, etc.
        
        def count_conflicts(color_arr, K):
            # Returns number of pairs (i,j) with color[i]==color[j] and i,j in conflict
            # We can optimize this by checking only neighbors.
            conflicts = 0
            for i in range(m):
                if color_arr[i] < K:
                    temp = conflict[i]
                    while temp:
                        lsb = temp & -temp
                        j = lsb.bit_length() - 1
                        if color_arr[j] == color_arr[i]:
                            conflicts += 1
                        temp ^= lsb
            return conflicts // 2
        
        def anneal_reduce(color_init, target_K, time_limit, rng):
            # Initialize
            color_arr = list(color_init)
            for i in range(m):
                if color_arr[i] >= target_K:
                    color_arr[i] = rng.randrange(target_K)
            
            current_conflicts = count_conflicts(color_arr, target_K)
            if current_conflicts == 0:
                return color_arr
            
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            best_color = color_arr[:]
            best_conflicts = current_conflicts
            start_time = time.time()
            
            while time.time() < time_limit:
                if current_conflicts == 0:
                    return color_arr
                
                v = rng.randrange(m)
                old_c = color_arr[v]
                new_c = rng.randrange(target_K)
                
                if new_c == old_c:
                    continue
                
                # Calculate delta
                delta = 0
                temp = conflict[v]
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    if color_arr[u] == new_c:
                        delta += 1
                    if color_arr[u] == old_c:
                        delta -= 1
                    temp ^= lsb
                
                color_arr[v] = new_c
                current_conflicts += delta
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_color = color_arr[:]
                    if best_conflicts == 0:
                        return best_color
                elif delta > 0:
                    # Accept with probability
                    if T > T_min and rng.random() < math.exp(-delta / T):
                        pass
                    else:
                        # Reject
                        color_arr[v] = old_c
                        current_conflicts -= delta
                else:
                    # Accept (delta < 0 or 0)
                    pass
                
                T = max(T * cooling, T_min)
            
            if best_conflicts == 0:
                return best_color
            return None

        # Try to reduce K
        rng = Random(seed * 7919 + 1)
        
        # Determine a reasonable lower bound to stop trying
        # Clique number lower bound?
        # Just try down to max_degree_conflict + 1
        max_deg_conflict = 0
        for i in range(m):
            max_deg_conflict = max(max_deg_conflict, conflict[i].bit_count())
        lower_bound = max_deg_conflict + 1
        
        for target_K in range(best_K - 1, max(lower_bound - 1, 2), -1):
            if time.time() > deadline - 2.0:
                break
            
            success = False
            # Multiple restarts for each target K
            max_restarts = 20
            for r in range(max_restarts):
                if time.time() > deadline - 2.0:
                    break
                
                # Perturb initial solution
                trial = best_color[:]
                # Randomly reassign some colors
                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)
                    elif rng.random() < 0.2:
                        trial[i] = rng.randrange(target_K)
                
                # Time budget per restart
                remaining = deadline - 2.0 - time.time()
                if remaining <= 0:
                    break
                anneal_time = time.time() + max(0.1, remaining / (max_restarts - r) * 0.8)
                
                result = anneal_reduce(trial, target_K, anneal_time, rng)
                if result is not None:
                    best_color = result
                    best_K = target_K
                    success = True
                    break
            
            if not success:
                # If we failed to reduce to target_K, we might still try lower?
                # Usually if we can't reduce by 1, we can't reduce by 2 easily without a good start.
                # But we can try again with more restarts?
                # For now, break to save time, or continue if time allows?
                # Let's break to ensure we have time for other instances if this is not the last one.
                # Actually, we have a per-instance deadline.
                break
        
        return best_color

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)
        
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        
        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = build_instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            color = {}
            for e in edges:
                a, b = e
                # Forbidden colors are those of edges that conflict with (a,b)
                # Conflict if share vertex or joined.
                # Vertices involved in conflict with (a,b):
                # 1. a, b (share vertex)
                # 2. neighbors of a, neighbors of b (joined)
                # Let's collect all edges that conflict with (a,b)
                forbidden = set()
                for f, c in color.items():
                    c1, c2 = f
                    if c1 == a or c1 == b or c2 == a or c2 == b:
                        forbidden.add(c)
                    else:
                        if c1 in adj[a] or c2 in adj[a] or c1 in adj[b] or c2 in adj[b]:
                            forbidden.add(c)
                
                c = 0
                while c in forbidden:
                    c += 1
                color[e] = c
            
            out.append([color[e] for e in edges])
        return out

