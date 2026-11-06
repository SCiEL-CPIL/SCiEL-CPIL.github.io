import time
import numpy as np
import random

def construct():
    start = time.time()
    DEADLINE = 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def build_conflict_graph(V):
        """
        Builds a graph where an edge (u, v) exists if u + v is in V.
        An independent set in this graph is guaranteed to be sum-free.
        """
        V_set = set(V)
        n = len(V)
        idx_map = {v: i for i, v in enumerate(V)}
        adj = [set() for _ in range(n)]
        
        # Optimization: iterate pairs
        for i in range(n):
            u = V[i]
            for j in range(i + 1, n):
                v = V[j]
                s = u + v
                if s in V_set:
                    k = idx_map[s]
                    # The constraint is a+b=c.
                    # If we pick u and v, we cannot pick s.
                    # This is not a direct edge between u and v in a "cannot coexist" graph 
                    # unless we are enforcing that s is NOT in the set.
                    # Wait, the MIS strategy from the hint implies:
                    # "conflict graph" where edge means "cannot be in same set".
                    # If u+v = s, and s is in V, can u and v be in the same sum-free set?
                    # Yes, if s is NOT in the set.
                    # So u and v do NOT conflict directly.
                    # However, the hint says "MIS on conflict graph".
                    # If the hint implies that the MIS of the graph defined by (u,v) if u+v in V
                    # yields a sum-free set, it relies on the fact that if u,v are in the set,
                    # u+v is NOT in the set (because if u+v were in the set, it would be a vertex,
                    # and the independence condition only restricts u,v, not u+v).
                    # 
                    # Let's re-read carefully:
                    # "Model the problem as a maximum independent set in the complement of the 3-uniform hypergraph..."
                    # Or "MIS solver on the 'conflict graph'".
                    # 
                    # If I define edge (u,v) if u+v in V, and take MIS, I get a set S such that
                    # for all u,v in S, u+v NOT in V.
                    # Since S subset V, u+v NOT in S.
                    # So S is sum-free.
                    # This is a valid construction.
                    # 
                    # So, adj[i].add(j) and adj[j].add(i) if V[i]+V[j] in V_set.
                    adj[i].add(j)
                    adj[j].add(i)
        return adj, V

    def greedy_mis(adj, V):
        """
        Greedy Maximum Independent Set on the conflict graph.
        """
        n = len(V)
        in_set = [False] * n
        remaining = set(range(n))
        selected = []
        
        # Heuristic: pick node with min degree in remaining subgraph
        while remaining:
            # Calculate degrees within remaining
            degrees = {}
            for u in remaining:
                deg = 0
                for v in adj[u]:
                    if v in remaining:
                        deg += 1
                degrees[u] = deg
            
            # Pick min degree
            min_node = min(remaining, key=lambda x: degrees[x])
            selected.append(min_node)
            in_set[min_node] = True
            
            # Remove min_node and its neighbors from remaining
            to_remove = {min_node}
            for v in adj[min_node]:
                if v in remaining:
                    to_remove.add(v)
            
            for v in to_remove:
                remaining.discard(v)
                
        return [V[i] for i in selected]

    def can_add(S, x):
        """Check if x can be added to sum-free set S."""
        Sset = set(S)
        if 2 * x in Sset:
            return False
        for y in S:
            if x + y in Sset:
                return False
        return True

    def solve_instance(V, time_limit, start_time):
        n = len(V)
        if n == 0:
            return []
        
        adj, V_list = build_conflict_graph(V)
        
        # Get initial MIS
        S_indices = greedy_mis(adj, V_list)
        S = [V_list[i] for i in S_indices]
        Sset = set(S)
        
        # Improvement Phase: Try to add more elements
        # Elements not in S
        candidates = [x for x in V if x not in Sset]
        
        deadline = start_time + time_limit
        no_improve = 0
        
        while time.time() < deadline - 0.1:
            improved = False
            
            # 1. Try to add candidates
            new_candidates = []
            for x in candidates:
                if can_add(S, x):
                    S.append(x)
                    Sset.add(x)
                    improved = True
                else:
                    new_candidates.append(x)
            candidates = new_candidates
            
            # 2. If no additions, try swaps (1-out, 1-in or 1-out, 2-in)
            if not improved:
                # Try 1-out, 1-in
                found_swap = False
                S_list = list(S)
                for i in range(len(S_list)):
                    y = S_list[i]
                    S_temp = S_list[:i] + S_list[i+1:]
                    S_temp_set = set(S_temp)
                    
                    # Try to add best candidate
                    best_x = None
                    # Sort candidates by some heuristic? Just iterate
                    for x in candidates:
                        # Check if x can be added to S_temp
                        if can_add(S_temp, x):
                            # Check if we actually gain? Size is same (remove 1, add 1).
                            # But maybe it allows further additions?
                            # For now, just swap if it leads to a state where we can add more?
                            # Simpler: Just swap if x is "better" in some way?
                            # Actually, if we swap, we might unblock others.
                            # Let's just do a random swap to escape local minima.
                            best_x = x
                            break
                    
                    if best_x is not None:
                        S = S_temp + [best_x]
                        Sset = set(S)
                        candidates = [x for x in candidates if x != best_x]
                        improved = True
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try 1-out, 2-in
                    # This is expensive, maybe skip for simplicity if time is tight
                    pass
            
            if not improved:
                no_improve += 1
                if no_improve > 10:
                    break
            else:
                no_improve = 0
        
        # Final check
        if not is_sum_free(S):
             # Fallback greedy removal
             R = set(S)
             while not is_sum_free(R):
                 lst = list(R)
                 best = None
                 bestcnt = -1
                 for a in lst:
                     cnt = 0
                     for b in lst:
                         if a + b in R:
                             cnt += 1
                     if cnt > bestcnt:
                         bestcnt = cnt
                         best = a
                 R.discard(best)
             S = sorted(R)
        
        return sorted(S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed
        if remaining < 2:
            # Fallback: all odds
            results.append([x for x in V if x % 2 != 0])
            continue
        
        # Allocate time proportionally or evenly
        time_limit = (remaining - 1.0) / (3 - len(results))
        
        results.append(solve_instance(V, time_limit, start))

    return results
