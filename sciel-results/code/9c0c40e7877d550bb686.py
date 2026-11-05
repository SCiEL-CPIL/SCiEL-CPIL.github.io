import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 31
    target_edges = 91
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    def build_adj(edges):
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def count_c4(adj):
        """Count 4-cycles: (1/2) * sum_{u<v} C(common(u,v), 2)"""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = int(adj[u] @ adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    def delta_c4(adj, u_rm, v_rm, u_add, v_add):
        """
        Compute change in C4 count when removing edge (u_rm,v_rm) and adding edge (u_add,v_add).
        Only pairs involving {u_rm, v_rm, u_add, v_add} can change.
        """
        affected = set()
        for x in (u_rm, v_rm, u_add, v_add):
            for y in range(n):
                if y != x:
                    a, b = min(x, y), max(x, y)
                    affected.add((a, b))
        
        def pair_contrib(a, b):
            c = int(adj[a] @ adj[b])
            if c >= 2:
                return c * (c - 1) // 2
            return 0
        
        # Sum contributions BEFORE the swap
        before = 0
        for (a, b) in affected:
            before += pair_contrib(a, b)
        
        # Perform swap on adj
        adj[u_rm, v_rm] = 0
        adj[v_rm, u_rm] = 0
        adj[u_add, v_add] = 1
        adj[v_add, u_add] = 1
        
        # Sum contributions AFTER the swap
        after = 0
        for (a, b) in affected:
            after += pair_contrib(a, b)
        
        # Revert adj
        adj[u_rm, v_rm] = 1
        adj[v_rm, u_rm] = 1
        adj[u_add, u_add] = 0  # ensure no self
        adj[u_add, v_add] = 0
        adj[v_add, u_add] = 0
        
        # The total C4 = sum_all_pairs / 2. The affected pairs contribute (after - before) to the sum.
        # So delta_c4 = (after - before) // 2 ... but we need integer.
        # Actually the sum changes by (after - before), and C4 = sum/2.
        # But (after - before) should be even? Not necessarily for partial sums.
        # Let's just return the raw difference in the sum, and divide by 2 at the end.
        # Actually, let's be safe: compute full count difference.
        # For safety with small n, just return (after - before) // 2 might lose info.
        # Let's return the difference in the full sum (not divided by 2), caller divides.
        return after - before
    
    def generate_initial_graph():
        """Greedy C4-free construction, then add one edge."""
        neighbors = [set() for _ in range(n)]
        edges = set()
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for e in shuffled_edges:
            u, v = e
            if len(neighbors[u].intersection(neighbors[v])) < 2:
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.add(e)
            if len(edges) >= target_edges:
                break
        
        if len(edges) < target_edges:
            remaining = [e for e in all_edges if e not in edges]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= target_edges:
                    break
                u, v = e
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.add(e)
        
        if len(edges) > target_edges:
            to_remove = len(edges) - target_edges
            edge_list = list(edges)
            random.shuffle(edge_list)
            for _ in range(to_remove):
                e = edge_list.pop()
                edges.discard(e)
                u, v = e
                neighbors[u].discard(v)
                neighbors[v].discard(u)
        
        return edges
    
    def local_search(edges_init, time_limit):
        """SA with edge swaps, using numpy for fast C4 counting."""
        adj = build_adj(edges_init)
        edges = set(edges_init)
        
        current_score = count_c4(adj)
        best_score = current_score
        best_edges = set(edges)
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() - start_time < time_limit and T > min_T:
            if not edges:
                break
            
            e_rm = random.choice(list(edges))
            u_rm, v_rm = e_rm
            
            # Pick random non-edge
            for _ in range(100):
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add:
                    continue
                if u_add > v_add:
                    u_add, v_add = v_add, u_add
                e_add = (u_add, v_add)
                if e_add not in edges:
                    break
            else:
                continue
            
            # Compute delta using affected pairs
            affected = set()
            for x in (u_rm, v_rm, u_add, v_add):
                for y in range(n):
                    if y != x:
                        a, b = min(x, y), max(x, y)
                        affected.add((a, b))
            
            def pair_contrib(a, b):
                c = int(adj[a] @ adj[b])
                if c >= 2:
                    return c * (c - 1) // 2
                return 0
            
            before = sum(pair_contrib(a, b) for (a, b) in affected)
            
            # Apply swap
            adj[u_rm, v_rm] = 0
            adj[v_rm, u_rm] = 0
            adj[u_add, v_add] = 1
            adj[v_add, u_add] = 1
            edges.discard(e_rm)
            edges.add(e_add)
            
            after = sum(pair_contrib(a, b) for (a, b) in affected)
            
            delta = (after - before) // 2
            new_score = current_score + delta
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edges)
            else:
                # Revert
                adj[u_rm, v_rm] = 1
                adj[v_rm, u_rm] = 1
                adj[u_add, v_add] = 0
                adj[v_add, u_add] = 0
                edges.discard(e_add)
                edges.add(e_rm)
            
            T *= cooling
        
        return best_score, best_edges
    
    best_overall_score = float('inf')
    best_overall_edges = None
    
    while time.time() - start_time < deadline - 3:
        init_edges = generate_initial_graph()
        score, edges = local_search(init_edges, deadline - 3)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
            if best_overall_score <= 1:
                break
    
    if best_overall_edges is None:
        best_overall_edges = generate_initial_graph()
    
    return [[u, v] for u, v in best_overall_edges]
