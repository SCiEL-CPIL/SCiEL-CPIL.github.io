import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
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
    
    def generate_random_graph():
        """Generate a random graph with exactly target_edges edges."""
        random.shuffle(all_edges)
        edges = set(all_edges[:target_edges])
        return edges
    
    def generate_greedy_graph():
        """Greedy C4-minimizing construction."""
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
        
        return edges
    
    def local_search(edges_init, time_limit):
        """Simulated annealing with edge swaps, using numpy for fast C4 counting."""
        adj = build_adj(edges_init)
        edges = set(edges_init)
        
        current_score = count_c4(adj)
        best_score = current_score
        best_edges = set(edges)
        
        T = 5.0
        cooling = 0.999
        min_T = 0.005
        
        # Precompute edge list for faster random selection
        edge_list = list(edges)
        
        while time.time() - start_time < time_limit and T > min_T:
            if not edges:
                break
            
            # Pick random edge to remove
            e_rm = random.choice(edge_list)
            u_rm, v_rm = e_rm
            
            # Pick random non-edge to add
            found = False
            for _ in range(50):
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add:
                    continue
                if u_add > v_add:
                    u_add, v_add = v_add, u_add
                e_add = (u_add, v_add)
                if e_add not in edges:
                    found = True
                    break
            if not found:
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
                # Update edge_list
                edge_list.remove(e_rm)
                edge_list.append(e_add)
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
                edge_list.remove(e_add)
                edge_list.append(e_rm)
            
            T *= cooling
        
        return best_score, best_edges
    
    def hill_climbing(edges_init, time_limit):
        """Steepest-descent hill climbing with edge swaps."""
        adj = build_adj(edges_init)
        edges = set(edges_init)
        
        current_score = count_c4(adj)
        best_score = current_score
        best_edges = set(edges)
        
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            # Try all possible swaps, find the best improvement
            best_delta = 0
            best_swap = None
            
            for e_rm in edges:
                u_rm, v_rm = e_rm
                for u_add in range(n):
                    for v_add in range(u_add + 1, n):
                        if (u_add, v_add) in edges:
                            continue
                        if u_add == u_rm and v_add == v_rm:
                            continue
                        if u_add == v_rm and v_add == u_rm:
                            continue
                        
                        e_add = (u_add, v_add)
                        
                        # Compute delta
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
                        
                        after = sum(pair_contrib(a, b) for (a, b) in affected)
                        
                        delta = (after - before) // 2
                        
                        # Revert
                        adj[u_rm, v_rm] = 1
                        adj[v_rm, u_rm] = 1
                        adj[u_add, v_add] = 0
                        adj[v_add, u_add] = 0
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (e_rm, e_add)
            
            if best_swap is None or best_delta >= 0:
                no_improve += 1
                # Random perturbation to escape local minimum
                if no_improve < max_no_improve:
                    e_rm = random.choice(list(edges))
                    u_rm, v_rm = e_rm
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
                    
                    adj[u_rm, v_rm] = 0
                    adj[v_rm, u_rm] = 0
                    adj[u_add, v_add] = 1
                    adj[v_add, u_add] = 1
                    edges.discard(e_rm)
                    edges.add(e_add)
                    current_score = count_c4(adj)
                    no_improve = 0
            else:
                e_rm, e_add = best_swap
                u_rm, v_rm = e_rm
                u_add, v_add = e_add
                
                adj[u_rm, v_rm] = 0
                adj[v_rm, u_rm] = 0
                adj[u_add, v_add] = 1
                adj[v_add, u_add] = 1
                edges.discard(e_rm)
                edges.add(e_add)
                
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
                no_improve = 0
        
        return best_score, best_edges
    
    best_overall_score = float('inf')
    best_overall_edges = None
    
    # Phase 1: Multiple SA runs with different initializations
    while time.time() - start_time < deadline - 30:
        # Alternate between random and greedy initialization
        if random.random() < 0.5:
            init_edges = generate_random_graph()
        else:
            init_edges = generate_greedy_graph()
        
        score, edges = local_search(init_edges, deadline - 30)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
            if best_overall_score <= 2:
                break
    
    # Phase 2: Hill climbing on the best result
    if best_overall_edges is not None and time.time() - start_time < deadline - 5:
        score, edges = hill_climbing(best_overall_edges, deadline - 5)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
    
    # Phase 3: More SA runs if time permits
    while time.time() - start_time < deadline - 3:
        init_edges = generate_greedy_graph()
        score, edges = local_search(init_edges, deadline - 3)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
            if best_overall_score <= 1:
                break
    
    if best_overall_edges is None:
        best_overall_edges = generate_greedy_graph()
    
    return [[u, v] for u, v in best_overall_edges]
