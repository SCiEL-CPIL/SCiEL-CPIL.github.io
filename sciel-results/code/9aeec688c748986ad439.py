import random
import math
import time

def construct():
    n = 16
    target_edges = 65
    
    def get_score_and_worst(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        max_sum = 0
        worst_tris = []
        # Iterate over edges to find triangles
        for u, v in edge_set:
            common = adj[u].intersection(adj[v])
            for w in common:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tris = [(u, v, w)]
                elif s == max_sum:
                    worst_tris.append((u, v, w))
        return max_sum, worst_tris

    all_possible_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    
    best_score = float('inf')
    best_edges = None
    
    # Multi-start SA
    num_starts = 20
    time_per_start = 4.0 # seconds
    start_time = time.time()
    total_time_limit = 110.0 # leave some buffer for 120s limit
    
    for start_idx in range(num_starts):
        if time.time() - start_time > total_time_limit:
            break
            
        # Initialize
        # Strategy: Start with a random graph with 65 edges
        # Or start with K_8,8 + 1 edge for the first few, then random
        if start_idx < 5:
            edges = set()
            for u in range(8):
                for v in range(8, 16):
                    edges.add((u, v))
            # Add 1 random edge within partitions
            possible_extra = [(u, v) for u in range(8) for v in range(u+1, 8)] + \
                              [(u+8, v+8) for u in range(8) for v in range(u+1, 8)]
            edges.add(random.choice(possible_extra))
        else:
            edges = set(random.sample(list(all_possible_edges), target_edges))
            
        current_score, worst_tris = get_score_and_worst(edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        if best_score <= 23:
            return {'n': n, 'edges': [[u, v] for u, v in best_edges]}

        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 0
        max_steps = 50000
        
        local_start_time = time.time()
        
        while T > T_min and steps < max_steps and (time.time() - local_start_time) < time_per_start:
            steps += 1
            
            # Move generation
            # 80%: 1-edge swap (remove 1, add 1)
            # 20%: 2-edge swap (remove 2, add 2)
            
            if random.random() < 0.8:
                # 1-edge swap
                # Prefer removing from worst triangles
                if worst_tris and random.random() < 0.7:
                    tri = random.choice(worst_tris)
                    e1 = (min(tri[0], tri[1]), max(tri[0], tri[1]))
                    e2 = (min(tri[0], tri[2]), max(tri[0], tri[2]))
                    e3 = (min(tri[1], tri[2]), max(tri[1], tri[2]))
                    e_remove = random.choice([e1, e2, e3])
                else:
                    e_remove = random.choice(list(edges))
                
                available_adds = [e for e in all_possible_edges if e not in edges]
                if not available_adds:
                    continue
                e_add = random.choice(available_adds)
                
                new_edges = set(edges)
                new_edges.remove(e_remove)
                new_edges.add(e_add)
                
            else:
                # 2-edge swap
                if len(edges) < 2:
                    continue
                e_rem1, e_rem2 = random.sample(list(edges), 2)
                available_adds = [e for e in all_possible_edges if e not in edges]
                if len(available_adds) < 2:
                    continue
                e_add1, e_add2 = random.sample(available_adds, 2)
                
                new_edges = set(edges)
                new_edges.remove(e_rem1)
                new_edges.remove(e_rem2)
                new_edges.add(e_add1)
                new_edges.add(e_add2)
            
            new_score, new_worst_tris = get_score_and_worst(new_edges)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                edges = new_edges
                current_score = new_score
                worst_tris = new_worst_tris
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
                if best_score <= 23:
                    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}
            
            T *= cooling

    if best_edges is None:
        # Fallback: return K_8,8 + 1 edge
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((u, v))
        edges.add((0, 1))
        best_edges = edges

    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}
