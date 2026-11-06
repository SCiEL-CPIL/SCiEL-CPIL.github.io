import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 16
    
    # Grötzsch graph (11 vertices, triangle-free, 4-chromatic)
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # Build initial adjacency as bitmasks
    adj = [0] * n
    for u, v in core_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    core_mask = [0] * n
    for u, v in core_edges:
        core_mask[u] |= (1 << v)
        core_mask[v] |= (1 << u)
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0
    
    def edge_count(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    core_edge_set = set((min(u,v), max(u,v)) for u, v in core_edges)
    
    def get_missing_edges(adj):
        missing = []
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                missing.append((u, v))
        return missing
    
    def get_non_core_missing_edges(adj):
        missing = []
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1) and (u, v) not in core_edge_set:
                missing.append((u, v))
        return missing
    
    def close_graph(adj):
        """Add all possible triangle-free edges to maximize density."""
        new_adj = adj[:]
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if not ((new_adj[u] >> v) & 1):
                    if is_triangle_free_add(new_adj, u, v):
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                        changed = True
        return new_adj
    
    # Simulated annealing parameters
    best_adj = adj[:]
    best_count = edge_count(best_adj)
    
    # First, greedily add all possible edges
    greedy_adj = close_graph(adj)
    greedy_count = edge_count(greedy_adj)
    if greedy_count > best_count:
        best_count = greedy_count
        best_adj = greedy_adj[:]
    
    # Now do simulated annealing starting from greedy_adj
    current_adj = greedy_adj[:]
    current_count = greedy_count
    
    T_start = 50.0
    T_end = 0.1
    total_steps = 200000
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
        
        # Cool temperature
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Get non-core edges that are present (can be removed) and missing (can be added)
        present_non_core = []
        for u, v in all_edges:
            if ((current_adj[u] >> v) & 1) and (u, v) not in core_edge_set:
                present_non_core.append((u, v))
        
        missing_non_core = []
        for u, v in all_edges:
            if not ((current_adj[u] >> v) & 1) and (u, v) not in core_edge_set:
                missing_non_core.append((u, v))
        
        move_type = random.random()
        
        if move_type < 0.6 and len(missing_non_core) > 0:
            # Try to add a random missing non-core edge
            u, v = random.choice(missing_non_core)
            if is_triangle_free_add(current_adj, u, v):
                new_adj = current_adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_count = edge_count(new_adj)
                delta = new_count - current_count
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current_adj = new_adj
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
            else:
                # Can't add, try a swap: remove one present edge that blocks this
                blockers = (current_adj[u] & current_adj[v])
                blocker_list = [i for i in range(n) if (blockers >> i) & 1]
                if blocker_list:
                    b = random.choice(blocker_list)
                    # Find an edge involving b that we can remove
                    b_edges = []
                    for x in range(n):
                        if x != b and ((current_adj[b] >> x) & 1) and (min(b,x), max(b,x)) not in core_edge_set:
                            b_edges.append((min(b,x), max(b,x)))
                    if b_edges:
                        ru, rv = random.choice(b_edges)
                        new_adj = current_adj[:]
                        new_adj[ru] &= ~(1 << rv)
                        new_adj[rv] &= ~(1 << ru)
                        # Now try to add (u,v)
                        if is_triangle_free_add(new_adj, u, v):
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                        new_count = edge_count(new_adj)
                        delta = new_count - current_count
                        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                            current_adj = new_adj
                            current_count = new_count
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = current_adj[:]
        
        elif move_type < 0.85 and len(present_non_core) > 0:
            # Try to remove a random present non-core edge
            u, v = random.choice(present_non_core)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_count = edge_count(new_adj)
            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current_adj = new_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
        
        else:
            # Swap: remove one present non-core, add one missing non-core
            if len(present_non_core) > 0 and len(missing_non_core) > 0:
                ru, rv = random.choice(present_non_core)
                au, av = random.choice(missing_non_core)
                new_adj = current_adj[:]
                new_adj[ru] &= ~(1 << rv)
                new_adj[rv] &= ~(1 << ru)
                if is_triangle_free_add(new_adj, au, av):
                    new_adj[au] |= (1 << av)
                    new_adj[av] |= (1 << au)
                new_count = edge_count(new_adj)
                delta = new_count - current_count
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current_adj = new_adj
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
        
        # Periodically close the graph to find more edges
        if step % 5000 == 0 and step > 0:
            closed = close_graph(current_adj)
            closed_count = edge_count(closed)
            if closed_count > current_count:
                current_adj = closed
                current_count = closed_count
            if closed_count > best_count:
                best_count = closed_count
                best_adj = closed
    
    # Final closure
    final_adj = close_graph(best_adj)
    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj
    
    return edges_from_adj(best_adj)
